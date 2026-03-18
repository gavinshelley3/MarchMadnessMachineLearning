"""Utilities for loading structured bracket input files."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class TeamSlot:
    """Represents a team entry within a bracket slot."""

    team_key: str
    display_name: str
    seed: int
    options: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "TeamSlot":
        options = payload.get("options", []) or []
        team_key = payload.get("team_key")
        if not team_key:
            if options:
                team_key = options[0].get("team_key")
            if not team_key:
                raise ValueError("Each team entry must define 'team_key' or provide options with one.")
        display_name = payload.get("display_name", team_key)
        placeholder = str(team_key).lower()
        if options and (placeholder.endswith("playin") or placeholder in {"tbd"}):
            primary = options[0]
            primary_key = primary.get("team_key")
            primary_name = primary.get("display_name", primary_key)
            if primary_key:
                team_key = primary_key
                display_name = primary_name or primary_key
        seed_raw = payload.get("seed")
        if seed_raw is None:
            raise ValueError(f"Team entry for {display_name} is missing a seed value.")
        return cls(team_key=str(team_key), display_name=str(display_name), seed=int(seed_raw), options=options)


@dataclass
class MatchupSlot:
    """One game slot inside a region/round."""

    slot: str
    round_name: str
    region: str
    team1: TeamSlot
    team2: TeamSlot

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], region_name: str) -> "MatchupSlot":
        slot = payload.get("slot")
        if not slot:
            raise ValueError(f"Matchup in region {region_name} is missing a slot identifier.")
        round_name = payload.get("round", "R64")
        team1 = TeamSlot.from_payload(payload.get("team1", {}))
        team2 = TeamSlot.from_payload(payload.get("team2", {}))
        return cls(slot=str(slot), round_name=str(round_name), region=region_name, team1=team1, team2=team2)


@dataclass
class RegionDefinition:
    name: str
    locations: List[str]
    round_of_64: List[MatchupSlot]

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "RegionDefinition":
        name = payload.get("name")
        if not name:
            raise ValueError("Each region requires a name.")
        locations = payload.get("locations", [])
        round_payload = payload.get("round_of_64") or payload.get("games")
        if not isinstance(round_payload, list):
            raise ValueError(f"Region {name} must include a 'round_of_64' list of games.")
        games = [MatchupSlot.from_payload(item, name) for item in round_payload]
        return cls(name=str(name), locations=[str(loc) for loc in locations], round_of_64=games)


@dataclass
class FinalFourPairing:
    regions: List[str]
    name: str

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "FinalFourPairing":
        regions = payload.get("regions")
        if not regions or len(regions) != 2:
            raise ValueError("Final Four pairings must specify exactly two regions.")
        name = payload.get("name") or payload.get("slot") or "semifinal"
        return cls(regions=[str(r) for r in regions], name=str(name))


@dataclass
class BracketDefinition:
    season: int
    bracket_type: str
    source_note: str
    bracket_file: Path
    updated_at: Optional[str]
    regions: List[RegionDefinition]
    final_four_pairings: List[FinalFourPairing]

    def iter_round_of_64(self) -> Iterable[MatchupSlot]:
        for region in self.regions:
            for matchup in region.round_of_64:
                yield matchup


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_bracket_definition(path: Path) -> BracketDefinition:
    """Parse and validate a structured bracket file."""

    payload = _load_json(path)
    season = payload.get("season")
    bracket_type = payload.get("bracket_type", "projected")
    if season is None:
        raise ValueError("Bracket file must include a 'season' value.")
    regions_payload = payload.get("regions")
    if not isinstance(regions_payload, list) or not regions_payload:
        raise ValueError("Bracket file must define at least one region.")
    regions = [RegionDefinition.from_payload(item) for item in regions_payload]
    _ensure_unique_regions(regions)
    final_four_payload = payload.get("final_four_pairings", [])
    final_four = [FinalFourPairing.from_payload(item) for item in final_four_payload]
    if not final_four:
        raise ValueError("Bracket file must describe Final Four pairings so semifinal matchups are deterministic.")
    return BracketDefinition(
        season=int(season),
        bracket_type=str(bracket_type),
        source_note=str(payload.get("source_note", "")),
        bracket_file=path,
        updated_at=payload.get("updated_at"),
        regions=regions,
        final_four_pairings=final_four,
    )


def _ensure_unique_regions(regions: List[RegionDefinition]) -> None:
    seen = set()
    for region in regions:
        if region.name in seen:
            raise ValueError(f"Duplicate region name detected: {region.name}")
        seen.add(region.name)


def describe_bracket(definition: BracketDefinition) -> Dict[str, Any]:
    """Return a lightweight dict description for diagnostics/logging."""

    return {
        "season": definition.season,
        "bracket_type": definition.bracket_type,
        "regions": {region.name: len(region.round_of_64) for region in definition.regions},
        "final_four_pairings": [pair.regions for pair in definition.final_four_pairings],
    }
