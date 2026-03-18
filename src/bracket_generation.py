"""Bracket generation logic that consumes saved matchup predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from . import data_loading
from .bracket_loader import BracketDefinition, MatchupSlot, TeamSlot, describe_bracket
from .config import get_config


ROUND_SEQUENCE = [
    ("R64", "Round of 64"),
    ("R32", "Round of 32"),
    ("Sweet16", "Sweet 16"),
    ("Elite8", "Elite 8"),
    ("FinalFour", "Final Four"),
    ("Championship", "Championship"),
]
ROUND_ORDER_INDEX = {name: idx for idx, (name, _) in enumerate(ROUND_SEQUENCE)}


@dataclass
class TeamState:
    """Tracks a team's metadata as it advances through the bracket."""

    team_key: str
    display_name: str
    seed: int
    region: str
    source_slot: str
    team_id: Optional[int] = None
    options: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Optional[object]]:
        return {
            "team_key": self.team_key,
            "display_name": self.display_name,
            "seed": self.seed,
            "region": self.region,
            "source_slot": self.source_slot,
            "team_id": self.team_id,
            "options": self.options,
        }


@dataclass
class GameResult:
    round_name: str
    slot: str
    region: Optional[str]
    team1: TeamState
    team2: TeamState
    prob_team1: float
    prob_team2: float
    winner: TeamState
    loser: TeamState

    @property
    def winner_probability(self) -> float:
        return self.prob_team1 if self.winner is self.team1 else self.prob_team2

    @property
    def loser_probability(self) -> float:
        return self.prob_team2 if self.winner is self.team1 else self.prob_team1

    @property
    def is_upset(self) -> bool:
        return self.winner.seed > self.loser.seed

    def to_row(self, model_name: str, feature_set: str) -> Dict[str, object]:
        return {
            "round": self.round_name,
            "round_label": dict(ROUND_SEQUENCE).get(self.round_name, self.round_name),
            "region": self.region or "National",
            "slot": self.slot,
            "team1": self.team1.display_name,
            "team1_seed": self.team1.seed,
            "team1_key": self.team1.team_key,
            "team2": self.team2.display_name,
            "team2_seed": self.team2.seed,
            "team2_key": self.team2.team_key,
            "prob_team1": self.prob_team1,
            "prob_team2": self.prob_team2,
            "winner": self.winner.display_name,
            "winner_seed": self.winner.seed,
            "winner_probability": self.winner_probability,
            "loser": self.loser.display_name,
            "loser_seed": self.loser.seed,
            "model_name": model_name,
            "feature_set": feature_set,
        }

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "round": self.round_name,
            "slot": self.slot,
            "region": self.region,
            "team1": self.team1.to_dict(),
            "team2": self.team2.to_dict(),
            "prob_team1": self.prob_team1,
            "prob_team2": self.prob_team2,
            "winner": self.winner.to_dict(),
            "loser": self.loser.to_dict(),
        }


class PredictionLookup:
    """Caches matchup probabilities from the saved prediction file."""

    def __init__(self, predictions_path: Path, season: int):
        self.predictions_path = predictions_path
        self.season = season
        self._lookup_by_ids: Dict[Tuple[int, int], Tuple[float, float]] = {}
        usecols = [
            "Season",
            "Team1Name",
            "Team2Name",
            "PredTeam1WinProb",
            "PredTeam2WinProb",
            "Team1ID",
            "Team2ID",
            "model_name",
            "feature_set",
        ]
        df = pd.read_csv(predictions_path, usecols=usecols, low_memory=False)
        season_df = df[df["Season"] == season]
        if season_df.empty:
            raise ValueError(
                f"No predictions for season {season} were found in {predictions_path}."
            )
        self._normalized_name_lookup = _build_team_name_lookup()
        self.model_names = _sorted_unique(season_df["model_name"])
        self.feature_sets = _sorted_unique(season_df["feature_set"])
        self._lookup: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._team_ids: Dict[str, Optional[int]] = {}
        for row in season_df.itertuples(index=False):
            if not row.Team1Name or not row.Team2Name:
                continue
            key = (str(row.Team1Name), str(row.Team2Name))
            p1 = float(row.PredTeam1WinProb)
            p2 = float(row.PredTeam2WinProb) if pd.notna(row.PredTeam2WinProb) else 1.0 - p1
            total = p1 + p2
            if total <= 0:
                continue
            if abs(total - 1.0) > 1e-6:
                p1 = p1 / total
                p2 = p2 / total
            self._lookup[key] = (p1, p2)
            self._team_ids.setdefault(str(row.Team1Name), _safe_int(row.Team1ID))
            self._team_ids.setdefault(str(row.Team2Name), _safe_int(row.Team2ID))
            team1_id = _safe_int(row.Team1ID)
            team2_id = _safe_int(row.Team2ID)
            if team1_id is not None and team2_id is not None:
                self._lookup_by_ids[(team1_id, team2_id)] = (p1, p2)
                self._lookup_by_ids[(team2_id, team1_id)] = (p2, p1)
                norm1 = _normalize_name(str(row.Team1Name))
                norm2 = _normalize_name(str(row.Team2Name))
                self._normalized_name_lookup.setdefault(norm1, team1_id)
                self._normalized_name_lookup.setdefault(norm2, team2_id)

    @property
    def primary_model(self) -> str:
        return self.model_names[0] if self.model_names else "unknown"

    @property
    def primary_feature_set(self) -> str:
        return self.feature_sets[0] if self.feature_sets else "unknown"

    def probability(self, team_a: str, team_b: str) -> Tuple[float, float]:
        key = (team_a, team_b)
        if key in self._lookup:
            return self._lookup[key]
        reverse_key = (team_b, team_a)
        if reverse_key in self._lookup:
            p_b, p_a = self._lookup[reverse_key]
            return p_a, p_b
        id_a = self._resolve_team_id(team_a)
        id_b = self._resolve_team_id(team_b)
        if id_a is not None and id_b is not None:
            lookup_key = (id_a, id_b)
            if lookup_key in self._lookup_by_ids:
                return self._lookup_by_ids[lookup_key]
        raise KeyError(f"No prediction entry for matchup {team_a} vs {team_b}.")

    def team_id(self, team_name: str) -> Optional[int]:
        existing = self._team_ids.get(team_name)
        if existing is not None:
            return existing
        return self._resolve_team_id(team_name)

    def _resolve_team_id(self, team_name: str) -> Optional[int]:
        if not team_name:
            return None
        direct = self._team_ids.get(team_name)
        if direct is not None:
            return direct
        normalized = _normalize_name(team_name)
        return self._normalized_name_lookup.get(normalized)


SelectionPolicy = Callable[
    [TeamState, TeamState, float, float, str, str, Optional[str]],
    TeamState,
]


class BracketGenerator:
    """Deterministically generates a bracket using saved predictions."""

    def __init__(
        self,
        predictions_path: Path,
        output_dir: Optional[Path] = None,
        ensure_dirs: bool = True,
        selection_policy: Optional[SelectionPolicy] = None,
    ):
        self.predictions_path = predictions_path
        config = get_config()
        default_dir = config.paths.outputs_dir / "brackets"
        self.output_dir = output_dir or default_dir
        if ensure_dirs:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.selection_policy = selection_policy

    def compute_round_results(
        self,
        bracket_def: BracketDefinition,
        season: Optional[int] = None,
        lookup: Optional[PredictionLookup] = None,
    ) -> Tuple[Dict[str, List[GameResult]], TeamState, PredictionLookup, List[Dict[str, object]]]:
        season_to_use = season or bracket_def.season
        lookup_instance = lookup or PredictionLookup(self.predictions_path, season_to_use)
        round_results: Dict[str, List[GameResult]] = {name: [] for name, _ in ROUND_SEQUENCE}
        unresolved_slots = list(_collect_unresolved_slots(bracket_def))
        region_champions: Dict[str, TeamState] = {}

        for region in bracket_def.regions:
            champion = self._run_region(region.name, region.round_of_64, lookup_instance, round_results)
            region_champions[region.name] = champion

        semifinal_winners: List[TeamState] = []
        for pairing_idx, pairing in enumerate(bracket_def.final_four_pairings, start=1):
            region_a, region_b = pairing.regions
            if region_a not in region_champions or region_b not in region_champions:
                raise ValueError(f"Missing regional champion for Final Four pairing {pairing.regions}.")
            slot_name = pairing.name or f"Semifinal {pairing_idx}"
            result = self._play_game(
                region_champions[region_a],
                region_champions[region_b],
                round_name="FinalFour",
                slot=slot_name,
                region="National",
                lookup=lookup_instance,
            )
            round_results["FinalFour"].append(result)
            semifinal_winners.append(result.winner)

        if len(semifinal_winners) != 2:
            raise ValueError("Final Four phase must return exactly two semifinal winners.")
        championship = self._play_game(
            semifinal_winners[0],
            semifinal_winners[1],
            round_name="Championship",
            slot="Title",
            region="National",
            lookup=lookup_instance,
        )
        round_results["Championship"].append(championship)
        return round_results, championship.winner, lookup_instance, unresolved_slots

    def generate(
        self,
        bracket_def: BracketDefinition,
        season: Optional[int] = None,
        label_suffix: Optional[str] = None,
        lookup: Optional[PredictionLookup] = None,
    ) -> Dict[str, Path]:
        round_results, champion, lookup_instance, unresolved_slots = self.compute_round_results(
            bracket_def, season, lookup=lookup
        )
        writer = _BracketWriter(
            output_dir=self.output_dir,
            bracket_definition=bracket_def,
            lookup=lookup_instance,
            unresolved_slots=unresolved_slots,
            label_suffix=label_suffix,
        )
        return writer.write(round_results, champion=champion)

    def _run_region(
        self,
        region_name: str,
        matchups: Iterable[MatchupSlot],
        lookup: PredictionLookup,
        round_results: Dict[str, List[GameResult]],
    ) -> TeamState:
        matchups = list(matchups)
        winners: List[TeamState] = []
        for matchup in matchups:
            team1 = self._team_state(region_name, matchup.slot, matchup.team1, lookup)
            team2 = self._team_state(region_name, matchup.slot, matchup.team2, lookup)
            result = self._play_game(
                team1,
                team2,
                round_name=matchup.round_name or "R64",
                slot=matchup.slot,
                region=region_name,
                lookup=lookup,
            )
            round_results["R64"].append(result)
            winners.append(result.winner)
        next_round_names = ["R32", "Sweet16", "Elite8"]
        for round_name in next_round_names:
            if len(winners) == 1:
                break
            winners = self._advance_round(region_name, round_name, winners, lookup, round_results)
        if len(winners) != 1:
            raise ValueError(f"Region {region_name} did not resolve to a single champion.")
        return winners[0]

    def _advance_round(
        self,
        region_name: str,
        round_name: str,
        teams: List[TeamState],
        lookup: PredictionLookup,
        round_results: Dict[str, List[GameResult]],
    ) -> List[TeamState]:
        if len(teams) % 2 != 0:
            raise ValueError(f"Round {round_name} in region {region_name} has an uneven number of teams.")
        winners: List[TeamState] = []
        for idx in range(0, len(teams), 2):
            slot = f"{region_name}-{round_name}-{idx // 2 + 1}"
            result = self._play_game(
                teams[idx],
                teams[idx + 1],
                round_name=round_name,
                slot=slot,
                region=region_name,
                lookup=lookup,
            )
            round_results[round_name].append(result)
            winners.append(result.winner)
        return winners

    def _team_state(
        self,
        region: str,
        slot_name: str,
        team_slot: TeamSlot,
        lookup: PredictionLookup,
    ) -> TeamState:
        team_id = lookup.team_id(team_slot.team_key)
        return TeamState(
            team_key=team_slot.team_key,
            display_name=team_slot.display_name,
            seed=team_slot.seed,
            region=region,
            source_slot=slot_name,
            team_id=team_id,
            options=team_slot.options,
        )

    def _play_game(
        self,
        team1: TeamState,
        team2: TeamState,
        round_name: str,
        slot: str,
        region: Optional[str],
        lookup: PredictionLookup,
    ) -> GameResult:
        prob_team1, prob_team2 = lookup.probability(team1.team_key, team2.team_key)
        if self.selection_policy:
            selected = self.selection_policy(team1, team2, prob_team1, prob_team2, round_name, slot, region)
            if selected is team1:
                winner, loser = team1, team2
            elif selected is team2:
                winner, loser = team2, team1
            else:  # pragma: no cover - defensive guard
                raise ValueError("Selection policy must return one of the provided teams.")
        else:
            if prob_team1 >= prob_team2:
                winner, loser = team1, team2
            else:
                winner, loser = team2, team1
        return GameResult(
            round_name=round_name,
            slot=slot,
            region=region,
            team1=team1,
            team2=team2,
            prob_team1=prob_team1,
            prob_team2=prob_team2,
            winner=winner,
            loser=loser,
        )


def _sanitize_label(label: Optional[str]) -> str:
    if not label:
        return ""
    slug = str(label).strip().lower().replace(" ", "_")
    return f"_{slug}" if slug else ""


class _BracketWriter:
    """Handles serialization of generated bracket artifacts."""

    def __init__(
        self,
        output_dir: Path,
        bracket_definition: BracketDefinition,
        lookup: PredictionLookup,
        unresolved_slots: List[Dict[str, object]],
        label_suffix: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.bracket_definition = bracket_definition
        self.lookup = lookup
        self.unresolved_slots = unresolved_slots
        self.label_suffix = _sanitize_label(label_suffix)

    def write(self, rounds: Dict[str, List[GameResult]], champion: TeamState) -> Dict[str, Path]:
        prefix = f"{self.bracket_definition.season}_{self.bracket_definition.bracket_type}".replace(" ", "_").lower()
        suffix = self.label_suffix
        json_path = self.output_dir / f"{prefix}_bracket_results{suffix}.json"
        csv_path = self.output_dir / f"{prefix}_bracket_results{suffix}.csv"
        summary_path = self.output_dir / f"{prefix}_bracket_summary{suffix}.txt"
        upsets_path = self.output_dir / f"{prefix}_top_upsets{suffix}.csv"

        json_payload = self._build_json(rounds, champion)
        _write_json(json_path, json_payload)

        rows = self._build_rows(rounds)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        summary_text = self._build_summary(champion, rounds)
        summary_path.write_text(summary_text, encoding="utf-8")

        upsets = _collect_upsets(rows)
        if upsets:
            pd.DataFrame(upsets).to_csv(upsets_path, index=False)

        return {
            "json": json_path,
            "csv": csv_path,
            "summary": summary_path,
            "top_upsets": upsets_path if upsets else None,
        }

    def _build_rows(self, rounds: Dict[str, List[GameResult]]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for round_name, games in rounds.items():
            for game in games:
                rows.append(game.to_row(self.lookup.primary_model, self.lookup.primary_feature_set))
        rows.sort(key=lambda row: (ROUND_ORDER_INDEX.get(row["round"], 99), row["region"], row["slot"]))
        return rows

    def _build_json(self, rounds: Dict[str, List[GameResult]], champion: TeamState) -> Dict[str, object]:
        return {
            "metadata": {
                **describe_bracket(self.bracket_definition),
                "prediction_file": str(self.lookup.predictions_path),
                "model_name": self.lookup.primary_model,
                "feature_set": self.lookup.primary_feature_set,
                "unresolved_slots": self.unresolved_slots,
            },
            "rounds": {round_name: [game.to_json_dict() for game in games] for round_name, games in rounds.items()},
            "champion": champion.to_dict(),
        }

    def _build_summary(self, champion: TeamState, rounds: Dict[str, List[GameResult]]) -> str:
        lines = [
            f"Bracket: {self.bracket_definition.bracket_type.title()} {self.bracket_definition.season}",
            f"Model: {self.lookup.primary_model} ({self.lookup.primary_feature_set})",
            f"Champion: {champion.display_name} (Seed {champion.seed}, {champion.region})",
            "",
            "Final Four Results:",
        ]
        for game in rounds.get("FinalFour", []):
            lines.append(
                f"- {game.winner.display_name} over {game.loser.display_name} ({game.winner_probability:.3f})"
            )
        lines.append("")
        lines.append("Championship:")
        for game in rounds.get("Championship", []):
            lines.append(
                f"- {game.winner.display_name} over {game.loser.display_name} ({game.winner_probability:.3f})"
            )
        if self.unresolved_slots:
            lines.append("")
            lines.append("Unresolved play-in slots:")
            for slot in self.unresolved_slots:
                options = ", ".join(opt["display_name"] for opt in slot["options"])
                lines.append(
                    f"- {slot['region']} {slot['slot']} seed {slot['seed']} currently set to {slot['team_key']} (options: {options})"
                )
        return "\n".join(lines)


def get_deterministic_round_results(
    bracket_def: BracketDefinition,
    predictions_path: Path,
    season: Optional[int] = None,
) -> Tuple[Dict[str, List[GameResult]], TeamState, PredictionLookup, List[Dict[str, object]]]:
    generator = BracketGenerator(predictions_path, ensure_dirs=False)
    return generator.compute_round_results(bracket_def, season)


def _collect_upsets(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    upsets = []
    for row in rows:
        winner_seed = int(row["winner_seed"])
        loser_seed = int(row["loser_seed"])
        if winner_seed <= loser_seed:
            continue
        upsets.append({
            "round": row["round"],
            "region": row["region"],
            "winner": row["winner"],
            "winner_seed": winner_seed,
            "loser": row["loser"],
            "loser_seed": loser_seed,
            "probability": row["winner_probability"],
        })
    upsets.sort(key=lambda item: (-item["probability"], ROUND_ORDER_INDEX.get(item["round"], 99)))
    return upsets


def _collect_unresolved_slots(bracket: BracketDefinition) -> Iterable[Dict[str, object]]:
    for matchup in bracket.iter_round_of_64():
        for team_slot in (matchup.team1, matchup.team2):
            if team_slot.options:
                yield {
                    "region": matchup.region,
                    "slot": matchup.slot,
                    "seed": team_slot.seed,
                    "team_key": team_slot.team_key,
                    "options": team_slot.options,
                }


NAME_ALIASES = {
    "cabaptist": "calbaptist",
    "ohiostate": "ohiost",
    "miami": "miamifl",
    "mcneese": "mcneesest",
    "iowastate": "iowast",
    "utahstate": "utahst",
    "saintmarys": "stmarysca",
    "saintlouis": "stlouis",
    "longisland": "liubrooklyn",
    "queens": "queensnc",
    "kennesawst": "kennesaw",
    "uconn": "connecticut",
}


def _normalize_name(value: str) -> str:
    normalized = "".join(ch.lower() for ch in str(value) if ch.isalnum())
    return NAME_ALIASES.get(normalized, normalized)


def _build_team_name_lookup() -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    try:
        teams = data_loading.load_teams()
        for row in teams.itertuples(index=False):
            if pd.notna(row.TeamName):
                lookup[_normalize_name(row.TeamName)] = int(row.TeamID)
    except FileNotFoundError:
        pass
    try:
        spellings = data_loading.load_team_spellings()
        for row in spellings.itertuples(index=False):
            if pd.notna(row.TeamNameSpelling):
                lookup[_normalize_name(row.TeamNameSpelling)] = int(row.TeamID)
    except FileNotFoundError:
        pass
    return lookup


def _safe_int(value: object) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _sorted_unique(series: pd.Series) -> List[str]:
    values = [str(val) for val in series.dropna().unique() if str(val)]
    return sorted(values)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
