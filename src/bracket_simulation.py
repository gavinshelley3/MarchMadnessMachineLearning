"""Monte Carlo simulation utilities for NCAA tournament brackets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .bracket_generation import (
    GameResult,
    PredictionLookup,
    TeamState,
    get_deterministic_round_results,
    ROUND_ORDER_INDEX,
    ROUND_SEQUENCE,
)
from .bracket_loader import BracketDefinition, TeamSlot, describe_bracket
from .config import get_config


ADVANCEMENT_MAP = {
    "R64": "round_of_32",
    "R32": "sweet_16",
    "Sweet16": "elite_8",
    "Elite8": "final_four",
}

ROUND_OUTPUTS = [
    ("round_of_32", "Round of 32"),
    ("sweet_16", "Sweet 16"),
    ("elite_8", "Elite 8"),
    ("final_four", "Final Four"),
    ("championship_game", "Championship Game"),
    ("champion", "Champion"),
]

CONFIDENCE_THRESHOLDS = {
    "high": 0.75,
    "medium": 0.60,
}


@dataclass
class SimulationPaths:
    probabilities: Path
    summary: Path
    confidence: Path
    upset_report: Optional[Path]


class AdvancementTracker:
    """Accumulates advancement counts for every team across simulations."""

    def __init__(self, team_metadata: Dict[str, Dict[str, object]], n_sims: int) -> None:
        self.team_metadata = team_metadata
        self.n_sims = n_sims
        self.counts: Dict[str, Dict[str, int]] = {
            key: {round_key: 0 for round_key, _ in ROUND_OUTPUTS} for key in team_metadata
        }
        self.final_four_combos: Counter[Tuple[str, ...]] = Counter()
        self.championship_matchups: Counter[Tuple[str, str]] = Counter()

    def record_advancement(self, team_key: str, stage: str) -> None:
        if stage not in self.counts[team_key]:
            return
        self.counts[team_key][stage] += 1

    def record_final_four_combo(self, champions: Dict[str, TeamState]) -> None:
        combo = tuple(sorted(team.team_key for team in champions.values()))
        self.final_four_combos[combo] += 1

    def record_championship_matchup(self, team_a: TeamState, team_b: TeamState) -> None:
        pair = tuple(sorted((team_a.team_key, team_b.team_key)))
        self.championship_matchups[pair] += 1

    def probabilities_df(self) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for team_key, meta in self.team_metadata.items():
            row: Dict[str, object] = {
                "team": meta["display_name"],
                "team_key": team_key,
                "region": meta["region"],
                "seed": meta["seed"],
            }
            for key, _ in ROUND_OUTPUTS:
                row[f"{key}_probability"] = self.counts[team_key][key] / self.n_sims
            rows.append(row)
        df = pd.DataFrame(rows)
        df.sort_values("champion_probability", ascending=False, inplace=True)
        return df

    def top_final_four(self, limit: int = 3) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for combo, freq in self.final_four_combos.most_common(limit):
            results.append(
                {
                    "teams": [self.team_metadata[key]["display_name"] for key in combo],
                    "probability": freq / self.n_sims,
                }
            )
        return results

    def top_championship_matchups(self, limit: int = 5) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for pair, freq in self.championship_matchups.most_common(limit):
            results.append(
                {
                    "teams": [self.team_metadata[key]["display_name"] for key in pair],
                    "probability": freq / self.n_sims,
                }
            )
        return results

    def get_probabilities(self, team_key: str) -> Dict[str, float]:
        return {
            key: self.counts[team_key][key] / self.n_sims
            for key, _ in ROUND_OUTPUTS
        }


class BracketSimulator:
    """Runs Monte Carlo tournaments based on saved matchup probabilities."""

    def __init__(
        self,
        bracket_def: BracketDefinition,
        predictions_path: Path,
        n_sims: int = 5000,
        seed: int = 123,
        output_dir: Optional[Path] = None,
        label_suffix: Optional[str] = None,
    ):
        self.bracket_def = bracket_def
        self.predictions_path = predictions_path
        self.n_sims = n_sims
        self.seed = seed
        self.config = get_config()
        self.output_dir = output_dir or (self.config.paths.outputs_dir / "brackets")
        self._label_suffix = _sanitize_label(label_suffix)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.lookup = PredictionLookup(predictions_path, bracket_def.season)
        self.rng = np.random.default_rng(seed)
        self.region_matchups, self.team_metadata = self._prepare_matchups(bracket_def)

    def run(self) -> SimulationPaths:
        tracker = AdvancementTracker(self.team_metadata, self.n_sims)
        for _ in range(self.n_sims):
            region_winners = self._simulate_regions(tracker)
            tracker.record_final_four_combo(region_winners)
            finalists = self._simulate_final_four(region_winners, tracker)
            tracker.record_championship_matchup(finalists[0], finalists[1])
            self._simulate_championship(finalists, tracker)

        deterministic_results = get_deterministic_round_results(self.bracket_def, self.predictions_path)
        confidence_rows, tier_counts = self._build_confidence_rows(deterministic_results)
        upset_df = self._build_upset_report(tracker)
        return self._write_outputs(tracker, confidence_rows, tier_counts, upset_df, deterministic_results)

    def _prepare_matchups(
        self, bracket_def: BracketDefinition
    ) -> Tuple[Dict[str, List[Tuple[TeamState, TeamState]]], Dict[str, Dict[str, object]]]:
        matchups: Dict[str, List[Tuple[TeamState, TeamState]]] = {}
        metadata: Dict[str, Dict[str, object]] = {}
        for region in bracket_def.regions:
            region_pairs: List[Tuple[TeamState, TeamState]] = []
            for matchup in region.round_of_64:
                team1 = self._build_team_state(region.name, matchup.slot, matchup.team1)
                team2 = self._build_team_state(region.name, matchup.slot, matchup.team2)
                for team in (team1, team2):
                    metadata.setdefault(
                        team.team_key,
                        {
                            "display_name": team.display_name,
                            "region": team.region,
                            "seed": team.seed,
                        },
                    )
                region_pairs.append((team1, team2))
            matchups[region.name] = region_pairs
        return matchups, metadata

    def _build_team_state(self, region: str, slot_name: str, team_slot: TeamSlot) -> TeamState:
        team_id = self.lookup.team_id(team_slot.team_key)
        return TeamState(
            team_key=team_slot.team_key,
            display_name=team_slot.display_name,
            seed=team_slot.seed,
            region=region,
            source_slot=slot_name,
            team_id=team_id,
            options=team_slot.options,
        )

    def _simulate_regions(self, tracker: AdvancementTracker) -> Dict[str, TeamState]:
        winners: Dict[str, TeamState] = {}
        for region_name, matchups in self.region_matchups.items():
            current = []
            for team1, team2 in matchups:
                winner, _ = self._sample_game(team1, team2)
                tracker.record_advancement(winner.team_key, "round_of_32")
                current.append(winner)
            for round_name in ["R32", "Sweet16", "Elite8"]:
                if len(current) == 1:
                    break
                next_round = []
                for idx in range(0, len(current), 2):
                    winner, _ = self._sample_game(current[idx], current[idx + 1])
                    tracker.record_advancement(winner.team_key, ADVANCEMENT_MAP[round_name])
                    next_round.append(winner)
                current = next_round
            if len(current) != 1:
                raise ValueError(f"Region {region_name} failed to resolve to a champion.")
            winners[region_name] = current[0]
        return winners

    def _simulate_final_four(
        self, region_winners: Dict[str, TeamState], tracker: AdvancementTracker
    ) -> List[TeamState]:
        semifinal_winners: List[TeamState] = []
        for idx, pairing in enumerate(self.bracket_def.final_four_pairings, start=1):
            team_a = region_winners[pairing.regions[0]]
            team_b = region_winners[pairing.regions[1]]
            winner, _ = self._sample_game(team_a, team_b)
            tracker.record_advancement(winner.team_key, "championship_game")
            semifinal_winners.append(winner)
        if len(semifinal_winners) != 2:
            raise ValueError("Final Four must yield exactly two finalists.")
        return semifinal_winners

    def _simulate_championship(
        self, finalists: List[TeamState], tracker: AdvancementTracker
    ) -> None:
        champion, _ = self._sample_game(finalists[0], finalists[1])
        tracker.record_advancement(champion.team_key, "champion")

    def _sample_game(self, team1: TeamState, team2: TeamState) -> Tuple[TeamState, TeamState]:
        prob_team1, prob_team2 = self.lookup.probability(team1.team_key, team2.team_key)
        roll = self.rng.random()
        if roll < prob_team1:
            return team1, team2
        return team2, team1

    def _build_confidence_rows(
        self, deterministic_results: Tuple[Dict[str, List[GameResult]], TeamState, PredictionLookup, List[Dict[str, object]]]
    ) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
        round_results, _, _, _ = deterministic_results
        rows: List[Dict[str, object]] = []
        tier_counts: Counter[str] = Counter()
        for round_name, games in round_results.items():
            for game in games:
                prob = game.winner_probability
                tier = classify_confidence(prob)
                tier_counts[tier] += 1
                note = "Upset pick" if game.winner.seed > game.loser.seed else ""
                rows.append(
                    {
                        "round": round_name,
                        "round_label": dict(ROUND_SEQUENCE).get(round_name, round_name),
                        "region": game.region or "National",
                        "slot": game.slot,
                        "team1": game.team1.display_name,
                        "team1_seed": game.team1.seed,
                        "team2": game.team2.display_name,
                        "team2_seed": game.team2.seed,
                        "deterministic_winner": game.winner.display_name,
                        "winner_seed": game.winner.seed,
                        "winner_probability": prob,
                        "confidence_tier": tier,
                        "note": note,
                    }
                )
        rows.sort(key=lambda row: (ROUND_ORDER_INDEX.get(row["round"], 99), row["region"], row["slot"]))
        return rows, dict(tier_counts)

    def _build_upset_report(self, tracker: AdvancementTracker) -> Optional[pd.DataFrame]:
        rows: List[Dict[str, object]] = []
        for team_key, meta in tracker.team_metadata.items():
            probs = tracker.get_probabilities(team_key)
            seed = int(meta["seed"])
            display_name = meta["display_name"]
            region = meta["region"]
            if seed >= 9 and probs["round_of_32"] >= 0.30:
                rows.append(
                    {
                        "team": display_name,
                        "seed": seed,
                        "region": region,
                        "focus_round": "Round of 32",
                        "probability": probs["round_of_32"],
                        "note": "Lower-seed upset potential",
                    }
                )
            if seed <= 4 and probs["round_of_32"] <= 0.65:
                rows.append(
                    {
                        "team": display_name,
                        "seed": seed,
                        "region": region,
                        "focus_round": "Round of 32",
                        "probability": 1 - probs["round_of_32"],
                        "note": "Favorite with early upset risk",
                    }
                )
            if seed >= 6 and probs["final_four"] >= 0.08:
                rows.append(
                    {
                        "team": display_name,
                        "seed": seed,
                        "region": region,
                        "focus_round": "Final Four",
                        "probability": probs["final_four"],
                        "note": "Dark-horse Final Four path",
                    }
                )
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.sort_values("probability", ascending=False, inplace=True)
        return df

    def _write_outputs(
        self,
        tracker: AdvancementTracker,
        confidence_rows: List[Dict[str, object]],
        tier_counts: Dict[str, int],
        upset_df: Optional[pd.DataFrame],
        deterministic_results: Tuple[Dict[str, List[GameResult]], TeamState, PredictionLookup, List[Dict[str, object]]],
    ) -> SimulationPaths:
        prefix = f"{self.bracket_def.season}_{self.bracket_def.bracket_type}".replace(" ", "_").lower()
        if self._label_suffix:
            prefix = f"{prefix}{self._label_suffix}"
        probabilities_path = self.output_dir / f"{prefix}_simulation_team_probabilities.csv"
        summary_path = self.output_dir / f"{prefix}_simulation_summary.json"
        confidence_path = self.output_dir / f"{prefix}_pick_confidence.csv"
        upset_path = self.output_dir / f"{prefix}_upset_risk_report.csv"

        tracker.probabilities_df().to_csv(probabilities_path, index=False)
        pd.DataFrame(confidence_rows).to_csv(confidence_path, index=False)
        if upset_df is not None:
            upset_df.to_csv(upset_path, index=False)
        else:
            upset_path = None

        summary_payload = build_summary(
            tracker=tracker,
            tier_counts=tier_counts,
            deterministic_results=deterministic_results,
            n_sims=self.n_sims,
            seed=self.seed,
            predictions_path=self.predictions_path,
            bracket_def=self.bracket_def,
        )
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        return SimulationPaths(
            probabilities=probabilities_path,
            summary=summary_path,
            confidence=confidence_path,
            upset_report=upset_path,
        )


def _sanitize_label(label: Optional[str]) -> str:
    if not label:
        return ""
    slug = str(label).strip().lower().replace(" ", "_")
    return f"_{slug}" if slug else ""


def classify_confidence(probability: float) -> str:
    if probability >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if probability >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def build_summary(
    tracker: AdvancementTracker,
    tier_counts: Dict[str, int],
    deterministic_results: Tuple[Dict[str, List[GameResult]], TeamState, PredictionLookup, List[Dict[str, object]]],
    n_sims: int,
    seed: int,
    predictions_path: Path,
    bracket_def: BracketDefinition,
) -> Dict[str, object]:
    round_results, champion_state, lookup, _ = deterministic_results
    prob_df = tracker.probabilities_df()
    top_contenders = prob_df[["team", "champion_probability"]].head(5).to_dict(orient="records")
    final_four_combos = tracker.top_final_four()
    champ_matchups = tracker.top_championship_matchups()
    most_likely_champion = top_contenders[0] if top_contenders else {}
    most_common_final_four = final_four_combos[0] if final_four_combos else {}
    most_common_championship = champ_matchups[0] if champ_matchups else {}

    return {
        "metadata": {
            **describe_bracket(bracket_def),
            "source_note": bracket_def.source_note,
            "updated_at": bracket_def.updated_at,
            "bracket_file": str(bracket_def.bracket_file),
        },
        "simulations": {
            "n_sims": n_sims,
            "seed": seed,
            "prediction_file": str(predictions_path),
            "model_name": lookup.primary_model,
            "feature_set": lookup.primary_feature_set,
            "confidence_thresholds": CONFIDENCE_THRESHOLDS,
            "confidence_counts": tier_counts,
        },
        "deterministic": {
            "champion": champion_state.display_name,
            "bracket_highlights": {
                "final_four_games": len(round_results.get("FinalFour", [])),
                "championship_matchup": [
                    round_results["Championship"][0].team1.display_name,
                    round_results["Championship"][0].team2.display_name,
                ]
                if round_results.get("Championship")
                else [],
            },
        },
        "most_likely_champion": most_likely_champion,
        "most_common_final_four": most_common_final_four,
        "most_common_championship_matchup": most_common_championship,
        "top_championship_contenders": top_contenders,
        "championship_matchup_distribution": champ_matchups,
        "final_four_distribution": final_four_combos,
    }
