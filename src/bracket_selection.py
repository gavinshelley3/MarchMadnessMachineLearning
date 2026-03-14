"""Smarter bracket selection policies that reuse saved predictions and simulations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .bracket_generation import BracketGenerator, PredictionLookup, TeamState
from .bracket_loader import load_bracket_definition
from .bracket_simulation import CONFIDENCE_THRESHOLDS
from .bracket_view import BracketViewRenderer
from .config import get_config


ROUND_STAGE_MAP = {
    "R64": "round_of_32_probability",
    "R32": "sweet_16_probability",
    "Sweet16": "elite_8_probability",
    "Elite8": "final_four_probability",
    "FinalFour": "championship_game_probability",
    "Championship": "champion_probability",
}

DEFAULT_BRACKET_PATH = Path("data/brackets/projected_2026_bracket.json")


def _normalize_key(value: str) -> str:
    return str(value).strip().lower()


class BlendedPredictionLookup:
    """Runtime blending between baseline and enriched prediction lookups."""

    def __init__(
        self,
        baseline_path: Path,
        enriched_path: Path,
        season: int,
        baseline_weight: float = 0.6,
    ) -> None:
        if not 0.0 <= baseline_weight <= 1.0:
            raise ValueError("baseline_weight must be between 0 and 1.")
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline predictions file {baseline_path} not found.")
        if not enriched_path.exists():
            raise FileNotFoundError(f"Enriched predictions file {enriched_path} not found.")
        self.baseline_lookup = PredictionLookup(baseline_path, season)
        self.enriched_lookup = PredictionLookup(enriched_path, season)
        self.baseline_weight = baseline_weight
        self.enriched_weight = 1.0 - baseline_weight
        suffix = f"blend_{baseline_path.stem}_{enriched_path.stem}"
        self.predictions_path = baseline_path.parent / f"{suffix}.csv"
        self.model_names = [
            f"blend({self.baseline_lookup.primary_model},{self.enriched_lookup.primary_model})"
        ]
        self.feature_sets = [
            f"blend({self.baseline_lookup.primary_feature_set},{self.enriched_lookup.primary_feature_set})"
        ]

    @property
    def primary_model(self) -> str:
        return self.model_names[0]

    @property
    def primary_feature_set(self) -> str:
        return self.feature_sets[0]

    def team_id(self, team_name: str) -> Optional[int]:
        return self.baseline_lookup.team_id(team_name) or self.enriched_lookup.team_id(team_name)

    def probability(self, team_a: str, team_b: str) -> Tuple[float, float]:
        base_a, base_b = self.baseline_lookup.probability(team_a, team_b)
        enr_a, enr_b = self.enriched_lookup.probability(team_a, team_b)
        p1 = self.baseline_weight * base_a + self.enriched_weight * enr_a
        p2 = self.baseline_weight * base_b + self.enriched_weight * enr_b
        total = p1 + p2
        if total <= 0:
            raise ValueError(f"Blended probabilities for {team_a} vs {team_b} are invalid.")
        return p1 / total, p2 / total


class SimulationProbabilities:
    """Helper to look up advancement probabilities from simulation outputs."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Simulation probabilities file {path} not found.")
        df = pd.read_csv(path)
        if "team_key" not in df.columns:
            raise ValueError(f"Simulation probabilities file {path} must include 'team_key'.")
        self.path = path
        self.rows: Dict[str, Dict[str, float]] = {}
        for record in df.to_dict(orient="records"):
            key = _normalize_key(record.get("team_key") or record.get("team"))
            if not key:
                continue
            self.rows[key] = {k: float(v) for k, v in record.items() if isinstance(v, (int, float))}

    def stage_probability(self, team_key: str, round_name: str) -> Optional[float]:
        column = ROUND_STAGE_MAP.get(round_name)
        if not column:
            return None
        entry = self.rows.get(_normalize_key(team_key))
        if not entry:
            return None
        return entry.get(column)


class AdvancementProbabilities:
    """Lookup wrapper around advancement NN probabilities."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Advancement probabilities file {path} not found.")
        df = pd.read_csv(path)
        if "team_key" not in df.columns and "TeamName" not in df.columns:
            raise ValueError(
                f"Advancement probabilities file {path} must include a 'team_key' or 'TeamName' column."
            )
        self.path = path
        self.rows: Dict[str, Dict[str, float]] = {}
        for record in df.to_dict(orient="records"):
            key = _normalize_key(record.get("team_key") or record.get("TeamName"))
            if not key:
                continue
            numeric_fields = {k: float(v) for k, v in record.items() if isinstance(v, (int, float))}
            self.rows[key] = numeric_fields

    def stage_probability(self, team_key: str, round_name: str) -> Optional[float]:
        column = ROUND_STAGE_MAP.get(round_name)
        if not column:
            return None
        entry = self.rows.get(_normalize_key(team_key))
        if not entry:
            return None
        return entry.get(column)


class SimulationInformedPolicy:
    """Selection policy that consults advancement probabilities for close games."""

    def __init__(
        self,
        simulation_probs: SimulationProbabilities,
        close_margin: float = 0.12,
        stage_weight: float = 0.65,
    ) -> None:
        self.simulation_probs = simulation_probs
        self.close_margin = max(0.0, close_margin)
        self.stage_weight = min(0.95, max(0.0, stage_weight))
        self.high_conf_threshold = CONFIDENCE_THRESHOLDS["high"]

    def __call__(
        self,
        team1: TeamState,
        team2: TeamState,
        prob_team1: float,
        prob_team2: float,
        round_name: str,
        slot: str,
        region: Optional[str],
    ) -> TeamState:
        del slot, region  # unused in policy
        margin = abs(prob_team1 - prob_team2)
        favorite = team1 if prob_team1 >= prob_team2 else team2
        underdog = team2 if favorite is team1 else team1
        favorite_prob = max(prob_team1, prob_team2)

        if favorite_prob >= self.high_conf_threshold or margin >= self.close_margin:
            return favorite

        stage_prob_fav = self.simulation_probs.stage_probability(favorite.team_key, round_name)
        stage_prob_under = self.simulation_probs.stage_probability(underdog.team_key, round_name)
        if stage_prob_fav is None or stage_prob_under is None:
            return favorite

        fav_score = (1 - self.stage_weight) * favorite_prob + self.stage_weight * stage_prob_fav
        under_score = (1 - self.stage_weight) * (1 - favorite_prob) + self.stage_weight * stage_prob_under

        if abs(fav_score - under_score) < 1e-4:
            # Break exact ties by preferring the lower seed to keep brackets balanced.
            return favorite if favorite.seed <= underdog.seed else underdog
        return favorite if fav_score >= under_score else underdog


class AdvancementInformedPolicy:
    """Selection policy that blends matchup, simulation, and advancement NN signals."""

    def __init__(
        self,
        advancement_probs: AdvancementProbabilities,
        simulation_probs: Optional[SimulationProbabilities] = None,
        close_margin: float = 0.10,
        simulation_weight: float = 0.5,
        advancement_weight: float = 0.35,
    ) -> None:
        if simulation_weight < 0 or advancement_weight < 0:
            raise ValueError("Weights must be non-negative.")
        total_weight = simulation_weight + advancement_weight
        if total_weight >= 1.0:
            raise ValueError("simulation_weight + advancement_weight must be < 1.")
        self.advancement_probs = advancement_probs
        self.simulation_probs = simulation_probs
        self.close_margin = max(0.0, close_margin)
        self.simulation_weight = simulation_weight
        self.advancement_weight = advancement_weight
        self.base_weight = 1.0 - total_weight
        self.high_conf_threshold = CONFIDENCE_THRESHOLDS["high"]

    def _score_with_signal(
        self,
        base_prob: float,
        signal_prob: Optional[float],
        weight: float,
        fallback: float,
    ) -> float:
        if signal_prob is None:
            return weight * fallback
        return weight * signal_prob

    def __call__(
        self,
        team1: TeamState,
        team2: TeamState,
        prob_team1: float,
        prob_team2: float,
        round_name: str,
        slot: str,
        region: Optional[str],
    ) -> TeamState:
        del slot, region
        margin = abs(prob_team1 - prob_team2)
        favorite = team1 if prob_team1 >= prob_team2 else team2
        underdog = team2 if favorite is team1 else team1
        favorite_prob = max(prob_team1, prob_team2)

        if favorite_prob >= self.high_conf_threshold or margin >= self.close_margin:
            return favorite

        sim_fav = self.simulation_probs.stage_probability(favorite.team_key, round_name) if self.simulation_probs else None
        sim_under = (
            self.simulation_probs.stage_probability(underdog.team_key, round_name) if self.simulation_probs else None
        )
        adv_fav = self.advancement_probs.stage_probability(favorite.team_key, round_name)
        adv_under = self.advancement_probs.stage_probability(underdog.team_key, round_name)

        fav_score = self.base_weight * favorite_prob
        under_score = self.base_weight * (1 - favorite_prob)
        fav_score += self._score_with_signal(favorite_prob, sim_fav, self.simulation_weight, favorite_prob)
        under_score += self._score_with_signal(1 - favorite_prob, sim_under, self.simulation_weight, 1 - favorite_prob)
        fav_score += self._score_with_signal(favorite_prob, adv_fav, self.advancement_weight, favorite_prob)
        under_score += self._score_with_signal(1 - favorite_prob, adv_under, self.advancement_weight, 1 - favorite_prob)

        if abs(fav_score - under_score) < 1e-4:
            return favorite if favorite.seed <= underdog.seed else underdog
        return favorite if fav_score >= under_score else underdog


def _summarize_bracket(bracket_payload: Dict[str, object]) -> Dict[str, object]:
    rounds = bracket_payload.get("rounds", {})
    winners = {}
    for round_name, games in rounds.items():
        for game in games:
            slot = game.get("slot")
            winner = (game.get("winner") or {}).get("display_name")
            if slot and winner:
                winners[(round_name, slot)] = winner
    elite_round = rounds.get("Elite8", [])
    final_four = sorted(
        { (game.get("winner") or {}).get("display_name") for game in elite_round if (game.get("winner") or {}).get("display_name") }
    )
    champion = (bracket_payload.get("champion") or {}).get("display_name")
    return {
        "winners": winners,
        "final_four": final_four,
        "champion": champion,
    }


def _build_difference(report_a: Dict[str, object], report_b: Dict[str, object]) -> Dict[str, object]:
    winners_a = report_a["winners"]
    winners_b = report_b["winners"]
    all_slots = sorted(set(winners_a) | set(winners_b))
    changed = []
    for key in all_slots:
        pick_a = winners_a.get(key)
        pick_b = winners_b.get(key)
        if pick_a == pick_b:
            continue
        changed.append({"round": key[0], "slot": key[1], "reference": pick_a, "final": pick_b})
    return {
        "champion_changed": report_a["champion"] != report_b["champion"],
        "champion_reference": report_a["champion"],
        "champion_final": report_b["champion"],
        "final_four_changed": sorted(report_a["final_four"]) != sorted(report_b["final_four"]),
        "reference_final_four": report_a["final_four"],
        "final_final_four": report_b["final_four"],
        "slot_differences": changed,
        "total_changed_games": len(changed),
    }


def _write_comparison_summary(
    final_payload: Dict[str, object],
    baseline_payload: Dict[str, object],
    enriched_payload: Dict[str, object],
    strategy: str,
    label: str,
    metadata: Dict[str, object],
    output_json: Path,
    output_md: Optional[Path],
) -> Dict[str, object]:
    summary_final = _summarize_bracket(final_payload)
    summary_baseline = _summarize_bracket(baseline_payload)
    summary_enriched = _summarize_bracket(enriched_payload)
    report = {
        "strategy": strategy,
        "label": label,
        "metadata": metadata,
        "final": {
            "champion": summary_final["champion"],
            "final_four": summary_final["final_four"],
        },
        "differences": {
            "vs_baseline": _build_difference(summary_baseline, summary_final),
            "vs_enriched": _build_difference(summary_enriched, summary_final),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if output_md:
        lines = [
            "# Final Bracket Selection Summary",
            f"- Strategy: {strategy}",
            f"- Label: {label}",
            f"- Champion: {summary_final['champion']}",
            "",
            "## Differences vs. Baseline",
            f"- Champion changed: {'Yes' if report['differences']['vs_baseline']['champion_changed'] else 'No'}",
            f"- Final Four changed: {'Yes' if report['differences']['vs_baseline']['final_four_changed'] else 'No'}",
            f"- Changed games: {report['differences']['vs_baseline']['total_changed_games']}",
            "",
            "## Differences vs. Enriched",
            f"- Champion changed: {'Yes' if report['differences']['vs_enriched']['champion_changed'] else 'No'}",
            f"- Final Four changed: {'Yes' if report['differences']['vs_enriched']['final_four_changed'] else 'No'}",
            f"- Changed games: {report['differences']['vs_enriched']['total_changed_games']}",
        ]
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_selection(
    strategy: str,
    bracket_file: Path,
    season: int,
    baseline_predictions: Path,
    enriched_predictions: Optional[Path],
    baseline_weight: float,
    simulation_probabilities: Optional[Path],
    advancement_probabilities: Optional[Path],
    advancement_weight: float,
    label_suffix: Optional[str],
    output_dir: Path,
    comparison_json: Path,
    comparison_md: Optional[Path],
    baseline_bracket_json: Path,
    enriched_bracket_json: Path,
    render_html: bool,
    html_output: Path,
    close_margin: float,
    simulation_weight: float,
) -> Dict[str, object]:
    bracket_def = load_bracket_definition(bracket_file)
    if not baseline_bracket_json.exists():
        raise FileNotFoundError(f"Baseline bracket JSON {baseline_bracket_json} not found.")
    if not enriched_bracket_json.exists():
        raise FileNotFoundError(f"Enriched bracket JSON {enriched_bracket_json} not found.")

    lookup: Optional[PredictionLookup] = None
    selection_policy = None
    predictions_path = baseline_predictions
    label = label_suffix or strategy

    if strategy == "baseline_deterministic":
        predictions_path = baseline_predictions
    elif strategy == "enriched_deterministic":
        if not enriched_predictions:
            raise ValueError("Enriched predictions are required for enriched_deterministic strategy.")
        predictions_path = enriched_predictions
    elif strategy in {"blended_probabilities", "simulation_informed", "advancement_informed"}:
        if not enriched_predictions:
            raise ValueError("Enriched predictions are required for blending or simulation strategies.")
        lookup = BlendedPredictionLookup(
            baseline_path=baseline_predictions,
            enriched_path=enriched_predictions,
            season=season,
            baseline_weight=baseline_weight,
        )
        predictions_path = baseline_predictions
        if strategy == "simulation_informed":
            if not simulation_probabilities:
                raise ValueError("Simulation probabilities are required for simulation_informed strategy.")
            sim_probs = SimulationProbabilities(simulation_probabilities)
            selection_policy = SimulationInformedPolicy(
                simulation_probs=sim_probs,
                close_margin=close_margin,
                stage_weight=simulation_weight,
            )
        elif strategy == "advancement_informed":
            if not advancement_probabilities:
                raise ValueError("Advancement probabilities are required for advancement_informed strategy.")
            adv_probs = AdvancementProbabilities(advancement_probabilities)
            sim_probs = SimulationProbabilities(simulation_probabilities) if simulation_probabilities else None
            selection_policy = AdvancementInformedPolicy(
                advancement_probs=adv_probs,
                simulation_probs=sim_probs,
                close_margin=close_margin,
                simulation_weight=simulation_weight,
                advancement_weight=advancement_weight,
            )
    else:
        raise ValueError(f"Unsupported strategy '{strategy}'.")

    generator = BracketGenerator(
        predictions_path=predictions_path,
        output_dir=output_dir,
        selection_policy=selection_policy,
    )
    outputs = generator.generate(
        bracket_def,
        season=season,
        label_suffix=label,
        lookup=lookup,
    )

    baseline_payload = json.loads(baseline_bracket_json.read_text(encoding="utf-8"))
    enriched_payload = json.loads(enriched_bracket_json.read_text(encoding="utf-8"))
    final_payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    metadata = {
        "baseline_predictions": str(baseline_predictions),
        "enriched_predictions": str(enriched_predictions) if enriched_predictions else None,
        "simulation_probabilities": str(simulation_probabilities) if simulation_probabilities else None,
        "baseline_weight": baseline_weight if strategy in {"blended_probabilities", "simulation_informed", "advancement_informed"} else None,
        "simulation_weight": simulation_weight if strategy in {"simulation_informed", "advancement_informed"} else None,
        "advancement_probabilities": str(advancement_probabilities) if strategy == "advancement_informed" else None,
        "advancement_weight": advancement_weight if strategy == "advancement_informed" else None,
    }
    _write_comparison_summary(
        final_payload,
        baseline_payload,
        enriched_payload,
        strategy,
        label,
        metadata,
        comparison_json,
        comparison_md,
    )

    final_html_path = None
    if render_html:
        renderer = BracketViewRenderer(
            bracket_results_path=outputs["json"],
            output_path=html_output,
        )
        final_html_path = renderer.render()

    result = {
        "outputs": outputs,
        "comparison_json": comparison_json,
        "comparison_md": comparison_md,
        "html": final_html_path,
    }
    return result


def parse_args() -> argparse.Namespace:
    config = get_config()
    outputs_dir = config.paths.outputs_dir
    predictions_dir = outputs_dir / "predictions"
    brackets_dir = outputs_dir / "brackets"
    reports_dir = outputs_dir / "reports"

    parser = argparse.ArgumentParser(description="Select a smarter final bracket using blending or simulation signals.")
    parser.add_argument(
        "--strategy",
        choices=[
            "baseline_deterministic",
            "enriched_deterministic",
            "blended_probabilities",
            "simulation_informed",
            "advancement_informed",
        ],
        default="simulation_informed",
        help="Bracket-selection strategy to run.",
    )
    parser.add_argument("--season", type=int, default=2026, help="Season / tournament year.")
    parser.add_argument("--bracket-file", type=Path, default=DEFAULT_BRACKET_PATH, help="Bracket definition JSON.")
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=predictions_dir / "2026_matchup_predictions_projected_field_baseline.csv",
        help="Baseline field-only predictions CSV.",
    )
    parser.add_argument(
        "--enriched-predictions",
        type=Path,
        default=predictions_dir / "2026_matchup_predictions_projected_field_cbbpy.csv",
        help="CBBpy-enriched field-only predictions CSV.",
    )
    parser.add_argument(
        "--simulation-probabilities",
        type=Path,
        default=brackets_dir / "2026_projected_simulation_team_probabilities_cbbpy.csv",
        help="Simulation team probability CSV to guide close decisions.",
    )
    parser.add_argument(
        "--advancement-probabilities",
        type=Path,
        default=predictions_dir / "2026_advancement_probabilities.csv",
        help="Advancement NN probability CSV for the projected field.",
    )
    parser.add_argument(
        "--baseline-bracket-json",
        type=Path,
        default=brackets_dir / "2026_projected_bracket_results_baseline.json",
        help="Existing baseline deterministic bracket JSON.",
    )
    parser.add_argument(
        "--enriched-bracket-json",
        type=Path,
        default=brackets_dir / "2026_projected_bracket_results_cbbpy.json",
        help="Existing enriched deterministic bracket JSON.",
    )
    parser.add_argument(
        "--baseline-weight",
        type=float,
        default=0.65,
        help="Weight assigned to baseline probabilities during blending.",
    )
    parser.add_argument(
        "--close-margin",
        type=float,
        default=0.12,
        help="If the win-probability margin is below this value, consult simulation probabilities.",
    )
    parser.add_argument(
        "--simulation-weight",
        type=float,
        default=0.65,
        help="Weight for simulation advancement probabilities when making close calls.",
    )
    parser.add_argument(
        "--advancement-weight",
        type=float,
        default=0.3,
        help="Weight for advancement NN probabilities when making close calls.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label suffix appended to output artifact names (defaults to the strategy name).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=brackets_dir,
        help="Destination directory for bracket artifacts.",
    )
    parser.add_argument(
        "--comparison-json",
        type=Path,
        default=reports_dir / "final_bracket_selection_comparison.json",
        help="Where to write the JSON comparison summary.",
    )
    parser.add_argument(
        "--comparison-md",
        type=Path,
        default=reports_dir / "final_bracket_selection_comparison.md",
        help="Optional Markdown summary output.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=outputs_dir / "final_report" / "bracket_view_final.html",
        help="Destination for the rendered HTML bracket view.",
    )
    parser.add_argument(
        "--render-html",
        dest="render_html",
        action="store_true",
        help="Render a polished HTML bracket view for the final selection.",
    )
    parser.add_argument(
        "--no-render-html",
        dest="render_html",
        action="store_false",
        help="Skip HTML rendering.",
    )
    parser.set_defaults(render_html=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_selection(
        strategy=args.strategy,
        bracket_file=args.bracket_file,
        season=args.season,
        baseline_predictions=args.baseline_predictions,
        enriched_predictions=args.enriched_predictions,
        baseline_weight=args.baseline_weight,
        simulation_probabilities=args.simulation_probabilities,
        advancement_probabilities=args.advancement_probabilities,
        advancement_weight=args.advancement_weight,
        label_suffix=args.label,
        output_dir=args.output_dir,
        comparison_json=args.comparison_json,
        comparison_md=args.comparison_md,
        baseline_bracket_json=args.baseline_bracket_json,
        enriched_bracket_json=args.enriched_bracket_json,
        render_html=args.render_html,
        html_output=args.html_output,
        close_margin=args.close_margin,
        simulation_weight=args.simulation_weight,
    )
    print(f"[bracket_selection] Wrote bracket artifacts: {result['outputs']}")
    if result["html"]:
        print(f"[bracket_selection] Rendered HTML view at {result['html']}")
    print(f"[bracket_selection] Comparison JSON written to {args.comparison_json}")


if __name__ == "__main__":
    main()
