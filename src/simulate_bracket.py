"""CLI entry point for running bracket simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from .bracket_loader import load_bracket_definition
from .bracket_simulation import BracketSimulator
from .config import get_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations for the NCAA tournament bracket."
    )
    parser.add_argument(
        "--bracket-file",
        type=Path,
        default=Path("data/brackets/projected_2026_bracket.json"),
        help="Path to the bracket JSON input.",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=Path("outputs/predictions/2026_matchup_predictions.csv"),
        help="CSV with matchup predictions (created by the inference pipeline).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for simulation outputs (defaults to outputs/brackets).",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=5000,
        help="Number of Monte Carlo tournaments to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible simulations.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label suffix (e.g., 'baseline' or 'cbbpy') for output filenames.",
    )
    return parser.parse_args()


def main() -> Dict[str, Path]:
    args = parse_args()
    config = get_config()
    bracket_path = args.bracket_file
    predictions_path = args.predictions_file

    if not bracket_path.exists():
        raise SystemExit(f"Bracket file {bracket_path} not found.")
    if not predictions_path.exists():
        raise SystemExit(
            f"Prediction file {predictions_path} not found. "
            "Run the inference pipeline before simulating brackets."
        )

    bracket_def = load_bracket_definition(bracket_path)
    simulator = BracketSimulator(
        bracket_def=bracket_def,
        predictions_path=predictions_path,
        n_sims=args.n_sims,
        seed=args.seed,
        output_dir=args.output_dir or (config.paths.outputs_dir / "brackets"),
        label_suffix=args.label,
    )
    outputs = simulator.run()
    print("Simulation outputs:")
    print(f"- Team probabilities: {outputs.probabilities}")
    print(f"- Summary: {outputs.summary}")
    print(f"- Pick confidence: {outputs.confidence}")
    if outputs.upset_report:
        print(f"- Upset risk report: {outputs.upset_report}")
    else:
        print("- Upset risk report: not generated (no qualifying teams)")
    return {
        "probabilities": outputs.probabilities,
        "summary": outputs.summary,
        "confidence": outputs.confidence,
        "upset_report": outputs.upset_report,
    }


if __name__ == "__main__":
    main()
