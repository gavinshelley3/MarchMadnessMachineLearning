"""CLI entry point for bracket generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from .bracket_generation import BracketGenerator
from .bracket_loader import describe_bracket, load_bracket_definition
from .config import get_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deterministic bracket from saved predictions.")
    parser.add_argument(
        "--bracket-file",
        type=Path,
        default=Path("data/brackets/projected_2026_bracket.json"),
        help="Path to the bracket input JSON.",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=Path("outputs/predictions/2026_matchup_predictions.csv"),
        help="CSV of saved matchup predictions.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Override season (defaults to value inside the bracket file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (defaults to outputs/brackets).",
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
    if not bracket_path.exists():
        raise SystemExit(f"Bracket file {bracket_path} not found. Please provide a valid file.")

    predictions_path = args.predictions_file
    if not predictions_path.exists():
        raise SystemExit(
            f"Prediction file {predictions_path} not found. "
            "Run the inference pipeline (python -m src.predict_2026) before generating a bracket."
        )

    bracket = load_bracket_definition(bracket_path)
    print("Loaded bracket definition:", describe_bracket(bracket))
    generator = BracketGenerator(
        predictions_path=predictions_path,
        output_dir=args.output_dir or (config.paths.outputs_dir / "brackets"),
    )
    outputs = generator.generate(bracket, season=args.season, label_suffix=args.label)
    print("Bracket artifacts written:")
    for label, path in outputs.items():
        if path is None:
            continue
        print(f"- {label}: {path}")
    return outputs


if __name__ == "__main__":
    main()
