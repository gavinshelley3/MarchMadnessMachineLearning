"""CLI wrapper for rendering a bracket HTML view."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bracket_view import BracketViewRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a visual HTML bracket from deterministic outputs.")
    parser.add_argument("--bracket-results", type=Path, default=None, help="Path to *_bracket_results.json")
    parser.add_argument("--output", type=Path, default=None, help="Destination HTML file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renderer = BracketViewRenderer(
        bracket_results_path=args.bracket_results,
        output_path=args.output,
    )
    output_path = renderer.render()
    print(f"Wrote bracket view to {output_path}")


if __name__ == "__main__":
    main()
