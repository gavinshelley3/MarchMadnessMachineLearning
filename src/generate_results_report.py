"""CLI entry point for generating presentation-ready reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from .results_report import ResultsReportGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate polished March Madness project summaries.")
    parser.add_argument("--reports-dir", type=Path, default=None, help="Directory containing analytics reports.")
    parser.add_argument("--predictions-dir", type=Path, default=None, help="Directory containing prediction artifacts.")
    parser.add_argument("--brackets-dir", type=Path, default=None, help="Directory containing bracket/simulation outputs.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination directory for summary artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = ResultsReportGenerator(
        reports_dir=args.reports_dir,
        predictions_dir=args.predictions_dir,
        brackets_dir=args.brackets_dir,
        output_dir=args.output_dir,
    )
    generator.run()


if __name__ == "__main__":
    main()
