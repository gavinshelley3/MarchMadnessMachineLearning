from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import get_config
from .data_inventory import write_local_dataset_inventory
from .data_pipeline import load_and_build_dataset, save_feature_summary_report
from .utils import ensure_parent_dir


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the modeling dataset and print diagnostics without training.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing Kaggle CSVs (defaults to data/raw)",
    )
    parser.add_argument(
        "--massey-system",
        type=str,
        default=None,
        help="Optional Massey system code to filter on.",
    )
    parser.add_argument(
        "--save-dataset",
        type=Path,
        default=None,
        help="Optional path to save the modeling dataset CSV.",
    )
    parser.add_argument(
        "--include-supplemental-kaggle",
        action="store_true",
        help="Include supplemental Kaggle NCAA Basketball features if available.",
    )
    parser.add_argument(
        "--supplemental-dir",
        type=Path,
        default=None,
        help="Override path for the supplemental NCAA Basketball dataset.",
    )
    parser.add_argument(
        "--include-cbbpy-current",
        action="store_true",
        help="Include current-season CBBpy enrichment features if cached.",
    )
    parser.add_argument(
        "--cbbpy-season",
        type=int,
        default=None,
        help="Season (e.g., 2026) to load from the cached CBBpy features.",
    )
    parser.add_argument(
        "--cbbpy-features",
        type=Path,
        default=None,
        help="Optional explicit path to a cached CBBpy team-features CSV.",
    )
    args = parser.parse_args()

    config = get_config()
    data_dir = args.data_dir if args.data_dir else config.paths.raw_data_dir
    supplemental_dir = args.supplemental_dir if args.supplemental_dir else config.paths.supplemental_ncaa_dir
    dataset, diagnostics = load_and_build_dataset(
        data_dir=data_dir,
        massey_system=args.massey_system,
        include_supplemental_kaggle=args.include_supplemental_kaggle,
        supplemental_dir=supplemental_dir,
        reports_dir=config.paths.outputs_dir / "reports",
        include_cbbpy_current=args.include_cbbpy_current,
        cbbpy_features_path=args.cbbpy_features,
        cbbpy_season=args.cbbpy_season,
    )

    save_path = args.save_dataset or (config.paths.processed_data_dir / "matchup_dataset.csv")
    ensure_parent_dir(save_path)
    dataset.to_csv(save_path, index=False)

    reports_dir = config.paths.outputs_dir / "reports"
    diagnostics_path = reports_dir / "dataset_diagnostics.json"
    ensure_parent_dir(diagnostics_path)
    diagnostics_path.write_text(json.dumps(diagnostics.to_dict(), indent=2))
    feature_summary_path = reports_dir / "feature_summary.json"
    save_feature_summary_report(diagnostics, feature_summary_path)
    if diagnostics.supplemental_details:
        supplemental_summary_path = reports_dir / "supplemental_feature_summary.json"
        supplemental_summary_path.write_text(
            json.dumps(diagnostics.supplemental_details, indent=2)
        )

    inventory_path = reports_dir / "local_dataset_inventory.json"
    write_local_dataset_inventory(Path(data_dir), Path(supplemental_dir), inventory_path)

    print("Dataset saved to", save_path)
    print("Diagnostics saved to", diagnostics_path)
    print("Feature summary saved to", feature_summary_path)
    print("\n=== Dataset Diagnostics ===")
    print(f"Total tournament games: {diagnostics.total_tourney_games}")
    print(
        f"Symmetric training rows: {diagnostics.rows_after_filter} (from {diagnostics.rows_before_filter} before filtering)"
    )
    print(f"Dropped rows: {diagnostics.dropped_rows}")
    if diagnostics.dropped_seasons:
        print("Dropped seasons due to missing features:", diagnostics.dropped_seasons)
    print(f"Team feature join coverage: {format_percentage(diagnostics.team_feature_join_coverage)}")
    print(f"Seed parse coverage: {format_percentage(diagnostics.seed_parse_coverage)}")
    print(f"Seed numeric coverage in modeling set: {format_percentage(diagnostics.seed_numeric_coverage)}")
    if diagnostics.massey_system:
        print(f"Massey system: {diagnostics.massey_system}")
    print(f"Massey coverage in modeling set: {format_percentage(diagnostics.massey_coverage)}")
    print(f"Feature columns: {diagnostics.feature_count}")
    print(f"Advanced feature columns present: {diagnostics.advanced_feature_count}")
    if diagnostics.supplemental_feature_count:
        print(
            f"Supplemental feature columns present: {diagnostics.supplemental_feature_count} "
            f"(join coverage {format_percentage(diagnostics.supplemental_join_coverage)})"
        )
    if diagnostics.cbbpy_feature_count:
        print(
            f"CBBpy current-season feature columns present: {diagnostics.cbbpy_feature_count} "
            f"(join coverage {format_percentage(diagnostics.cbbpy_join_coverage)})"
        )
    print("Class balance:")
    for label, pct in diagnostics.class_balance.items():
        print(f"  label {label}: {format_percentage(pct)}")
    null_columns = [col for col, count in diagnostics.feature_null_counts.items() if count]
    if null_columns:
        print("Columns with nulls in final dataset:", null_columns)
    else:
        print("Columns with nulls in final dataset: none")


if __name__ == "__main__":
    main()
