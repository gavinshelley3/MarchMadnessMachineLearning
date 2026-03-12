from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import data_loading
from .config import get_config
from .feature_engineering import parse_numeric_seed


def summarize_dataframe(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("Head:")
    print(df.head())
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("Missing values: none")
    else:
        print("Missing values:")
        print(missing)
    if "Season" in df.columns:
        seasons = df["Season"].dropna()
        if not seasons.empty:
            print(
                f"Season coverage: {int(seasons.min())} - {int(seasons.max())}"
            )
    if name == "tourney_seeds" and "Seed" in df.columns:
        parsed = df["Seed"].apply(parse_numeric_seed)
        coverage = parsed.notna().mean()
        print(f"Seed parsing coverage: {coverage:.2%}")


def run_inspection(data_dir: Path | None = None, tables: Iterable[str] | None = None) -> None:
    report_expected_files(data_dir)
    loaders = {
        "teams": data_loading.load_teams,
        "regular_season_detailed": data_loading.load_regular_season_detailed_results,
        "tourney_compact": data_loading.load_tourney_compact_results,
        "tourney_seeds": data_loading.load_tourney_seeds,
        "massey_ordinals": data_loading.load_massey_ordinals,
    }

    selected_tables = list(tables) if tables else loaders.keys()
    for table_name in selected_tables:
        if table_name not in loaders:
            raise ValueError(f"Unknown table '{table_name}'. Choices: {list(loaders)}")
        df = loaders[table_name](data_dir)
        summarize_dataframe(table_name, df)


def report_expected_files(data_dir: Path | None) -> None:
    config = get_config()
    base_dir = Path(data_dir) if data_dir else config.paths.raw_data_dir
    print("=== Required file check ===")
    missing = []
    for key, filename in data_loading.CORE_FILES.items():
        exists = (base_dir / filename).exists()
        status = "FOUND" if exists else "MISSING"
        print(f"{filename}: {status}")
        if not exists:
            missing.append(filename)
    if missing:
        print("Missing files will prevent the pipeline from running:", missing)
    else:
        print("All core Kaggle files are present.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Kaggle March Madness data files for sanity checks.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing Kaggle CSV files (defaults to data/raw)",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=None,
        help="Optional subset of tables to inspect (use keys from data_loading.CORE_FILES)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_inspection(args.data_dir, args.tables)


if __name__ == "__main__":
    main()
