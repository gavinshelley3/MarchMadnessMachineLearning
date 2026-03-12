from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import data_loading, dataset_builder
from .feature_engineering import parse_numeric_seed


@dataclass
class DatasetDiagnostics:
    total_tourney_games: int
    rows_before_filter: int
    rows_after_filter: int
    dropped_rows: int
    dropped_seasons: List[int]
    unique_matchups: int
    feature_columns: List[str]
    feature_null_counts: Dict[str, int]
    team_feature_join_coverage: float
    seed_parse_coverage: float
    seed_numeric_coverage: float
    massey_coverage: float
    class_balance: Dict[str, float]
    feature_count: int
    massey_system: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        return data


def build_dataset_from_frames(
    regular_season_detailed: pd.DataFrame,
    tourney_compact: pd.DataFrame,
    seeds: pd.DataFrame,
    massey_ordinals: Optional[pd.DataFrame] = None,
    massey_system: str | None = None,
) -> Tuple[pd.DataFrame, DatasetDiagnostics]:
    dataset_raw = dataset_builder.build_matchup_dataset(
        regular_season_detailed,
        tourney_compact,
        seeds,
        massey_ordinals,
        massey_system=massey_system,
    )

    rows_before = len(dataset_raw)
    valid_seasons = set(regular_season_detailed["Season"].unique())
    mask = dataset_raw["Season"].isin(valid_seasons)
    dropped_rows = int((~mask).sum())
    dropped_seasons = (
        sorted(int(season) for season in dataset_raw.loc[~mask, "Season"].unique())
        if dropped_rows
        else []
    )
    dataset = dataset_raw.loc[mask].reset_index(drop=True)

    diff_cols = [col for col in dataset.columns if col.startswith("Diff_")]
    feature_null_counts = {
        col: int(count) for col, count in dataset[diff_cols].isna().sum().items()
    }
    feature_count = len(diff_cols)
    missing_feature_rows = dataset_raw[diff_cols].isna().any(axis=1).sum()
    team_feature_join_coverage = (
        1.0 - (missing_feature_rows / rows_before)
        if rows_before
        else 0.0
    )

    seed_parse_coverage = float(
        seeds.copy().assign(SeedNum=lambda df: df["Seed"].apply(parse_numeric_seed))["SeedNum"].notna().mean()
    )
    seed_cols = [col for col in dataset.columns if "SeedNum" in col]
    seed_numeric_coverage = (
        float(dataset[seed_cols].notna().mean().mean()) if seed_cols else 0.0
    )
    massey_cols = [col for col in dataset.columns if "MasseyOrdinal" in col]
    massey_coverage = (
        float(dataset[massey_cols].notna().mean().mean()) if massey_cols else 0.0
    )

    class_counts = dataset["Label"].value_counts().sort_index()
    class_balance = {}
    for label, count in class_counts.items():
        class_balance[str(int(label))] = float(count / len(dataset))
    unique_matchups = dataset.shape[0] // 2

    diagnostics = DatasetDiagnostics(
        total_tourney_games=rows_before // 2,
        rows_before_filter=rows_before,
        rows_after_filter=len(dataset),
        dropped_rows=dropped_rows,
        dropped_seasons=dropped_seasons,
        unique_matchups=unique_matchups,
        feature_columns=diff_cols,
        feature_null_counts=feature_null_counts,
        team_feature_join_coverage=float(team_feature_join_coverage),
        seed_parse_coverage=seed_parse_coverage,
        seed_numeric_coverage=seed_numeric_coverage,
        massey_coverage=massey_coverage,
        class_balance=class_balance,
        feature_count=feature_count,
        massey_system=massey_system,
    )
    return dataset, diagnostics


def load_and_build_dataset(
    data_dir: Path | None = None, massey_system: str | None = None
) -> Tuple[pd.DataFrame, DatasetDiagnostics]:
    regular = data_loading.load_regular_season_detailed_results(data_dir)
    tourney = data_loading.load_tourney_compact_results(data_dir)
    seeds = data_loading.load_tourney_seeds(data_dir)
    massey = None
    try:
        massey = data_loading.load_massey_ordinals(data_dir)
    except FileNotFoundError:
        massey = None
    return build_dataset_from_frames(
        regular,
        tourney,
        seeds,
        massey_ordinals=massey,
        massey_system=massey_system,
    )
