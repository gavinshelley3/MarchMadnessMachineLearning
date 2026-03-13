from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import data_loading, dataset_builder
from .config import get_config
from .feature_engineering import parse_numeric_seed
from .feature_metadata import (
    ADVANCED_FEATURE_COLUMNS,
    SUPPLEMENTAL_NCAA_FEATURES,
    CBBPY_CURRENT_FEATURES,
)
from .supplemental_ncaa import (
    SupplementalDiagnostics,
    build_supplemental_features,
    save_supplemental_report,
)


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
    advanced_feature_count: int
    advanced_feature_names: List[str] = field(default_factory=list)
    feature_summary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    massey_system: Optional[str] = None
    supplemental_feature_count: int = 0
    supplemental_feature_names: List[str] = field(default_factory=list)
    supplemental_join_coverage: float = 0.0
    supplemental_details: Dict[str, object] = field(default_factory=dict)
    cbbpy_feature_count: int = 0
    cbbpy_feature_names: List[str] = field(default_factory=list)
    cbbpy_join_coverage: float = 0.0
    cbbpy_details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_dataset_from_frames(
    regular_season_detailed: pd.DataFrame,
    tourney_compact: pd.DataFrame,
    seeds: pd.DataFrame,
    massey_ordinals: Optional[pd.DataFrame] = None,
    massey_system: str | None = None,
    supplemental_features: Optional[pd.DataFrame] = None,
    supplemental_diagnostics: Optional[SupplementalDiagnostics] = None,
    cbbpy_features: Optional[pd.DataFrame] = None,
    cbbpy_diagnostics: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, DatasetDiagnostics]:
    dataset_raw = dataset_builder.build_matchup_dataset(
        regular_season_detailed,
        tourney_compact,
        seeds,
        massey_ordinals,
        massey_system=massey_system,
        supplemental_features=supplemental_features,
        cbbpy_features=cbbpy_features,
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
    advanced_diff_cols = [
        col for col in diff_cols if col.replace("Diff_", "") in ADVANCED_FEATURE_COLUMNS
    ]
    supplemental_diff_cols = [
        col for col in diff_cols if col.replace("Diff_", "") in SUPPLEMENTAL_NCAA_FEATURES
    ]
    supplemental_join_coverage = (
        1.0
        - (dataset_raw[supplemental_diff_cols].isna().any(axis=1).sum() / rows_before)
        if supplemental_diff_cols and rows_before
        else 0.0
    )
    cbbpy_diff_cols = [
        col for col in diff_cols if col.replace("Diff_", "") in CBBPY_CURRENT_FEATURES
    ]
    cbbpy_join_coverage = (
        1.0
        - (dataset_raw[cbbpy_diff_cols].isna().any(axis=1).sum() / rows_before)
        if cbbpy_diff_cols and rows_before
        else 0.0
    )
    optional_cols = set(supplemental_diff_cols + cbbpy_diff_cols)
    required_diff_cols = [col for col in diff_cols if col not in optional_cols]
    required_frame = dataset_raw[required_diff_cols] if required_diff_cols else dataset_raw[[]]
    missing_feature_rows = required_frame.isna().any(axis=1).sum() if not required_frame.empty else 0
    team_feature_join_coverage = (
        1.0 - (missing_feature_rows / rows_before) if rows_before else 0.0
    )
    feature_summary_stats = {
        col: {
            "mean": float(dataset[col].mean()),
            "std": float(dataset[col].std(ddof=0)),
            "min": float(dataset[col].min()),
            "max": float(dataset[col].max()),
        }
        for col in diff_cols
    }

    seed_parse_coverage = float(
        seeds.copy().assign(SeedNum=lambda df: df["Seed"].apply(parse_numeric_seed))["SeedNum"].notna().mean()
    )
    seed_cols = [col for col in dataset.columns if "SeedNum" in col]
    seed_numeric_coverage = float(dataset[seed_cols].notna().mean().mean()) if seed_cols else 0.0
    massey_cols = [col for col in dataset.columns if "MasseyOrdinal" in col]
    massey_coverage = float(dataset[massey_cols].notna().mean().mean()) if massey_cols else 0.0

    class_counts = dataset["Label"].value_counts().sort_index()
    class_balance = {str(int(label)): float(count / len(dataset)) for label, count in class_counts.items()}
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
        advanced_feature_count=len(advanced_diff_cols),
        advanced_feature_names=advanced_diff_cols,
        feature_summary_stats=feature_summary_stats,
        massey_system=massey_system,
        supplemental_feature_count=len(supplemental_diff_cols),
        supplemental_feature_names=supplemental_diff_cols,
        supplemental_join_coverage=float(supplemental_join_coverage),
        supplemental_details=supplemental_diagnostics.to_dict() if supplemental_diagnostics else {},
        cbbpy_feature_count=len(cbbpy_diff_cols),
        cbbpy_feature_names=cbbpy_diff_cols,
        cbbpy_join_coverage=float(cbbpy_join_coverage),
        cbbpy_details=cbbpy_diagnostics or {},
    )
    return dataset, diagnostics


def load_and_build_dataset(
    data_dir: Path | None = None,
    massey_system: str | None = None,
    include_supplemental_kaggle: bool = False,
    supplemental_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    include_cbbpy_current: bool = False,
    cbbpy_features_path: Optional[Path] = None,
    cbbpy_season: Optional[int] = None,
) -> Tuple[pd.DataFrame, DatasetDiagnostics]:
    regular = data_loading.load_regular_season_detailed_results(data_dir)
    tourney = data_loading.load_tourney_compact_results(data_dir)
    seeds = data_loading.load_tourney_seeds(data_dir)
    teams = data_loading.load_teams(data_dir)
    massey = None
    try:
        massey = data_loading.load_massey_ordinals(data_dir)
    except FileNotFoundError:
        massey = None

    supplemental_features = None
    supplemental_diagnostics = None
    if include_supplemental_kaggle:
        supplemental_path = Path(supplemental_dir) if supplemental_dir else Path("data/raw/ncaa_basketball")
        supplemental_features, supplemental_diagnostics = build_supplemental_features(
            supplemental_path,
            teams,
        )
        if supplemental_diagnostics and reports_dir:
            save_supplemental_report(
                supplemental_diagnostics,
                Path(reports_dir) / "supplemental_team_mapping_report.json",
            )

    cbbpy_features = None
    cbbpy_details = None
    if include_cbbpy_current:
        features_path = Path(cbbpy_features_path) if cbbpy_features_path else None
        if features_path is None or not features_path.exists():
            config = get_config()
            inferred_season = cbbpy_season if cbbpy_season is not None else datetime.now().year + 1
            inferred_path = config.paths.cbbpy_cache_dir / f"cbbpy_team_features_{inferred_season}.csv"
            features_path = inferred_path
        if features_path.exists():
            cbbpy_features = pd.read_csv(features_path)
            if "SourceTeamName" in cbbpy_features.columns:
                cbbpy_features = cbbpy_features.drop(columns=["SourceTeamName"])
            cbbpy_details = {
                "season": int(cbbpy_features["Season"].iloc[0]) if not cbbpy_features.empty else cbbpy_season,
                "rows": int(len(cbbpy_features)),
                "source": str(features_path),
            }
        else:
            print(f"CBBpy features file not found at {features_path}. Continuing without current-season enrichment.")
            cbbpy_features = None

    return build_dataset_from_frames(
        regular,
        tourney,
        seeds,
        massey_ordinals=massey,
        massey_system=massey_system,
        supplemental_features=supplemental_features,
        supplemental_diagnostics=supplemental_diagnostics,
        cbbpy_features=cbbpy_features,
        cbbpy_diagnostics=cbbpy_details,
    )


def save_feature_summary_report(diagnostics: DatasetDiagnostics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_count": diagnostics.feature_count,
        "advanced_feature_count": diagnostics.advanced_feature_count,
        "advanced_feature_columns": diagnostics.advanced_feature_names,
        "feature_null_counts": diagnostics.feature_null_counts,
        "feature_summary_stats": diagnostics.feature_summary_stats,
        "supplemental_feature_count": diagnostics.supplemental_feature_count,
        "supplemental_feature_columns": diagnostics.supplemental_feature_names,
        "supplemental_join_coverage": diagnostics.supplemental_join_coverage,
        "cbbpy_feature_count": diagnostics.cbbpy_feature_count,
        "cbbpy_feature_columns": diagnostics.cbbpy_feature_names,
        "cbbpy_join_coverage": diagnostics.cbbpy_join_coverage,
    }
    output_path.write_text(json.dumps(payload, indent=2))
