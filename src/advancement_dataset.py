"""Build team-level advancement datasets for tournament modeling."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import data_loading, dataset_builder
from .config import get_config
from .supplemental_ncaa import (
    SupplementalDiagnostics,
    build_supplemental_features,
    save_supplemental_report,
)


ADVANCEMENT_LABELS: List[Tuple[str, int]] = [
    ("reached_round_of_32", 1),
    ("reached_sweet_16", 2),
    ("reached_elite_8", 3),
    ("reached_final_four", 4),
    ("reached_championship_game", 5),
    ("won_championship", 6),
]


@dataclass
class AdvancementDatasetDiagnostics:
    total_rows: int
    seasons: List[int]
    feature_columns: List[str]
    feature_null_counts: Dict[str, int]
    label_positive_rates: Dict[str, float]
    play_in_team_count: int
    supplemental_feature_count: int
    cbbpy_feature_count: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _play_in_offset(seed: str) -> int:
    suffix = "".join(ch for ch in seed.lower() if ch.isalpha())
    if not suffix:
        return 0
    # Seeds that end with letters (a/b) indicate play-in participants.
    return 1 if suffix[-1] in {"a", "b"} else 0


def _summarize_labels(dataset: pd.DataFrame) -> Dict[str, float]:
    summary = {}
    for label, _ in ADVANCEMENT_LABELS:
        summary[label] = float(dataset[label].mean()) if label in dataset.columns else 0.0
    return summary


def build_team_context_with_features(
    regular_season: pd.DataFrame,
    seeds: pd.DataFrame,
    massey_summary: Optional[pd.DataFrame],
    supplemental_features: Optional[pd.DataFrame],
    cbbpy_features: Optional[pd.DataFrame],
) -> pd.DataFrame:
    context = dataset_builder.prepare_team_context(
        regular_season,
        seeds,
        massey_summary=massey_summary,
        supplemental_features=supplemental_features,
        cbbpy_features=cbbpy_features,
    )
    # Drop the duplicate Seed column; we will preserve Seed from the seeds table.
    if "Seed" in context.columns:
        context = context.drop(columns=["Seed"])
    return context


def build_advancement_dataset_from_frames(
    seeds: pd.DataFrame,
    tourney_results: pd.DataFrame,
    context: pd.DataFrame,
    teams: pd.DataFrame,
    min_season: int = 1985,
) -> Tuple[pd.DataFrame, AdvancementDatasetDiagnostics]:
    seeds = seeds.copy()
    seeds = seeds[seeds["Season"] >= min_season]
    team_names = teams[["TeamID", "TeamName"]].copy()
    dataset = seeds.merge(context, on=["Season", "TeamID"], how="left")
    dataset = dataset.merge(team_names, on="TeamID", how="left")
    wins = (
        tourney_results.groupby(["Season", "WTeamID"]).size().rename("Wins").reset_index()
    )
    dataset = dataset.merge(
        wins,
        left_on=["Season", "TeamID"],
        right_on=["Season", "WTeamID"],
        how="left",
    ).drop(columns=["WTeamID"])
    dataset["Wins"] = dataset["Wins"].fillna(0).astype(int)
    dataset["PlayInOffset"] = dataset["Seed"].astype(str).apply(_play_in_offset)

    for label, required_wins in ADVANCEMENT_LABELS:
        dataset[label] = (
            dataset["Wins"] >= (dataset["PlayInOffset"] + required_wins)
        ).astype(int)

    label_names = [label for label, _ in ADVANCEMENT_LABELS]
    excluded = {"Season", "TeamID", "TeamName", "Seed", "Wins", "PlayInOffset"} | set(label_names)
    feature_columns = [
        col
        for col in dataset.columns
        if col not in excluded and dataset[col].dtype != object
    ]
    feature_null_counts = {
        col: int(dataset[col].isna().sum()) for col in feature_columns
    }

    diagnostics = AdvancementDatasetDiagnostics(
        total_rows=int(len(dataset)),
        seasons=sorted(dataset["Season"].unique().tolist()),
        feature_columns=feature_columns,
        feature_null_counts=feature_null_counts,
        label_positive_rates=_summarize_labels(dataset),
        play_in_team_count=int((dataset["PlayInOffset"] > 0).sum()),
        supplemental_feature_count=len(
            [col for col in feature_columns if col.startswith("Supplemental_")]
        ),
        cbbpy_feature_count=len(
            [col for col in feature_columns if col.startswith("CBBpy_")]
        ),
    )
    ordered_cols = (
        ["Season", "TeamID", "TeamName", "Seed", "SeedNum", "Wins", "PlayInOffset"]
        + feature_columns
        + [label for label, _ in ADVANCEMENT_LABELS]
    )
    dataset = dataset[ordered_cols]
    return dataset, diagnostics


def build_advancement_dataset(
    data_dir: Path | None = None,
    include_supplemental_kaggle: bool = False,
    supplemental_dir: Optional[Path] = None,
    include_cbbpy_current: bool = False,
    cbbpy_features_path: Optional[Path] = None,
    cbbpy_season: Optional[int] = None,
    massey_system: str | None = None,
    reports_dir: Optional[Path] = None,
    min_season: int = 1985,
) -> Tuple[pd.DataFrame, AdvancementDatasetDiagnostics]:
    regular = data_loading.load_regular_season_detailed_results(data_dir)
    seeds = data_loading.load_tourney_seeds(data_dir)
    tourney_results = data_loading.load_tourney_compact_results(data_dir)
    teams = data_loading.load_teams(data_dir)
    massey = None
    try:
        massey = data_loading.load_massey_ordinals(data_dir)
    except FileNotFoundError:
        massey = None

    supplemental_features: Optional[pd.DataFrame] = None
    supplemental_diagnostics: Optional[SupplementalDiagnostics] = None
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

    cbbpy_features: Optional[pd.DataFrame] = None
    if include_cbbpy_current:
        features_path = Path(cbbpy_features_path) if cbbpy_features_path else None
        if features_path is None or not features_path.exists():
            config = get_config()
            if cbbpy_season is not None:
                inferred_season = cbbpy_season
            elif not tourney_results.empty:
                inferred_season = int(tourney_results["Season"].max())
            else:
                inferred_season = datetime.now().year
            features_path = config.paths.cbbpy_cache_dir / f"cbbpy_team_features_{inferred_season}.csv"
        if features_path.exists():
            cbbpy_features = pd.read_csv(features_path)
            if "SourceTeamName" in cbbpy_features.columns:
                cbbpy_features = cbbpy_features.drop(columns=["SourceTeamName"])
        else:
            print(f"[advancement_dataset] CBBpy features not found at {features_path}. Continuing without them.")

    massey_summary = (
        dataset_builder.summarize_massey_ordinals(massey, massey_system)
        if massey is not None
        else None
    )
    context = build_team_context_with_features(
        regular,
        seeds,
        massey_summary=massey_summary,
        supplemental_features=supplemental_features,
        cbbpy_features=cbbpy_features,
    )
    dataset, diagnostics = build_advancement_dataset_from_frames(
        seeds,
        tourney_results,
        context,
        teams,
        min_season=min_season,
    )
    if reports_dir:
        summary_path = Path(reports_dir) / "advancement_dataset_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(diagnostics.to_dict(), indent=2),
            encoding="utf-8",
        )
    return dataset, diagnostics
