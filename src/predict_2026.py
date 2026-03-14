from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import data_loading, dataset_builder
from .config import get_config
from .data_pipeline import load_and_build_dataset
from .feature_metadata import available_feature_sets, select_diff_columns
from .supplemental_ncaa import build_supplemental_features
from .train import prepare_features, run_logistic_regression_baseline
from .utils import ensure_parent_dir, set_random_seed


CBBPY_OVERRIDE_MAP: Dict[str, str] = {
    "GamesPlayed": "CBBpy_GamesPlayed",
    "WinPercentage": "CBBpy_WinPct",
    "PointsPerGame": "CBBpy_PointsForPerGame",
    "PointsAllowedPerGame": "CBBpy_PointsAgainstPerGame",
    "ScoringMargin": "CBBpy_ScoringMargin",
    "OffensiveRating": "CBBpy_OffRating",
    "DefensiveRating": "CBBpy_DefRating",
    "NetRating": "CBBpy_NetRating",
}


def _load_optional_massey(data_dir: Path, system: str | None = None) -> pd.DataFrame | None:
    try:
        massey = data_loading.load_massey_ordinals(data_dir)
    except FileNotFoundError:
        return None
    return dataset_builder.summarize_massey_ordinals(massey, system_name=system)


def _load_supplemental_features(include: bool, supplemental_dir: Path, teams_df: pd.DataFrame):
    if not include:
        return None
    features, _ = build_supplemental_features(supplemental_dir, teams_df)
    return features if not features.empty else None


def _load_cbbpy_features(include: bool, features_path: Path | None) -> pd.DataFrame | None:
    if not include or features_path is None or not features_path.exists():
        return None
    df = pd.read_csv(features_path)
    if "SourceTeamName" in df.columns:
        df = df.drop(columns=["SourceTeamName"])
    return df


def apply_cbbpy_overrides(context: pd.DataFrame, cbbpy_frame: pd.DataFrame | None, season: int | None) -> pd.DataFrame:
    if cbbpy_frame is None or cbbpy_frame.empty or season is None:
        return context
    season_subset = cbbpy_frame[cbbpy_frame["Season"] == season].copy()
    if season_subset.empty:
        return context
    available = {base: col for base, col in CBBPY_OVERRIDE_MAP.items() if base in context.columns and col in season_subset.columns}
    if not available:
        return context
    override = season_subset[["TeamID"] + list(available.values())].copy()
    rename_map = {col: f"CBBpyOverride_{base}" for base, col in available.items()}
    override = override.rename(columns=rename_map)
    merged = context.merge(override, on="TeamID", how="left")
    season_mask = merged["Season"] == season
    for base_col in available.keys():
        override_col = f"CBBpyOverride_{base_col}"
        if override_col not in merged.columns:
            continue
        mask = season_mask & merged[override_col].notna()
        merged.loc[mask, base_col] = merged.loc[mask, override_col]
        merged = merged.drop(columns=[override_col])
    return merged


def build_team_context(
    data_dir: Path,
    include_supplemental: bool,
    supplemental_dir: Path,
    include_cbbpy: bool,
    cbbpy_features: Path | None,
    cbbpy_season: int | None,
    massey_system: str | None = None,
) -> pd.DataFrame:
    regular = data_loading.load_regular_season_detailed_results(data_dir)
    seeds = data_loading.load_tourney_seeds(data_dir)
    teams = data_loading.load_teams(data_dir)
    massey_summary = _load_optional_massey(data_dir, massey_system)
    supplemental_features = _load_supplemental_features(include_supplemental, supplemental_dir, teams)
    cbbpy_frame = _load_cbbpy_features(include_cbbpy, cbbpy_features)
    if cbbpy_frame is not None and cbbpy_season is not None:
        season_filtered = cbbpy_frame[cbbpy_frame["Season"] == cbbpy_season]
        if not season_filtered.empty:
            cbbpy_frame = season_filtered
    context = dataset_builder.prepare_team_context(
        regular,
        seeds,
        massey_summary,
        supplemental_features=supplemental_features,
        cbbpy_features=cbbpy_frame,
    )
    if include_cbbpy and cbbpy_season is not None:
        context = apply_cbbpy_overrides(context, cbbpy_frame, cbbpy_season)
    return context


def build_inference_matchups(
    context: pd.DataFrame,
    teams_df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    season_context = context[context["Season"] == season].copy()
    if season_context.empty:
        raise ValueError(f"No team context rows found for season {season}.")
    season_context = season_context.dropna(subset=["TeamID"])
    if "Seed" in season_context.columns:
        seeded_only = season_context.dropna(subset=["Seed"])
        if not seeded_only.empty:
            season_context = seeded_only
    team_ids = season_context["TeamID"].unique()
    if len(team_ids) < 2:
        raise ValueError(f"Need at least two teams for season {season}, found {len(team_ids)}.")
    ordered_ids = sorted(int(tid) for tid in team_ids)
    matchups = pd.DataFrame(
        [
            (season, ordered_ids[i], ordered_ids[j])
            for i in range(len(ordered_ids))
            for j in range(i + 1, len(ordered_ids))
        ],
        columns=["Season", "Team1ID", "Team2ID"],
    )

    team1_ctx = season_context.rename(columns={"TeamID": "Team1ID"})
    team1_value_cols = [col for col in team1_ctx.columns if col not in {"Season", "Team1ID"}]
    for col in team1_value_cols:
        team1_ctx[f"Team1_{col}"] = team1_ctx.pop(col)
    matchups = matchups.merge(team1_ctx, on=["Season", "Team1ID"], how="left")

    team2_ctx = season_context.rename(columns={"TeamID": "Team2ID"})
    team2_value_cols = [col for col in team2_ctx.columns if col not in {"Season", "Team2ID"}]
    for col in team2_value_cols:
        team2_ctx[f"Team2_{col}"] = team2_ctx.pop(col)
    matchups = matchups.merge(team2_ctx, on=["Season", "Team2ID"], how="left")

    numeric_features = [
        col
        for col in season_context.columns
        if col not in {"Season", "TeamID", "Seed"}
    ]
    for feature in numeric_features:
        col1 = f"Team1_{feature}"
        col2 = f"Team2_{feature}"
        if col1 in matchups.columns and col2 in matchups.columns:
            matchups[f"Diff_{feature}"] = matchups[col1] - matchups[col2]

    teams_lookup = teams_df[["TeamID", "TeamName"]]
    matchups = matchups.merge(
        teams_lookup.rename(columns={"TeamID": "Team1ID", "TeamName": "Team1Name"}),
        on="Team1ID",
        how="left",
    ).merge(
        teams_lookup.rename(columns={"TeamID": "Team2ID", "TeamName": "Team2Name"}),
        on="Team2ID",
        how="left",
    )
    matchups["MatchupID"] = (
        matchups["Season"].astype(str)
        + "_"
        + matchups["Team1ID"].astype(str)
        + "_"
        + matchups["Team2ID"].astype(str)
    )
    matchups["Label"] = 0
    return matchups


def train_and_predict(
    dataset: pd.DataFrame,
    inference_df: pd.DataFrame,
    feature_set: str,
) -> Tuple[np.ndarray, List[str]]:
    diff_cols = sorted([col for col in dataset.columns if col.startswith("Diff_")])
    selected = select_diff_columns(diff_cols, feature_set)
    if not selected:
        raise ValueError(f"No diff columns available for feature set '{feature_set}'.")
    X_train, y_train, X_inf, _, feature_cols, _ = prepare_features(dataset, inference_df, selected)
    probs = run_logistic_regression_baseline(X_train, y_train, X_inf)
    return probs, feature_cols


def save_predictions(
    matchups: pd.DataFrame,
    probs: np.ndarray,
    output_path: Path,
    feature_set: str,
    model_name: str = "logistic_regression",
) -> None:
    preds = matchups.copy()
    preds["PredTeam1WinProb"] = probs
    preds["PredTeam2WinProb"] = 1.0 - probs
    preds["PredictedWinnerID"] = np.where(probs >= 0.5, preds["Team1ID"], preds["Team2ID"])
    preds["model_name"] = model_name
    preds["feature_set"] = feature_set
    metadata_cols = [
        "MatchupID",
        "Season",
        "Team1ID",
        "Team1Name",
        "Team1_Seed",
        "Team1_SeedNum",
        "Team2ID",
        "Team2Name",
        "Team2_Seed",
        "Team2_SeedNum",
    ]
    ordered_cols = metadata_cols + [
        "PredTeam1WinProb",
        "PredTeam2WinProb",
        "PredictedWinnerID",
        "model_name",
        "feature_set",
    ]
    output_df = preds[[col for col in ordered_cols if col in preds.columns]].copy()
    ensure_parent_dir(output_path)
    output_df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2026 matchup predictions with optional CBBpy enrichment.")
    parser.add_argument("--season", type=int, default=2026, help="Season to generate predictions for.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing Kaggle raw data.")
    parser.add_argument("--supplemental-dir", type=Path, default=None, help="Path to supplemental Kaggle NCAA dataset.")
    parser.add_argument("--include-supplemental-kaggle", action="store_true", help="Include supplemental Kaggle feature group.")
    parser.add_argument("--include-cbbpy-current", action="store_true", help="Include cached current-season CBBpy features.")
    parser.add_argument("--cbbpy-season", type=int, default=2026, help="Season for the cached CBBpy feature file.")
    parser.add_argument("--cbbpy-features", type=Path, default=None, help="Path to cached CBBpy features CSV.")
    parser.add_argument("--feature-set", choices=available_feature_sets(), default="core_plus_opponent_adjustment", help="Feature set to use for logistic regression.")
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions/2026_matchup_predictions.csv"), help="Destination CSV for predictions.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    set_random_seed(args.random_seed)
    data_dir = args.data_dir if args.data_dir else config.paths.raw_data_dir
    supplemental_dir = args.supplemental_dir if args.supplemental_dir else config.paths.supplemental_ncaa_dir
    cbbpy_features_path = args.cbbpy_features if args.cbbpy_features else config.paths.cbbpy_cache_dir / f"cbbpy_team_features_{args.cbbpy_season}.csv"

    dataset, _ = load_and_build_dataset(
        data_dir=data_dir,
        include_supplemental_kaggle=args.include_supplemental_kaggle,
        supplemental_dir=supplemental_dir,
        include_cbbpy_current=args.include_cbbpy_current,
        cbbpy_features_path=cbbpy_features_path,
        cbbpy_season=args.cbbpy_season,
    )
    train_df = dataset[dataset["Season"] < args.season].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Training split empty; ensure historical seasons are present.")

    context = build_team_context(
        data_dir=data_dir,
        include_supplemental=args.include_supplemental_kaggle,
        supplemental_dir=supplemental_dir,
        include_cbbpy=args.include_cbbpy_current,
        cbbpy_features=cbbpy_features_path,
        cbbpy_season=args.cbbpy_season,
    )
    teams_df = data_loading.load_teams(data_dir)
    inference_df = build_inference_matchups(context, teams_df, args.season)
    inference_df_with_label = inference_df.copy()
    inference_df_with_label["Label"] = 0

    probs, _ = train_and_predict(train_df, inference_df_with_label, args.feature_set)
    save_predictions(inference_df, probs, args.output, args.feature_set)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
