from __future__ import annotations

import argparse
import json
import difflib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from . import data_loading, dataset_builder
from .bracket_loader import BracketDefinition, load_bracket_definition
from .config import get_config
from .feature_metadata import select_diff_columns
from .production_artifacts import (
    ensure_production_artifacts,
    load_production_artifacts,
)
from .production_config import (
    PRODUCTION_FEATURE_SET,
    PRODUCTION_MODEL_NAME,
    PRODUCTION_SEASON,
    get_predictions_dir,
)
from .utils import ensure_parent_dir

DEFAULT_BRACKET_PATH = Path("data/brackets/projected_2026_bracket.json")
FIELD_ALIASES = {"miami": "miamifl", "queens": "queensnc"}


def parse_sample_submission(sample_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Parse a Kaggle sample submission into canonical matchup rows."""

    season_prefix = f"{season}_"
    season_rows = sample_df[sample_df["ID"].str.startswith(season_prefix)].copy()
    if season_rows.empty:
        raise ValueError(f"No sample submission rows found for season {season}.")
    split_cols = season_rows["ID"].str.split("_", expand=True)
    split_cols.columns = ["Season", "Team1ID", "Team2ID"]
    split_cols = split_cols.astype(int)
    split_cols = split_cols.drop_duplicates().sort_values(["Team1ID", "Team2ID"])
    return split_cols.reset_index(drop=True)


def build_context_for_season(
    data_dir: Path | None,
    season: int,
) -> pd.DataFrame:
    """Build team-level season context for inference."""

    regular = data_loading.load_regular_season_detailed_results(data_dir)
    season_regular = regular[regular["Season"] == season]
    if season_regular.empty:
        raise ValueError(f"No regular season data found for season {season}.")

    seeds = data_loading.load_tourney_seeds(data_dir)
    season_seeds = seeds[seeds["Season"] == season]

    try:
        massey = data_loading.load_massey_ordinals(data_dir)
        massey = massey[massey["Season"] == season]
    except FileNotFoundError:
        massey = None

    massey_summary = (
        dataset_builder.summarize_massey_ordinals(massey) if massey is not None else None
    )
    context = dataset_builder.prepare_team_context(
        season_regular, season_seeds, massey_summary
    )
    return context


def _subset_context_to_matchups(
    context: pd.DataFrame, matchups: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    team_ids = pd.unique(
        matchups[["Team1ID", "Team2ID"]].to_numpy().reshape(-1)
    )
    filtered = context[context["TeamID"].isin(team_ids)].copy()
    coverage = len(filtered) / len(team_ids) if team_ids.size else 0.0
    return filtered, {"team_count": len(team_ids), "coverage": coverage}


def _fill_missing_numeric(
    df: pd.DataFrame, columns: list[str]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    filled = df.copy()
    fill_values = {}
    for col in columns:
        filled[col] = filled[col].astype(float)
        if filled[col].isna().any():
            fill = float(filled[col].mean(skipna=True)) if filled[col].notna().any() else 0.0
            filled[col] = filled[col].fillna(fill)
            fill_values[col] = fill
    return filled, fill_values


def build_inference_dataset(
    context: pd.DataFrame,
    matchups: pd.DataFrame,
) -> pd.DataFrame:
    numeric_features = [
        col for col in context.columns if col not in {"Season", "TeamID", "Seed"}
    ]

    def _rename_context(prefix: str) -> pd.DataFrame:
        rename_map = {
            col: f"{prefix}_{col}" for col in context.columns if col != "Season"
        }
        return context.rename(columns=rename_map)

    dataset = matchups.copy()
    team1_frame = _rename_context("Team1")
    dataset = dataset.merge(
        team1_frame,
        left_on=["Season", "Team1ID"],
        right_on=["Season", "Team1_TeamID"],
        how="left",
    ).drop(columns=["Team1_TeamID"])

    team2_frame = _rename_context("Team2")
    dataset = dataset.merge(
        team2_frame,
        left_on=["Season", "Team2ID"],
        right_on=["Season", "Team2_TeamID"],
        how="left",
    ).drop(columns=["Team2_TeamID"])

    for feature in numeric_features:
        dataset[f"Diff_{feature}"] = (
            dataset[f"Team1_{feature}"] - dataset[f"Team2_{feature}"]
        )

    dataset["MatchupID"] = (
        dataset["Season"].astype(str)
        + "_"
        + dataset["Team1ID"].astype(str)
        + "_"
        + dataset["Team2ID"].astype(str)
    )
    return dataset


def load_matchups(sample_path: Path, season: int) -> pd.DataFrame:
    sample_df = pd.read_csv(sample_path)
    if "ID" not in sample_df.columns:
        raise ValueError("Sample submission file must contain an 'ID' column.")
    return parse_sample_submission(sample_df, season)


def _normalize_team_name(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    return FIELD_ALIASES.get(normalized, normalized)


def _iter_bracket_entries(bracket: BracketDefinition) -> Iterable[str]:
    for matchup in bracket.iter_round_of_64():
        for slot in (matchup.team1, matchup.team2):
            yield slot.team_key
            yield slot.display_name
            for option in slot.options:
                yield option.get("team_key") or ""
                yield option.get("display_name") or ""


def _resolve_field_team_ids(bracket_file: Path, data_dir: Optional[Path]) -> Set[int]:
    bracket = load_bracket_definition(bracket_file)
    teams = data_loading.load_teams(data_dir)
    lookup: Dict[str, int] = {
        _normalize_team_name(row.TeamName): int(row.TeamID)
        for row in teams.itertuples(index=False)
    }
    spellings = data_loading.load_team_spellings(data_dir)
    for row in spellings.itertuples(index=False):
        lookup[_normalize_team_name(row.TeamNameSpelling)] = int(row.TeamID)

    keys = list(lookup.keys())
    resolved: Set[int] = set()
    missing: List[str] = []
    for entry in _iter_bracket_entries(bracket):
        normalized = _normalize_team_name(entry)
        if not normalized:
            continue
        if normalized in lookup:
            resolved.add(lookup[normalized])
            continue
        close = difflib.get_close_matches(normalized, keys, n=1, cutoff=0.90)
        if close:
            resolved.add(lookup[close[0]])
        else:
            missing.append(entry)
    if not resolved:
        raise ValueError(f"Unable to resolve any teams from bracket file {bracket_file}.")
    if missing:
        unique = sorted({name for name in missing if name})
        raise ValueError(
            f"Unable to resolve TeamIDs for the following bracket entries: {unique}"
        )
    return resolved


def _restrict_matchups_to_field(matchups: pd.DataFrame, team_ids: Set[int]) -> pd.DataFrame:
    mask = matchups["Team1ID"].isin(team_ids) & matchups["Team2ID"].isin(team_ids)
    return matchups.loc[mask].reset_index(drop=True)


def _attach_team_names(df: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    names = teams[["TeamID", "TeamName"]].rename(columns={"TeamName": "Name"})
    merged = df.merge(
        names, left_on="Team1ID", right_on="TeamID", how="left"
    ).rename(columns={"Name": "Team1Name"})
    merged = merged.drop(columns=["TeamID"])
    merged = merged.merge(
        names, left_on="Team2ID", right_on="TeamID", how="left"
    ).rename(columns={"Name": "Team2Name"})
    merged = merged.drop(columns=["TeamID"])
    return merged


def run_inference(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    config = get_config()
    season = args.season or PRODUCTION_SEASON
    ensure_production_artifacts(
        config,
        data_dir=args.data_dir,
        feature_set=PRODUCTION_FEATURE_SET,
        force_retrain=args.force_retrain,
    )
    model, scaler, feature_cols, artifact_metadata = load_production_artifacts(config)

    matchups = load_matchups(args.sample_submission, season)
    field_team_ids: Optional[Set[int]] = None
    if args.field_only:
        bracket_path = args.bracket_file or DEFAULT_BRACKET_PATH
        field_team_ids = _resolve_field_team_ids(bracket_path, args.data_dir)
        matchups = _restrict_matchups_to_field(matchups, field_team_ids)
        if matchups.empty:
            raise ValueError(
                f"Bracket filtering removed all matchups; check {bracket_path} and prediction inputs."
            )

    context = build_context_for_season(args.data_dir, season)
    context_subset, coverage_stats = _subset_context_to_matchups(context, matchups)
    dataset = build_inference_dataset(context_subset, matchups)

    diff_cols = [col for col in dataset.columns if col.startswith("Diff_")]
    selected_cols = select_diff_columns(diff_cols, PRODUCTION_FEATURE_SET)
    missing_cols = sorted(set(feature_cols) - set(selected_cols))
    if missing_cols:
        raise ValueError(
            f"Feature mismatch. Saved model expects {len(feature_cols)} columns but "
            f"inference selected {len(selected_cols)}. Missing: {missing_cols}"
        )

    feature_matrix = dataset[feature_cols].copy()
    feature_matrix, fill_values = _fill_missing_numeric(feature_matrix, feature_cols)
    X_scaled = scaler.transform(feature_matrix)
    probs = model.predict_proba(X_scaled)[:, 1]

    predictions = dataset.copy()
    predictions["PredTeam1WinProb"] = probs
    predictions["PredTeam2WinProb"] = 1.0 - predictions["PredTeam1WinProb"]
    predictions["PredictedWinnerID"] = np.where(
        predictions["PredTeam1WinProb"] >= 0.5, predictions["Team1ID"], predictions["Team2ID"]
    )

    teams_df = data_loading.load_teams(args.data_dir)
    predictions = _attach_team_names(predictions, teams_df)
    predictions = predictions.rename(
        columns={
            "Team1_Seed": "Team1Seed",
            "Team2_Seed": "Team2Seed",
            "Team1_SeedNum": "Team1SeedNum",
            "Team2_SeedNum": "Team2SeedNum",
        }
    )
    predictions["model_name"] = PRODUCTION_MODEL_NAME
    predictions["feature_set"] = PRODUCTION_FEATURE_SET

    diagnostics = {
        "season": season,
        "matchup_rows": len(predictions),
        "unique_teams": coverage_stats["team_count"],
        "team_context_coverage": coverage_stats["coverage"],
        "feature_fill_values": fill_values,
        "artifact_metadata": artifact_metadata,
        "sample_submission": str(args.sample_submission),
        "field_only": bool(args.field_only),
        "bracket_file": str(args.bracket_file) if args.field_only else None,
        "field_team_count": len(field_team_ids) if field_team_ids else None,
    }
    return predictions, diagnostics


def write_outputs(
    predictions: pd.DataFrame,
    diagnostics: Dict[str, Any],
    config,
    output_path: Optional[Path] = None,
) -> None:
    predictions_dir = get_predictions_dir(config)
    ensure_parent_dir(predictions_dir / "placeholder.txt")

    if output_path:
        csv_path = output_path
        metadata_path = csv_path.with_name(f"{csv_path.stem}_metadata.json")
    else:
        csv_path = predictions_dir / f"{diagnostics['season']}_matchup_predictions.csv"
        metadata_path = predictions_dir / f"{diagnostics['season']}_prediction_metadata.json"

    ordered_cols = [
        "MatchupID",
        "Season",
        "Team1ID",
        "Team1Name",
        "Team1Seed",
        "Team1SeedNum",
        "Team2ID",
        "Team2Name",
        "Team2Seed",
        "Team2SeedNum",
        "PredTeam1WinProb",
        "PredTeam2WinProb",
        "PredictedWinnerID",
        "model_name",
        "feature_set",
    ]
    extra_cols = [col for col in predictions.columns if col not in ordered_cols]
    predictions[ordered_cols + extra_cols].to_csv(csv_path, index=False)

    diagnostics = {
        **diagnostics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prediction_file": str(csv_path),
    }
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(diagnostics, fp, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2026 matchup predictions.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing Kaggle CSVs. Defaults to config paths.",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/raw/SampleSubmissionStage2.csv"),
        help="Path to the sample submission file used for matchup IDs.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=PRODUCTION_SEASON,
        help="Season to score (defaults to 2026).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain the production model artifacts even if they exist.",
    )
    parser.add_argument(
        "--field-only",
        action="store_true",
        help="Restrict predictions to only the teams inside --bracket-file.",
    )
    parser.add_argument(
        "--bracket-file",
        type=Path,
        default=DEFAULT_BRACKET_PATH,
        help="Bracket JSON used when --field-only is enabled.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output CSV path (otherwise season-based default is used).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, diagnostics = run_inference(args)
    config = get_config()
    write_outputs(predictions, diagnostics, config, args.output)
    print(
        f"Generated {diagnostics['matchup_rows']} matchup predictions for season {diagnostics['season']}."
    )
    print(f"Outputs saved to {get_predictions_dir(config)}")


if __name__ == "__main__":
    main()
