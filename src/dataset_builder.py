from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd

from .feature_engineering import build_team_season_features, parse_numeric_seed


def summarize_massey_ordinals(
    massey_df: pd.DataFrame, system_name: str | None = None
) -> pd.DataFrame:
    df = massey_df.copy()
    if system_name:
        df = df[df["SystemName"] == system_name]
    summary = (
        df.groupby(["Season", "TeamID"], as_index=False)["OrdinalRank"].mean()
        if not df.empty
        else pd.DataFrame(columns=["Season", "TeamID", "OrdinalRank"])
    )
    return summary.rename(columns={"OrdinalRank": "MasseyOrdinal"})


def prepare_team_context(
    regular_season_detailed: pd.DataFrame,
    seeds: pd.DataFrame,
    massey_summary: Optional[pd.DataFrame] = None,
    supplemental_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    team_features = build_team_season_features(regular_season_detailed)
    seed_features = seeds.copy()
    seed_features["SeedNum"] = seed_features["Seed"].apply(parse_numeric_seed)
    context = team_features.merge(
        seed_features[["Season", "TeamID", "Seed", "SeedNum"]],
        on=["Season", "TeamID"],
        how="left",
    )
    if massey_summary is not None and not massey_summary.empty:
        context = context.merge(
            massey_summary, on=["Season", "TeamID"], how="left"
        )
    if supplemental_features is not None and not supplemental_features.empty:
        context = context.merge(
            supplemental_features, on=["Season", "TeamID"], how="left"
        )
    return context


def _attach_team_context(
    games: pd.DataFrame,
    context: pd.DataFrame,
    source_col: str,
    prefix: str,
    target_id_col: str,
) -> pd.DataFrame:
    merged = games.merge(
        context,
        how="left",
        left_on=["Season", source_col],
        right_on=["Season", "TeamID"],
    )
    rename_map = {
        col: f"{prefix}{col}"
        for col in context.columns
        if col not in {"Season", "TeamID"}
    }
    merged = merged.rename(columns={source_col: target_id_col, **rename_map})
    merged = merged.drop(columns=["TeamID"])
    return merged


def _build_ordered_matchups(
    games: pd.DataFrame,
    context: pd.DataFrame,
    team_a_col: str,
    team_b_col: str,
    label: int,
    diff_feature_names: Sequence[str],
) -> pd.DataFrame:
    ordered = games.copy()
    ordered = _attach_team_context(ordered, context, team_a_col, "Team1_", "Team1ID")
    ordered = _attach_team_context(ordered, context, team_b_col, "Team2_", "Team2ID")
    ordered["Label"] = label

    for feature in diff_feature_names:
        col_a = f"Team1_{feature}"
        col_b = f"Team2_{feature}"
        if col_a in ordered.columns and col_b in ordered.columns:
            ordered[f"Diff_{feature}"] = ordered[col_a] - ordered[col_b]

    ordered["MatchupID"] = (
        ordered["Season"].astype(str)
        + "_"
        + ordered["Team1ID"].astype(str)
        + "_"
        + ordered["Team2ID"].astype(str)
    )
    return ordered


def build_matchup_dataset(
    regular_season_detailed: pd.DataFrame,
    tourney_compact: pd.DataFrame,
    seeds: pd.DataFrame,
    massey_ordinals: Optional[pd.DataFrame] = None,
    massey_system: str | None = None,
    supplemental_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    massey_summary = (
        summarize_massey_ordinals(massey_ordinals, massey_system)
        if massey_ordinals is not None
        else None
    )
    context = prepare_team_context(
        regular_season_detailed,
        seeds,
        massey_summary,
        supplemental_features=supplemental_features,
    )
    numeric_features = [
        col
        for col in context.columns
        if col
        not in {
            "Season",
            "TeamID",
            "Seed",
        }
    ]

    base_games = tourney_compact[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]]

    positive = _build_ordered_matchups(
        base_games,
        context,
        team_a_col="WTeamID",
        team_b_col="LTeamID",
        label=1,
        diff_feature_names=numeric_features,
    )
    negative = _build_ordered_matchups(
        base_games,
        context,
        team_a_col="LTeamID",
        team_b_col="WTeamID",
        label=0,
        diff_feature_names=numeric_features,
    )
    dataset = pd.concat([positive, negative], ignore_index=True)

    metadata_cols: List[str] = [
        "MatchupID",
        "Season",
        "DayNum",
        "Team1ID",
        "Team2ID",
        "Team1_Seed",
        "Team1_SeedNum",
        "Team1_MasseyOrdinal",
        "Team2_Seed",
        "Team2_SeedNum",
        "Team2_MasseyOrdinal",
        "Label",
    ]
    metadata_cols = [col for col in metadata_cols if col in dataset.columns]
    diff_cols = sorted([col for col in dataset.columns if col.startswith("Diff_")])
    ordered_cols = metadata_cols + diff_cols
    dataset = dataset[ordered_cols]
    return dataset
