from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .feature_metadata import SUPPLEMENTAL_FEATURES

TABLE_FILENAMES = {
    "team_seasons": "mbb_historical_teams_seasons.csv",
    "team_games": "mbb_historical_teams_games.csv",
    "teams": "mbb_teams.csv",
}

MANUAL_TEAM_NAME_OVERRIDES = {
    "saintmarysca": "saint mary's (ca)",
    "saintmarys": "saint mary's (ca)",
    "uc santa barbara": "uc santa barbara",
    "brighamyoung": "byu",
}


@dataclass
class MappingDiagnostics:
    table: str
    method: str
    coverage: float
    total_rows: int
    matched_rows: int
    unmatched_examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class SupplementalDiagnostics:
    tables_requested: Dict[str, str]
    tables_found: Dict[str, bool] = field(default_factory=dict)
    mapping: List[MappingDiagnostics] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    season_coverage: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "tables_requested": self.tables_requested,
            "tables_found": self.tables_found,
            "mapping": [m.to_dict() for m in self.mapping],
            "feature_columns": self.feature_columns,
            "season_coverage": self.season_coverage,
            "notes": self.notes,
        }


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return MANUAL_TEAM_NAME_OVERRIDES.get(normalized, normalized)


def _load_table(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def _ensure_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _map_team_ids(
    frame: pd.DataFrame,
    teams_df: pd.DataFrame,
    table_name: str,
    diagnostics: SupplementalDiagnostics,
) -> pd.DataFrame:
    if frame.empty or teams_df is None or teams_df.empty:
        diagnostics.notes.append(f"Skipping {table_name}: missing required data.")
        return pd.DataFrame()

    df = frame.copy()
    season_col = _find_column(df, ["Season", "season", "Year", "year"])
    if season_col is None:
        diagnostics.notes.append(f"{table_name}: no season column found.")
        return pd.DataFrame()
    df["Season"] = _ensure_int_series(df[season_col])

    mania = teams_df[["TeamID", "TeamName"]].copy()
    mania["name_key"] = mania["TeamName"].fillna("").map(_normalize_name)
    mania_id_set = set(mania["TeamID"])

    team_id_col = _find_column(df, ["TeamID", "team_id", "ncaa_team_id"])
    mapping_method = "name"
    matched_rows = 0
    total_rows = len(df)
    unmatched_examples: List[str] = []

    if team_id_col is not None and df[team_id_col].isin(mania_id_set).any():
        df["TeamID"] = df[team_id_col].astype("Int64")
        df = df[df["TeamID"].isin(mania_id_set)]
        matched_rows = len(df)
        mapping_method = "id"
    else:
        name_col = _find_column(df, ["team_name", "TeamName", "school", "name", "team"])
        if name_col is None:
            diagnostics.notes.append(f"{table_name}: missing team name column for mapping.")
            return pd.DataFrame()
        df["name_key"] = df[name_col].fillna("").map(_normalize_name)
        merged = df.merge(mania[["TeamID", "name_key"]], on="name_key", how="left")
        unmatched = merged[merged["TeamID"].isna()]
        if not unmatched.empty:
            unmatched_examples = (
                unmatched[name_col].dropna().head(5).astype(str).unique().tolist()
            )
        df = merged[merged["TeamID"].notna()].copy()
        matched_rows = len(df)

    coverage = (matched_rows / total_rows) if total_rows else 0.0
    diagnostics.mapping.append(
        MappingDiagnostics(
            table=table_name,
            method=mapping_method,
            coverage=float(coverage),
            total_rows=total_rows,
            matched_rows=matched_rows,
            unmatched_examples=unmatched_examples,
        )
    )
    if matched_rows == 0:
        return pd.DataFrame()
    return df


def _aggregate_season_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    games_col = _find_column(df, ["games", "Games", "gp", "GP"])
    poss_col = _find_column(df, ["possessions", "Possessions", "poss", "Poss"])
    tempo_col = _find_column(df, ["tempo", "Tempo", "pace", "Pace"])
    net_col = _find_column(df, ["adj_em", "AdjEM", "net_rating", "NetRating", "srs", "SRS"])
    sos_col = _find_column(df, ["sos", "SOS", "strength_of_schedule"])

    agg_cols: Dict[str, str] = {}
    for col in [games_col, poss_col, tempo_col, net_col, sos_col]:
        if col:
            agg_cols[col] = "mean"

    grouped = df.groupby(["Season", "TeamID"], as_index=False).agg(agg_cols)
    features = grouped[["Season", "TeamID"]].copy()

    if poss_col and games_col:
        denom = grouped[games_col].replace(0, np.nan)
        features["SuppTempo"] = grouped[poss_col] / denom
    elif tempo_col:
        features["SuppTempo"] = grouped[tempo_col]

    if net_col:
        features["SuppAdjNetRating"] = grouped[net_col]
    if sos_col:
        features["SuppStrengthOfSchedule"] = grouped[sos_col]

    return features


def _aggregate_game_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    score_col = _find_column(df, ["team_score", "score", "points", "points_for"])
    opp_score_col = _find_column(df, ["opponent_score", "opp_score", "points_against"])
    location_col = _find_column(df, ["location", "game_location", "site", "game_type"])
    if score_col is None or opp_score_col is None:
        return pd.DataFrame()

    df = df.copy()
    df["Margin"] = df[score_col] - df[opp_score_col]
    df["IsWin"] = (df["Margin"] > 0).astype(int)

    close_games = df[df["Margin"].abs() <= 5]
    frames: List[pd.DataFrame] = []
    if not close_games.empty:
        agg = close_games.groupby(["Season", "TeamID"]).agg(
            CloseWins=("IsWin", "mean"),
            CloseGames=("IsWin", "size"),
        )
        agg["SuppCloseGameWinPct"] = agg["CloseWins"]
        frames.append(agg[["SuppCloseGameWinPct"]])

    if location_col:
        neutral_mask = df[location_col].fillna("").str.upper().str.startswith("N")
        neutral_games = df[neutral_mask]
        if not neutral_games.empty:
            neutral_agg = neutral_games.groupby(["Season", "TeamID"]).agg(
                NeutralWins=("IsWin", "mean"),
            )
            neutral_agg["SuppNeutralWinPct"] = neutral_agg["NeutralWins"]
            frames.append(neutral_agg[["SuppNeutralWinPct"]])

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, left_index=True, right_index=True, how="outer")
    merged = merged.reset_index()
    return merged


def build_supplemental_features(
    supplemental_dir: Path,
    teams_df: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, SupplementalDiagnostics]:
    diagnostics = SupplementalDiagnostics(tables_requested=TABLE_FILENAMES.copy())
    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in TABLE_FILENAMES.items():
        path = supplemental_dir / filename
        table = _load_table(path)
        tables[key] = table if table is not None else pd.DataFrame()
        diagnostics.tables_found[key] = table is not None

    seasons_df = tables["team_seasons"]
    if seasons_df.empty or teams_df is None or teams_df.empty:
        diagnostics.notes.append("Supplemental features skipped: required team season data missing.")
        return pd.DataFrame(columns=["Season", "TeamID"]), diagnostics

    mapped_seasons = _map_team_ids(seasons_df, teams_df, "team_seasons", diagnostics)
    if mapped_seasons.empty:
        diagnostics.notes.append("Supplemental features skipped: no mapped rows from team seasons table.")
        return pd.DataFrame(columns=["Season", "TeamID"]), diagnostics

    season_features = _aggregate_season_features(mapped_seasons)

    games_df = tables["team_games"]
    mapped_games = _map_team_ids(games_df, teams_df, "team_games", diagnostics) if not games_df.empty else pd.DataFrame()
    game_features = _aggregate_game_features(mapped_games) if not mapped_games.empty else pd.DataFrame()

    features = season_features
    if features.empty and not game_features.empty:
        features = game_features
    elif not features.empty and not game_features.empty:
        features = features.merge(game_features, on=["Season", "TeamID"], how="left")

    if features is None or features.empty:
        diagnostics.notes.append("Supplemental features computed to empty frame.")
        return pd.DataFrame(columns=["Season", "TeamID"]), diagnostics

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    features[numeric_cols] = features[numeric_cols].fillna(0.0)
    features = features.sort_values(["Season", "TeamID"]).reset_index(drop=True)
    ordered_cols = ["Season", "TeamID"] + [col for col in SUPPLEMENTAL_FEATURES if col in features.columns]
    features = features[[col for col in ordered_cols if col in features.columns]]

    diagnostics.feature_columns = [col for col in features.columns if col not in {"Season", "TeamID"}]
    if not features.empty:
        diagnostics.season_coverage = {
            "min_season": int(features["Season"].min()),
            "max_season": int(features["Season"].max()),
            "row_count": int(len(features)),
        }
    else:
        diagnostics.season_coverage = {}
    return features, diagnostics


def save_supplemental_report(diagnostics: SupplementalDiagnostics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostics.to_dict(), indent=2))
