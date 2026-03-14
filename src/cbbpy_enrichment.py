from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
from rapidfuzz import process as fuzz_process, distance as fuzz_distance

from .config import get_config
from . import data_loading
from .data_loading import _resolve_path
from .supplemental_ncaa import _normalize_name

try:  # pragma: no cover - import failure is handled at runtime
    import cbbpy.mens_scraper as mens_scraper
except ImportError:  # pragma: no cover - handled in CLI
    mens_scraper = None


BOX_NUMERIC_COLUMNS = [
    "pts",
    "fgm",
    "fga",
    "2pm",
    "2pa",
    "3pm",
    "3pa",
    "ftm",
    "fta",
    "reb",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "to",
]


def _compute_recent_cbbpy_features(games: pd.DataFrame) -> pd.DataFrame:
    if "game_date" not in games.columns:
        return pd.DataFrame()
    recent = games.dropna(subset=["game_date"]).copy()
    if recent.empty:
        return pd.DataFrame()
    recent["game_date"] = pd.to_datetime(recent["game_date"], errors="coerce")
    recent = recent.dropna(subset=["game_date"])
    if recent.empty:
        return pd.DataFrame()
    recent = recent.sort_values(["team", "game_date"])
    recent["win_float"] = recent["win"].astype(float)
    recent["margin_float"] = recent["margin"].astype(float)
    for window in (5, 10):
        recent[f"win_roll_{window}"] = (
            recent.groupby("team")["win_float"]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        recent[f"margin_roll_{window}"] = (
            recent.groupby("team")["margin_float"]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    summary = (
        recent.groupby("team")
        .agg(
            CBBpy_Last5_WinPct=("win_roll_5", "last"),
            CBBpy_Last10_WinPct=("win_roll_10", "last"),
            CBBpy_Last5_ScoringMargin=("margin_roll_5", "last"),
            CBBpy_Last10_ScoringMargin=("margin_roll_10", "last"),
        )
        .reset_index()
    )
    return summary


@dataclass
class CBBpyDiagnostics:
    season: int
    start_date: str
    end_date: str
    cache_file: Optional[str] = None
    features_file: Optional[str] = None
    games_scraped: int = 0
    teams_scraped: int = 0
    teams_matched: int = 0
    teams_unmatched: List[str] = field(default_factory=list)
    fetched_from_cache: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        return payload


def _infer_default_season() -> int:
    today = datetime.today()
    if today.month >= 7:
        return today.year + 1
    return today.year


def _parse_dates_for_season(season: int, start_override: Optional[str], end_override: Optional[str]) -> Tuple[date, date]:
    start_default = date(season - 1, 11, 1)
    end_default = min(date(season, 5, 1), datetime.today().date())
    start = datetime.strptime(start_override, "%Y-%m-%d").date() if start_override else start_default
    end = datetime.strptime(end_override, "%Y-%m-%d").date() if end_override else end_default
    if end < start:
        raise ValueError("End date must be after start date for CBBpy scraping.")
    return start, end


def _sanitize_date_component(value: date) -> str:
    return value.strftime("%Y%m%d")


def _fetch_boxscores_chunked(start: date, end: date, chunk_days: int) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    frames_box: List[pd.DataFrame] = []
    frames_info: List[pd.DataFrame] = []
    cursor = start
    window = max(1, chunk_days)
    info_supported = True
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=window - 1), end)
        try:
            info_chunk, box_chunk, _ = mens_scraper.get_games_range(
                cursor.isoformat(),
                chunk_end.isoformat(),
                info=True,
                box=True,
                pbp=False,
            )
        except Exception:
            info_supported = False
            _, box_chunk, _ = mens_scraper.get_games_range(
                cursor.isoformat(),
                chunk_end.isoformat(),
                info=False,
                box=True,
                pbp=False,
            )
            info_chunk = None
        if info_chunk is not None and not info_chunk.empty:
            frames_info.append(info_chunk.copy())
        if box_chunk is not None and not box_chunk.empty:
            frames_box.append(box_chunk.copy())
        cursor = chunk_end + timedelta(days=1)
    info_df = pd.concat(frames_info, ignore_index=True) if frames_info else pd.DataFrame()
    box_df = pd.concat(frames_box, ignore_index=True) if frames_box else pd.DataFrame()
    if not box_df.empty and not info_df.empty and {"game_id", "game_date"}.issubset(info_df.columns):
        info_subset = info_df[["game_id", "game_date"]].drop_duplicates("game_id")
        box_df = box_df.merge(info_subset, on="game_id", how="left")
    return box_df, info_df, info_supported


def _ensure_boxscores(season: int, start: date, end: date, cache_dir: Path, refresh: bool, chunk_days: int) -> Tuple[pd.DataFrame, CBBpyDiagnostics]:
    if mens_scraper is None:
        raise RuntimeError("cbbpy is not installed. Run `pip install cbbpy` to enable current-season enrichment.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"cbbpy_box_{season}_{_sanitize_date_component(start)}_{_sanitize_date_component(end)}.csv"
    cache_path = cache_dir / cache_name
    diagnostics = CBBpyDiagnostics(
        season=season,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        cache_file=str(cache_path),
    )

    diagnostics.notes.append(f"chunk_days={chunk_days}")

    if cache_path.exists() and not refresh:
        diagnostics.fetched_from_cache = True
        box_df = pd.read_csv(cache_path)
        diagnostics.games_scraped = int(box_df["game_id"].nunique()) if not box_df.empty else 0
        diagnostics.teams_scraped = int(box_df["team"].nunique()) if not box_df.empty else 0
        if "game_date" not in box_df.columns:
            diagnostics.notes.append("Cached boxscores missing game_date; rerun with --refresh to enable recency features.")
        return box_df, diagnostics

    box_df, _, info_supported = _fetch_boxscores_chunked(start, end, chunk_days)
    if not info_supported:
        diagnostics.notes.append("CBBpy info feed unavailable for part of the range; game_date column may be missing.")
    box_df.to_csv(cache_path, index=False)
    diagnostics.games_scraped = int(box_df["game_id"].nunique()) if not box_df.empty else 0
    diagnostics.teams_scraped = int(box_df["team"].nunique()) if not box_df.empty else 0
    return box_df, diagnostics


def _load_team_spellings(data_dir: Path) -> Optional[pd.DataFrame]:
    try:
        path = _resolve_path("MTeamSpellings.csv", data_dir)
    except FileNotFoundError:
        return None
    return pd.read_csv(path)


def _build_team_lookup(teams_df: pd.DataFrame, spellings_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    lookup_frames = []
    base = teams_df[["TeamID", "TeamName"]].copy()
    base["name_key"] = base["TeamName"].map(_normalize_name)
    lookup_frames.append(base[["TeamID", "name_key"]])
    if spellings_df is not None and not spellings_df.empty:
        alt = spellings_df[["TeamID", "TeamNameSpelling"]].copy()
        alt["name_key"] = alt["TeamNameSpelling"].map(_normalize_name)
        lookup_frames.append(alt[["TeamID", "name_key"]])
    combined = pd.concat(lookup_frames, ignore_index=True)
    combined = combined.dropna(subset=["name_key"]).drop_duplicates(subset=["TeamID", "name_key"])
    return combined


def _build_string_to_id_map(teams_df: pd.DataFrame, spellings_df: Optional[pd.DataFrame]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for _, row in teams_df.iterrows():
        mapping[str(row["TeamName"])] = int(row["TeamID"])
    if spellings_df is not None and not spellings_df.empty:
        for _, row in spellings_df.iterrows():
            mapping[str(row["TeamNameSpelling"])] = int(row["TeamID"])
    return mapping


def build_team_features_from_boxscores(
    box_df: pd.DataFrame,
    season: int,
    teams_df: pd.DataFrame,
    spellings_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if box_df.empty:
        return pd.DataFrame(columns=["Season", "TeamID"]), {
            "season": season,
            "notes": ["No CBBpy boxscore rows were available for the requested range."],
            "teams_matched": 0,
            "teams_unmatched": [],
        }

    work = box_df.copy()
    work["team"] = work["team"].astype(str)
    work["game_id"] = work["game_id"].astype(str)
    if "game_date" in work.columns:
        work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    for column in BOX_NUMERIC_COLUMNS:
        work[column] = pd.to_numeric(work.get(column, 0), errors="coerce").fillna(0.0)

    agg_map = {col: "sum" for col in BOX_NUMERIC_COLUMNS}
    if "game_date" in work.columns:
        agg_map["game_date"] = "max"
    grouped = work.groupby(["game_id", "team"], as_index=False).agg(agg_map)
    grouped["possessions"] = (
        grouped["fga"] - grouped["oreb"] + grouped["to"] + 0.44 * grouped["fta"]
    )

    opponent_stats = grouped[["game_id", "team", "pts", "possessions"]].rename(
        columns={"team": "opp_team", "pts": "opp_pts", "possessions": "opp_possessions"}
    )
    games = grouped.merge(opponent_stats, on="game_id")
    games = games[games["team"] != games["opp_team"]].drop_duplicates(subset=["game_id", "team"])
    if games.empty:
        return pd.DataFrame(columns=["Season", "TeamID"]), {
            "season": season,
            "notes": ["CBBpy returned boxscores but no complete games were available."],
            "teams_matched": 0,
            "teams_unmatched": [],
        }

    games["win"] = (games["pts"] > games["opp_pts"]).astype(int)
    games["margin"] = games["pts"] - games["opp_pts"]

    team_summary = (
        games.groupby("team")
        .agg(
            games_played=("game_id", "count"),
            wins=("win", "sum"),
            points_for=("pts", "sum"),
            points_against=("opp_pts", "sum"),
            possessions=("possessions", "sum"),
            opp_possessions=("opp_possessions", "sum"),
            avg_margin=("margin", "mean"),
        )
        .reset_index()
    )
    team_summary.rename(columns={"team": "SourceTeamName"}, inplace=True)

    def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denom = denominator.replace(0, np.nan)
        return numerator / denom

    team_summary["CBBpy_GamesPlayed"] = team_summary["games_played"]
    team_summary["CBBpy_WinPct"] = _safe_rate(team_summary["wins"], team_summary["games_played"]).fillna(0.0)
    team_summary["CBBpy_PointsForPerGame"] = _safe_rate(team_summary["points_for"], team_summary["games_played"]).fillna(0.0)
    team_summary["CBBpy_PointsAgainstPerGame"] = _safe_rate(
        team_summary["points_against"], team_summary["games_played"]
    ).fillna(0.0)
    team_summary["CBBpy_ScoringMargin"] = team_summary["avg_margin"].fillna(0.0)
    team_summary["CBBpy_PossessionsPerGame"] = _safe_rate(
        team_summary["possessions"], team_summary["games_played"]
    ).fillna(0.0)
    team_summary["CBBpy_OffRating"] = _safe_rate(
        100 * team_summary["points_for"], team_summary["possessions"]
    ).fillna(0.0)
    team_summary["CBBpy_DefRating"] = _safe_rate(
        100 * team_summary["points_against"], team_summary["opp_possessions"]
    ).fillna(0.0)
    team_summary["CBBpy_NetRating"] = team_summary["CBBpy_OffRating"] - team_summary["CBBpy_DefRating"]

    lookup = _build_team_lookup(teams_df, spellings_df)
    string_to_id = _build_string_to_id_map(teams_df, spellings_df)
    team_summary["name_key"] = team_summary["SourceTeamName"].map(_normalize_name)
    mapped = team_summary.merge(lookup, on="name_key", how="left")
    unmatched_mask = mapped["TeamID"].isna()
    if unmatched_mask.any() and string_to_id:
        choices = list(string_to_id.keys())
        for idx in mapped[unmatched_mask].index:
            candidate = mapped.at[idx, "SourceTeamName"]
            result = fuzz_process.extractOne(
                candidate,
                choices,
                scorer=fuzz_distance.JaroWinkler.normalized_similarity,
                score_cutoff=0.82,
            )
            if result:
                mapped.at[idx, "TeamID"] = string_to_id[result[0]]
    unmatched = (
        mapped.loc[mapped["TeamID"].isna(), "SourceTeamName"]
        .dropna()
        .drop_duplicates()
        .head(15)
        .tolist()
    )

    mapped = mapped.dropna(subset=["TeamID"]).copy()
    if mapped.empty:
        return pd.DataFrame(columns=["Season", "TeamID"]), {
            "season": season,
            "notes": ["CBBpy data could not be mapped to Kaggle TeamIDs."],
            "teams_matched": 0,
            "teams_unmatched": unmatched,
        }

    mapped["TeamID"] = mapped["TeamID"].astype(int)
    feature_cols = [
        "CBBpy_GamesPlayed",
        "CBBpy_WinPct",
        "CBBpy_PointsForPerGame",
        "CBBpy_PointsAgainstPerGame",
        "CBBpy_ScoringMargin",
        "CBBpy_PossessionsPerGame",
        "CBBpy_OffRating",
        "CBBpy_DefRating",
        "CBBpy_NetRating",
    ]
    output = mapped[["TeamID", "SourceTeamName"] + feature_cols].copy()
    recent_features = _compute_recent_cbbpy_features(games)
    if not recent_features.empty:
        output = output.merge(
            recent_features,
            left_on="SourceTeamName",
            right_on="team",
            how="left",
        )
        output = output.drop(columns=["team"], errors="ignore")
        for recent_col in [
            "CBBpy_Last5_WinPct",
            "CBBpy_Last10_WinPct",
            "CBBpy_Last5_ScoringMargin",
            "CBBpy_Last10_ScoringMargin",
        ]:
            if recent_col not in output.columns:
                output[recent_col] = np.nan
    output.insert(0, "Season", season)
    return output, {
        "season": season,
        "teams_matched": int(mapped["TeamID"].nunique()),
        "teams_unmatched": unmatched,
        "total_scraped_teams": int(team_summary["SourceTeamName"].nunique()),
    }


def _save_feature_summary(features_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "season": int(features_df["Season"].min()) if not features_df.empty else None,
        "row_count": int(len(features_df)),
        "feature_columns": [col for col in features_df.columns if col not in {"Season", "TeamID"}],
    }
    path.write_text(json.dumps(summary, indent=2))


def _build_seed_coverage_report(
    features_df: pd.DataFrame,
    data_dir: Path,
    season: int,
) -> Dict[str, object]:
    seeds = data_loading.load_tourney_seeds(data_dir)
    teams = data_loading.load_teams(data_dir)
    seeds_season = seeds[seeds["Season"] == season][["TeamID", "Seed"]].drop_duplicates()
    coverage_df = seeds_season.merge(teams[["TeamID", "TeamName"]], on="TeamID", how="left")
    coverage_df["has_cbbpy"] = coverage_df["TeamID"].isin(features_df["TeamID"])
    coverage = float(coverage_df["has_cbbpy"].mean()) if not coverage_df.empty else 0.0
    unmatched = (
        coverage_df.loc[~coverage_df["has_cbbpy"], "TeamName"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return {
        "season": season,
        "target_team_count": int(len(coverage_df)),
        "teams_with_cbbpy": int(coverage_df["has_cbbpy"].sum()),
        "coverage_rate": coverage,
        "teams_missing_cbbpy": unmatched,
    }


def save_cbbpy_reports(diagnostics: CBBpyDiagnostics, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "cbbpy_fetch_summary.json"
    path.write_text(json.dumps(diagnostics.to_dict(), indent=2))


def run_enrichment(
    season: int,
    data_dir: Path,
    cache_dir: Path,
    reports_dir: Path,
    start_date: Optional[str],
    end_date: Optional[str],
    refresh: bool,
    chunk_days: int,
) -> Path:
    start, end = _parse_dates_for_season(season, start_date, end_date)
    box_df, diagnostics = _ensure_boxscores(season, start, end, cache_dir, refresh, chunk_days)
    teams_df = data_loading.load_teams(data_dir)
    spellings_df = _load_team_spellings(data_dir)
    features_df, mapping_diag = build_team_features_from_boxscores(box_df, season, teams_df, spellings_df)
    if mapping_diag:
        diagnostics.teams_matched = mapping_diag.get("teams_matched", 0)
        diagnostics.teams_unmatched = mapping_diag.get("teams_unmatched", [])
    output_path = cache_dir / f"cbbpy_team_features_{season}.csv"
    features_df.to_csv(output_path, index=False)
    diagnostics.features_file = str(output_path)
    save_cbbpy_reports(diagnostics, reports_dir)
    if not features_df.empty:
        _save_feature_summary(features_df, reports_dir / "cbbpy_feature_summary.json")
        coverage_report = _build_seed_coverage_report(features_df, data_dir, season)
        coverage_path = reports_dir / f"cbbpy_coverage_{season}.json"
        coverage_path.write_text(json.dumps(coverage_report, indent=2))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch current-season CBBpy data and build enrichment features.")
    parser.add_argument("--season", type=int, default=None, help="Season year (e.g., 2026). Defaults to current season.")
    parser.add_argument("--start-date", type=str, default=None, help="Optional YYYY-MM-DD start date override.")
    parser.add_argument("--end-date", type=str, default=None, help="Optional YYYY-MM-DD end date override.")
    parser.add_argument("--refresh", action="store_true", help="Force re-scrape even if cache exists.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing Kaggle MTeams.csv (defaults to config raw data path).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where cached CBBpy files will be written (defaults to config.paths.cbbpy_cache_dir).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory for diagnostic JSON output (defaults to outputs/reports).",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=14,
        help="Number of days per scraping chunk (smaller windows reduce ESPN timeouts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    season = args.season if args.season is not None else _infer_default_season()
    data_dir = args.data_dir if args.data_dir else config.paths.raw_data_dir
    cache_dir = args.output_dir if args.output_dir else config.paths.cbbpy_cache_dir
    reports_dir = args.reports_dir if args.reports_dir else config.paths.outputs_dir / "reports"

    output_path = run_enrichment(
        season=season,
        data_dir=data_dir,
        cache_dir=cache_dir,
        reports_dir=reports_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        refresh=args.refresh,
        chunk_days=max(1, args.chunk_days),
    )
    print(f"CBBpy enrichment complete. Features saved to {output_path}")


if __name__ == "__main__":
    main()
