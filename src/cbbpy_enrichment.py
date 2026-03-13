from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
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


def _ensure_boxscores(season: int, start: date, end: date, cache_dir: Path, refresh: bool) -> Tuple[pd.DataFrame, CBBpyDiagnostics]:
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

    if cache_path.exists() and not refresh:
        diagnostics.fetched_from_cache = True
        box_df = pd.read_csv(cache_path)
        diagnostics.games_scraped = int(box_df["game_id"].nunique()) if not box_df.empty else 0
        diagnostics.teams_scraped = int(box_df["team"].nunique()) if not box_df.empty else 0
        return box_df, diagnostics

    _, box_df, _ = mens_scraper.get_games_range(
        start.isoformat(),
        end.isoformat(),
        info=False,
        box=True,
        pbp=False,
    )
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
    for column in BOX_NUMERIC_COLUMNS:
        work[column] = pd.to_numeric(work.get(column, 0), errors="coerce").fillna(0.0)

    grouped = work.groupby(["game_id", "team"], as_index=False)[BOX_NUMERIC_COLUMNS].sum()
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
    output.insert(0, "Season", season)
    return output, {
        "season": season,
        "teams_matched": int(mapped["TeamID"].nunique()),
        "teams_unmatched": unmatched,
        "total_scraped_teams": int(team_summary["SourceTeamName"].nunique()),
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
) -> Path:
    start, end = _parse_dates_for_season(season, start_date, end_date)
    box_df, diagnostics = _ensure_boxscores(season, start, end, cache_dir, refresh)
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
    )
    print(f"CBBpy enrichment complete. Features saved to {output_path}")


if __name__ == "__main__":
    main()
