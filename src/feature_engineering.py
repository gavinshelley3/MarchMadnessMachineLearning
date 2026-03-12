from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def parse_numeric_seed(seed_str: str | float | None) -> int | None:
    """Extract the numeric portion of a tournament seed string."""

    if seed_str is None or (isinstance(seed_str, float) and np.isnan(seed_str)):
        return None
    if not isinstance(seed_str, str):
        seed_str = str(seed_str)
    digits = "".join(ch for ch in seed_str if ch.isdigit())
    return int(digits) if digits else None


def _build_team_game_view(results: pd.DataFrame) -> pd.DataFrame:
    """Convert winner/loser rows into a team-centric long format."""

    winners = pd.DataFrame(
        {
            "Season": results["Season"],
            "TeamID": results["WTeamID"],
            "OpponentID": results["LTeamID"],
            "PointsFor": results["WScore"],
            "PointsAgainst": results["LScore"],
            "FGM": results["WFGM"],
            "FGA": results["WFGA"],
            "FG3M": results["WFGM3"],
            "FG3A": results["WFGA3"],
            "FTM": results["WFTM"],
            "FTA": results["WFTA"],
            "OffReb": results["WOR"],
            "DefReb": results["WDR"],
            "Assists": results["WAst"],
            "Turnovers": results["WTO"],
            "Steals": results["WStl"],
            "Blocks": results["WBlk"],
            "Fouls": results["WPF"],
            "IsWin": 1,
        }
    )

    losers = pd.DataFrame(
        {
            "Season": results["Season"],
            "TeamID": results["LTeamID"],
            "OpponentID": results["WTeamID"],
            "PointsFor": results["LScore"],
            "PointsAgainst": results["WScore"],
            "FGM": results["LFGM"],
            "FGA": results["LFGA"],
            "FG3M": results["LFGM3"],
            "FG3A": results["LFGA3"],
            "FTM": results["LFTM"],
            "FTA": results["LFTA"],
            "OffReb": results["LOR"],
            "DefReb": results["LDR"],
            "Assists": results["LAst"],
            "Turnovers": results["LTO"],
            "Steals": results["LStl"],
            "Blocks": results["LBlk"],
            "Fouls": results["LPF"],
            "IsWin": 0,
        }
    )

    return pd.concat([winners, losers], ignore_index=True)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def build_team_season_features(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regular season results into per-team season features."""

    team_games = _build_team_game_view(results)
    grouped = team_games.groupby(["Season", "TeamID"], as_index=False)
    sums = grouped.aggregate(
        GamesPlayed=("TeamID", "size"),
        Wins=("IsWin", "sum"),
        PointsForSum=("PointsFor", "sum"),
        PointsAgainstSum=("PointsAgainst", "sum"),
        FGMSum=("FGM", "sum"),
        FGASum=("FGA", "sum"),
        FG3MSum=("FG3M", "sum"),
        FG3ASum=("FG3A", "sum"),
        FTMSum=("FTM", "sum"),
        FTASum=("FTA", "sum"),
        OffRebSum=("OffReb", "sum"),
        DefRebSum=("DefReb", "sum"),
        AssistsSum=("Assists", "sum"),
        TurnoversSum=("Turnovers", "sum"),
        StealsSum=("Steals", "sum"),
        BlocksSum=("Blocks", "sum"),
        FoulsSum=("Fouls", "sum"),
    )

    games = sums["GamesPlayed"].replace(0, np.nan)

    features: Dict[str, pd.Series] = {
        "Season": sums["Season"],
        "TeamID": sums["TeamID"],
        "GamesPlayed": sums["GamesPlayed"],
        "WinPercentage": _safe_divide(sums["Wins"], games),
        "PointsPerGame": _safe_divide(sums["PointsForSum"], games),
        "PointsAllowedPerGame": _safe_divide(sums["PointsAgainstSum"], games),
        "ScoringMargin": _safe_divide(
            sums["PointsForSum"] - sums["PointsAgainstSum"], games
        ),
        "FGPercent": _safe_divide(sums["FGMSum"], sums["FGASum"]),
        "ThreePointPercent": _safe_divide(sums["FG3MSum"], sums["FG3ASum"]),
        "FreeThrowPercent": _safe_divide(sums["FTMSum"], sums["FTASum"]),
        "OffReboundsPerGame": _safe_divide(sums["OffRebSum"], games),
        "DefReboundsPerGame": _safe_divide(sums["DefRebSum"], games),
        "TotalReboundsPerGame": _safe_divide(
            sums["OffRebSum"] + sums["DefRebSum"], games
        ),
        "AssistsPerGame": _safe_divide(sums["AssistsSum"], games),
        "TurnoversPerGame": _safe_divide(sums["TurnoversSum"], games),
        "StealsPerGame": _safe_divide(sums["StealsSum"], games),
        "BlocksPerGame": _safe_divide(sums["BlocksSum"], games),
        "FoulsPerGame": _safe_divide(sums["FoulsSum"], games),
    }

    assist_turnover = _safe_divide(sums["AssistsSum"], sums["TurnoversSum"])
    features["AssistToTurnoverRatio"] = assist_turnover

    frame = pd.DataFrame(features)
    frame = frame.fillna(0)
    return frame.sort_values(["Season", "TeamID"]).reset_index(drop=True)
