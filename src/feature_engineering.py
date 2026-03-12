from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .feature_metadata import ADVANCED_FEATURE_COLUMNS


def parse_numeric_seed(seed_str: str | float | None) -> int | None:
    """Extract the numeric portion of a tournament seed string."""

    if seed_str is None or (isinstance(seed_str, float) and np.isnan(seed_str)):
        return None
    if not isinstance(seed_str, str):
        seed_str = str(seed_str)
    digits = "".join(ch for ch in seed_str if ch.isdigit())
    return int(digits) if digits else None


def _compute_possessions(fga: pd.Series, off_reb: pd.Series, turnovers: pd.Series, fta: pd.Series) -> pd.Series:
    return fga - off_reb + turnovers + 0.44 * fta


def _map_loser_loc(winner_loc: pd.Series) -> pd.Series:
    mapping = {"H": "A", "A": "H", "N": "N"}
    return winner_loc.replace(mapping)


def _build_team_game_view(results: pd.DataFrame) -> pd.DataFrame:
    """Convert winner/loser rows into a team-centric long format."""

    if "WLoc" in results.columns:
        winner_loc = results["WLoc"].fillna("N")
    else:
        winner_loc = pd.Series(["N"] * len(results), index=results.index)
    loser_loc = _map_loser_loc(winner_loc)

    winners = pd.DataFrame(
        {
            "Season": results["Season"],
            "DayNum": results["DayNum"],
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
            "OppOffReb": results["LOR"],
            "OppDefReb": results["LDR"],
            "OppTurnovers": results["LTO"],
            "TeamLoc": winner_loc,
            "IsWin": 1,
            "Possessions": _compute_possessions(results["WFGA"], results["WOR"], results["WTO"], results["WFTA"]),
            "OpponentPossessions": _compute_possessions(results["LFGA"], results["LOR"], results["LTO"], results["LFTA"]),
            "TeamRebounds": results["WOR"] + results["WDR"],
            "OpponentRebounds": results["LOR"] + results["LDR"],
        }
    )

    losers = pd.DataFrame(
        {
            "Season": results["Season"],
            "DayNum": results["DayNum"],
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
            "OppOffReb": results["WOR"],
            "OppDefReb": results["WDR"],
            "OppTurnovers": results["WTO"],
            "TeamLoc": loser_loc,
            "IsWin": 0,
            "Possessions": _compute_possessions(results["LFGA"], results["LOR"], results["LTO"], results["LFTA"]),
            "OpponentPossessions": _compute_possessions(results["WFGA"], results["WOR"], results["WTO"], results["WFTA"]),
            "TeamRebounds": results["LOR"] + results["LDR"],
            "OpponentRebounds": results["WOR"] + results["WDR"],
        }
    )

    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["NeutralGame"] = (team_games["TeamLoc"] == "N").astype(int)
    return team_games


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def _compute_recent_form(team_games: pd.DataFrame, window: int) -> pd.DataFrame:
    records = []
    grouped = team_games.sort_values(["Season", "TeamID", "DayNum"]).groupby(["Season", "TeamID"])
    for (season, team_id), df in grouped:
        tail = df.tail(window)
        games = len(tail)
        if games == 0:
            stats = {
                f"Recent{window}WinPct": 0.0,
                f"Recent{window}ScoringMargin": 0.0,
                f"Recent{window}OffRating": 0.0,
                f"Recent{window}DefRating": 0.0,
                f"Recent{window}EffectiveFG": 0.0,
            }
        else:
            points = tail["PointsFor"].sum()
            opp_points = tail["PointsAgainst"].sum()
            poss = tail["Possessions"].sum()
            opp_poss = tail["OpponentPossessions"].sum()
            stats = {
                f"Recent{window}WinPct": tail["IsWin"].mean(),
                f"Recent{window}ScoringMargin": (points - opp_points) / games,
                f"Recent{window}OffRating": float(points / poss) if poss else 0.0,
                f"Recent{window}DefRating": float(opp_points / opp_poss) if opp_poss else 0.0,
                f"Recent{window}EffectiveFG": float(
                    (tail["FGM"].sum() + 0.5 * tail["FG3M"].sum()) / tail["FGA"].sum()
                ) if tail["FGA"].sum() else 0.0,
            }
        stats.update({"Season": season, "TeamID": team_id})
        records.append(stats)
    return pd.DataFrame(records)


def _compute_neutral_site_stats(team_games: pd.DataFrame) -> pd.DataFrame:
    neutral_games = team_games[team_games["TeamLoc"] == "N"]
    if neutral_games.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "NeutralWinPercentage", "NeutralScoringMargin"])
    grouped = (
        neutral_games.groupby(["Season", "TeamID"])
        .agg(
            NeutralWinPercentage=("IsWin", "mean"),
            NeutralGameCount=("IsWin", "size"),
            PointsForSum=("PointsFor", "sum"),
            PointsAgainstSum=("PointsAgainst", "sum"),
        )
        .reset_index()
    )
    grouped["NeutralScoringMargin"] = _safe_divide(
        grouped["PointsForSum"] - grouped["PointsAgainstSum"],
        grouped["NeutralGameCount"],
    )
    return grouped[["Season", "TeamID", "NeutralWinPercentage", "NeutralScoringMargin"]]


def _compute_adjusted_margin(team_games: pd.DataFrame, base_frame: pd.DataFrame) -> pd.DataFrame:
    lookup = base_frame[["Season", "TeamID", "ScoringMargin"]].rename(columns={"TeamID": "LookupTeamID"})
    merged = team_games.merge(
        lookup,
        left_on=["Season", "OpponentID"],
        right_on=["Season", "LookupTeamID"],
        how="left",
    )
    avg = (
        merged.groupby(["Season", "TeamID"], as_index=False)["ScoringMargin"].mean().rename(
            columns={"ScoringMargin": "AvgOpponentMargin"}
        )
    )
    return avg


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
        OppOffRebSum=("OppOffReb", "sum"),
        OppDefRebSum=("OppDefReb", "sum"),
        OppTurnoversSum=("OppTurnovers", "sum"),
        PossessionSum=("Possessions", "sum"),
        OpponentPossessionSum=("OpponentPossessions", "sum"),
        TeamRebSum=("TeamRebounds", "sum"),
        OpponentRebSum=("OpponentRebounds", "sum"),
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
        "EffectiveFGPercent": _safe_divide(
            sums["FGMSum"] + 0.5 * sums["FG3MSum"], sums["FGASum"]
        ),
        "TrueShootingPercent": _safe_divide(
            sums["PointsForSum"], 2 * (sums["FGASum"] + 0.44 * sums["FTASum"])
        ),
        "ThreePointRate": _safe_divide(sums["FG3ASum"], sums["FGASum"]),
        "FreeThrowRate": _safe_divide(sums["FTASum"], sums["FGASum"]),
        "OffensiveRating": _safe_divide(sums["PointsForSum"], sums["PossessionSum"]),
        "DefensiveRating": _safe_divide(sums["PointsAgainstSum"], sums["OpponentPossessionSum"]),
        "NetRating": _safe_divide(sums["PointsForSum"], sums["PossessionSum"]) - _safe_divide(
            sums["PointsAgainstSum"], sums["OpponentPossessionSum"]
        ),
        "ReboundMargin": _safe_divide(
            sums["TeamRebSum"] - sums["OpponentRebSum"], games
        ),
        "OffensiveReboundRate": _safe_divide(sums["OffRebSum"], sums["OffRebSum"] + sums["OppDefRebSum"]),
        "TurnoverMargin": _safe_divide(sums["TurnoversSum"] - sums["OppTurnoversSum"], games),
        "AssistRate": _safe_divide(sums["AssistsSum"], sums["FGMSum"]),
    }

    assist_turnover = _safe_divide(sums["AssistsSum"], sums["TurnoversSum"])
    features["AssistToTurnoverRatio"] = assist_turnover

    frame = pd.DataFrame(features)

    recent5 = _compute_recent_form(team_games, 5)
    recent10 = _compute_recent_form(team_games, 10)
    neutral_stats = _compute_neutral_site_stats(team_games)

    frame = frame.merge(recent5, on=["Season", "TeamID"], how="left")
    frame = frame.merge(recent10, on=["Season", "TeamID"], how="left")
    frame = frame.merge(neutral_stats, on=["Season", "TeamID"], how="left")

    adjusted = _compute_adjusted_margin(team_games, frame)
    frame = frame.merge(adjusted, on=["Season", "TeamID"], how="left")
    frame["AdjustedScoringMargin"] = frame["ScoringMargin"] - frame["AvgOpponentMargin"].fillna(0)
    frame = frame.drop(columns=["AvgOpponentMargin"], errors="ignore")

    numeric_cols = frame.select_dtypes(include=[np.number]).columns
    frame[numeric_cols] = frame[numeric_cols].fillna(0)
    return frame.sort_values(["Season", "TeamID"]).reset_index(drop=True)
