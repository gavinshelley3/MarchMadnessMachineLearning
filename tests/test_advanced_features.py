import pandas as pd

from src.feature_engineering import build_team_season_features


def _base_rows():
    rows = []
    for day, wscore, lscore, wloc in [
        (10, 80, 70, "H"),
        (20, 75, 65, "A"),
        (30, 90, 60, "N"),
        (40, 85, 78, "N"),
        (50, 70, 68, "H"),
        (60, 88, 82, "A"),
    ]:
        rows.append(
            {
                "Season": 2024,
                "DayNum": day,
                "WTeamID": 1001,
                "LTeamID": 1002,
                "WScore": wscore,
                "LScore": lscore,
                "WLoc": wloc,
                "NumOT": 0,
                "WFGM": 30,
                "WFGA": 60,
                "WFGM3": 8,
                "WFGA3": 20,
                "WFTM": 12,
                "WFTA": 16,
                "WOR": 10,
                "WDR": 22,
                "WAst": 18,
                "WTO": 12,
                "WStl": 5,
                "WBlk": 4,
                "WPF": 15,
                "LFGM": 25,
                "LFGA": 55,
                "LFGM3": 6,
                "LFGA3": 18,
                "LFTM": 14,
                "LFTA": 20,
                "LOR": 8,
                "LDR": 20,
                "LAst": 14,
                "LTO": 15,
                "LStl": 6,
                "LBlk": 3,
                "LPF": 18,
            }
        )
    return pd.DataFrame(rows)


def test_effective_fg_and_possessions():
    results = _base_rows()
    features = build_team_season_features(results)
    team = features[features["TeamID"] == 1001].iloc[0]
    expected_efg = (30 * 6 + 0.5 * 8 * 6) / (60 * 6)
    assert abs(team["EffectiveFGPercent"] - expected_efg) < 1e-6
    total_points = 80 + 75 + 90 + 85 + 70 + 88
    total_fga = 60 * 6
    total_off_reb = 10 * 6
    total_to = 12 * 6
    total_fta = 16 * 6
    possessions = total_fga - total_off_reb + total_to + 0.44 * total_fta
    expected_off_rating = total_points / possessions
    assert abs(team["OffensiveRating"] - expected_off_rating) < 1e-6


def test_recent_form_windows():
    results = _base_rows()
    features = build_team_season_features(results)
    team = features[features["TeamID"] == 1001].iloc[0]
    assert team["Recent5WinPct"] == 1.0
    assert team["Recent10WinPct"] == team["WinPercentage"]


def test_adjusted_margin_differs_from_raw():
    results = _base_rows()
    features = build_team_season_features(results)
    team = features[features["TeamID"] == 1001].iloc[0]
    opponent = features[features["TeamID"] == 1002].iloc[0]
    assert team["AdjustedScoringMargin"] > opponent["AdjustedScoringMargin"]
    assert team["AdjustedScoringMargin"] != team["ScoringMargin"]
