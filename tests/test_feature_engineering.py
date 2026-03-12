import pandas as pd

from src.feature_engineering import build_team_season_features, parse_numeric_seed


def _sample_game_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": [2024, 2024],
            "DayNum": [1, 2],
            "WTeamID": [1001, 1002],
            "LTeamID": [1002, 1003],
            "WScore": [80, 75],
            "LScore": [70, 65],
            "WFGM": [30, 28],
            "WFGA": [60, 55],
            "WFGM3": [8, 7],
            "WFGA3": [20, 18],
            "WFTM": [12, 12],
            "WFTA": [15, 14],
            "WOR": [10, 9],
            "WDR": [20, 21],
            "WAst": [18, 17],
            "WTO": [12, 11],
            "WStl": [7, 6],
            "WBlk": [5, 4],
            "WPF": [15, 16],
            "LFGM": [25, 24],
            "LFGA": [58, 50],
            "LFGM3": [6, 5],
            "LFGA3": [18, 17],
            "LFTM": [12, 11],
            "LFTA": [14, 12],
            "LOR": [8, 7],
            "LDR": [18, 19],
            "LAst": [15, 14],
            "LTO": [13, 12],
            "LStl": [5, 4],
            "LBlk": [3, 2],
            "LPF": [17, 18],
        }
    )


def test_parse_numeric_seed_handles_letters_and_suffixes() -> None:
    assert parse_numeric_seed("W01") == 1
    assert parse_numeric_seed("X16b") == 16
    assert parse_numeric_seed("07") == 7
    assert parse_numeric_seed(None) is None


def test_build_team_season_features_basic_aggregations() -> None:
    df = _sample_game_rows()
    features = build_team_season_features(df)
    team_1001 = features[features["TeamID"] == 1001].iloc[0]
    assert team_1001["GamesPlayed"] == 1
    assert team_1001["WinPercentage"] == 1.0
    assert team_1001["PointsPerGame"] == 80
    assert round(team_1001["AssistToTurnoverRatio"], 3) == round(18 / 12, 3)
