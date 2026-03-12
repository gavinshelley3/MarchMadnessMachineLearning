import pandas as pd

from src.data_pipeline import build_dataset_from_frames


def _regular_season_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": [2024, 2024],
            "DayNum": [1, 2],
            "WTeamID": [1001, 1003],
            "WScore": [80, 75],
            "LTeamID": [1002, 1004],
            "LScore": [70, 65],
            "WLoc": ["H", "H"],
            "NumOT": [0, 0],
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


def _tourney_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": [2024],
            "DayNum": [136],
            "WTeamID": [1001],
            "WScore": [72],
            "LTeamID": [1002],
            "LScore": [66],
        }
    )


def _seeds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": [2024, 2024, 2024, 2024],
            "Seed": ["W01", "W16", "X05", "X12"],
            "TeamID": [1001, 1002, 1003, 1004],
        }
    )


def _massey() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": [2024, 2024, 2024, 2024],
            "RankingDayNum": [130, 130, 130, 130],
            "SystemName": ["TEST"] * 4,
            "TeamID": [1001, 1002, 1003, 1004],
            "OrdinalRank": [5, 25, 15, 40],
        }
    )


def test_build_dataset_from_frames_produces_symmetric_rows() -> None:
    dataset, diagnostics = build_dataset_from_frames(
        _regular_season_rows(),
        _tourney_games(),
        _seeds(),
        massey_ordinals=_massey(),
        massey_system="TEST",
    )
    assert len(dataset) == 2
    assert set(dataset["Label"].unique()) == {0, 1}
    assert diagnostics.rows_before_filter == diagnostics.rows_after_filter == 2
    assert diagnostics.dropped_rows == 0
    assert diagnostics.feature_count > 0
    assert diagnostics.class_balance["0"] == diagnostics.class_balance["1"] == 0.5
