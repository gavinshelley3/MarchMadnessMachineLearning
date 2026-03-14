import pandas as pd
import pytest

from src.predict_2026 import apply_cbbpy_overrides, build_inference_matchups


def test_build_inference_matchups_creates_ordered_pairs():
    context = pd.DataFrame(
        {
            "Season": [2026, 2026],
            "TeamID": [1, 2],
            "GamesPlayed": [30, 28],
            "WinPercentage": [0.8, 0.6],
            "Seed": ["W01", "W02"],
            "SeedNum": [1, 2],
        }
    )
    teams_df = pd.DataFrame({"TeamID": [1, 2], "TeamName": ["Alpha", "Beta"]})
    matchups = build_inference_matchups(context, teams_df, 2026)
    assert set(matchups["MatchupID"]) == {"2026_1_2"}
    row = matchups.loc[matchups["MatchupID"] == "2026_1_2"].iloc[0]
    assert row["Diff_WinPercentage"] == pytest.approx(0.2)
    assert row["Team1Name"] == "Alpha"
    assert row["Team2Name"] == "Beta"


def test_apply_cbbpy_overrides_updates_core_metrics():
    context = pd.DataFrame(
        {
            "Season": [2026],
            "TeamID": [1],
            "GamesPlayed": [28],
            "WinPercentage": [0.7],
            "PointsPerGame": [75.0],
            "PointsAllowedPerGame": [68.0],
        }
    )
    cbbpy_frame = pd.DataFrame(
        {
            "Season": [2026],
            "TeamID": [1],
            "CBBpy_GamesPlayed": [32],
            "CBBpy_WinPct": [0.875],
            "CBBpy_PointsForPerGame": [80.0],
            "CBBpy_PointsAgainstPerGame": [65.0],
        }
    )
    updated = apply_cbbpy_overrides(context, cbbpy_frame, 2026)
    assert updated.loc[0, "GamesPlayed"] == 32
    assert updated.loc[0, "WinPercentage"] == 0.875
    assert updated.loc[0, "PointsPerGame"] == 80.0
