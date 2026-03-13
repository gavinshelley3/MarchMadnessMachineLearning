import pandas as pd

from src.predict_2026 import (
    _fill_missing_numeric,
    build_inference_dataset,
    parse_sample_submission,
)


def test_parse_sample_submission_deduplicates_and_sorts():
    df = pd.DataFrame(
        {
            "ID": [
                "2026_1001_1003",
                "2026_1001_1002",
                "2025_999_1000",
                "2026_1001_1003",
            ]
        }
    )
    matchups = parse_sample_submission(df, 2026)
    assert list(matchups["Team1ID"]) == [1001, 1001]
    assert list(matchups["Team2ID"]) == [1002, 1003]
    assert all(matchups["Season"] == 2026)


def test_build_inference_dataset_matches_feature_differences():
    context = pd.DataFrame(
        {
            "Season": [2026, 2026],
            "TeamID": [1, 2],
            "Seed": ["W01", "W16"],
            "SeedNum": [1, 16],
            "ScoringMargin": [15.0, -3.0],
        }
    )
    matchups = pd.DataFrame({"Season": [2026], "Team1ID": [1], "Team2ID": [2]})
    dataset = build_inference_dataset(context, matchups)
    assert dataset["Team1_SeedNum"].iloc[0] == 1
    assert dataset["Team2_SeedNum"].iloc[0] == 16
    assert dataset["Diff_ScoringMargin"].iloc[0] == 18.0


def test_fill_missing_numeric_reports_columns():
    df = pd.DataFrame({"a": [1.0, None], "b": [2.0, 3.0]})
    filled, stats = _fill_missing_numeric(df, ["a", "b"])
    assert filled["a"].iloc[1] == 1.0  # mean of column
    assert "a" in stats and "b" not in stats
