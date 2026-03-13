from pathlib import Path

import pandas as pd

from src.supplemental_ncaa import build_supplemental_features


def _write_csv(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_supplemental_features_from_cbb(tmp_path: Path) -> None:
    supp_dir = tmp_path / "supp"
    supp_dir.mkdir()
    _write_csv(
        supp_dir / "cbb.csv",
        [
            {
                "TEAM": "Duke",
                "ADJOE": 120.0,
                "ADJDE": 95.0,
                "ADJ_T": 68.5,
                "BARTHAG": 0.98,
                "WAB": 9.5,
                "EFG_O": 55.0,
                "EFG_D": 47.0,
                "YEAR": 2024,
            },
            {
                "TEAM": "Michigan",
                "ADJOE": 114.0,
                "ADJDE": 100.0,
                "ADJ_T": 64.0,
                "BARTHAG": 0.92,
                "WAB": 5.5,
                "EFG_O": 51.0,
                "EFG_D": 49.0,
                "YEAR": 2024,
            },
        ],
    )
    teams_df = pd.DataFrame(
        {
            "TeamID": [1001, 1002],
            "TeamName": ["Duke", "Michigan"],
        }
    )
    features, diagnostics = build_supplemental_features(supp_dir, teams_df)
    assert not features.empty
    assert set(features.columns) == {
        "Season",
        "TeamID",
        "SuppAdjTempo",
        "SuppAdjEfficiencyMargin",
        "SuppBarthag",
        "SuppWab",
        "SuppEffectiveFgDiff",
    }
    assert diagnostics.feature_columns
    assert diagnostics.season_coverage["min_season"] == 2024


def test_skips_womens_files(tmp_path: Path) -> None:
    supp_dir = tmp_path / "supp"
    supp_dir.mkdir()
    _write_csv(
        supp_dir / "cbb_wbb_sample.csv",
        [
            {"TEAM": "Example", "YEAR": 2024},
        ],
    )
    teams_df = pd.DataFrame({"TeamID": [1], "TeamName": ["Example"]})
    features, diagnostics = build_supplemental_features(supp_dir, teams_df)
    assert features.empty
    assert diagnostics.tables_used == []
    assert diagnostics.tables_skipped == ["cbb_wbb_sample.csv"]
