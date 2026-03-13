from pathlib import Path

import pandas as pd

from src.supplemental_ncaa import build_supplemental_features


def _write_csv(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_supplemental_features_produces_expected_columns(tmp_path: Path) -> None:
    supp_dir = tmp_path / "ncaa"
    supp_dir.mkdir()
    _write_csv(
        supp_dir / "mbb_historical_teams_seasons.csv",
        [
            {"season": 2025, "team_id": 1001, "team_name": "Duke", "games": 30, "possessions": 2100, "adj_em": 25.0, "sos": 8.5},
            {"season": 2025, "team_id": 1002, "team_name": "Michigan", "games": 30, "possessions": 2050, "adj_em": 15.0, "sos": 5.0},
        ],
    )
    _write_csv(
        supp_dir / "mbb_historical_teams_games.csv",
        [
            {"season": 2025, "team_id": 1001, "team_name": "Duke", "team_score": 80, "opponent_score": 75, "location": "N"},
            {"season": 2025, "team_id": 1001, "team_name": "Duke", "team_score": 70, "opponent_score": 72, "location": "A"},
            {"season": 2025, "team_id": 1002, "team_name": "Michigan", "team_score": 68, "opponent_score": 65, "location": "H"},
        ],
    )
    _write_csv(
        supp_dir / "mbb_teams.csv",
        [
            {"team_id": 1001, "team_name": "Duke"},
            {"team_id": 1002, "team_name": "Michigan"},
        ],
    )
    teams_df = pd.DataFrame(
        {"TeamID": [1001, 1002], "TeamName": ["Duke", "Michigan"]}
    )
    features, diagnostics = build_supplemental_features(supp_dir, teams_df)
    assert not features.empty
    assert set(features.columns) >= {"Season", "TeamID", "SuppTempo", "SuppAdjNetRating"}
    diag_dict = diagnostics.to_dict()
    assert diag_dict["tables_found"]["team_seasons"]
    assert diag_dict["tables_found"]["team_games"]
    assert diag_dict["feature_columns"]


def test_missing_tables_returns_empty_frame(tmp_path: Path) -> None:
    teams_df = pd.DataFrame({"TeamID": [1], "TeamName": ["Example"]})
    features, diagnostics = build_supplemental_features(tmp_path, teams_df)
    assert features.empty
    assert "required team season data missing" in " ".join(diagnostics.notes).lower()
