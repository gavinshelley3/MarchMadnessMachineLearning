import pandas as pd

from src.cbbpy_enrichment import build_team_features_from_boxscores


def _sample_box_df():
    rows = [
        {
            "game_id": "G1",
            "team": "Alpha College",
            "player": "Player A1",
            "pts": 10,
            "fgm": 4,
            "fga": 8,
            "2pm": 3,
            "2pa": 5,
            "3pm": 1,
            "3pa": 3,
            "ftm": 1,
            "fta": 2,
            "reb": 4,
            "oreb": 1,
            "dreb": 3,
            "ast": 2,
            "stl": 1,
            "blk": 0,
            "to": 1,
            "game_date": "2026-01-01",
        },
        {
            "game_id": "G1",
            "team": "Alpha College",
            "player": "Player A2",
            "pts": 8,
            "fgm": 3,
            "fga": 6,
            "2pm": 2,
            "2pa": 4,
            "3pm": 1,
            "3pa": 2,
            "ftm": 1,
            "fta": 1,
            "reb": 5,
            "oreb": 2,
            "dreb": 3,
            "ast": 1,
            "stl": 0,
            "blk": 1,
            "to": 2,
            "game_date": "2026-01-01",
        },
        {
            "game_id": "G1",
            "team": "Beta State",
            "player": "Player B1",
            "pts": 12,
            "fgm": 5,
            "fga": 10,
            "2pm": 4,
            "2pa": 6,
            "3pm": 1,
            "3pa": 4,
            "ftm": 1,
            "fta": 2,
            "reb": 6,
            "oreb": 2,
            "dreb": 4,
            "ast": 3,
            "stl": 1,
            "blk": 0,
            "to": 1,
            "game_date": "2026-01-01",
        },
        {
            "game_id": "G1",
            "team": "Beta State",
            "player": "Player B2",
            "pts": 6,
            "fgm": 2,
            "fga": 5,
            "2pm": 2,
            "2pa": 4,
            "3pm": 0,
            "3pa": 1,
            "ftm": 2,
            "fta": 2,
            "reb": 5,
            "oreb": 1,
            "dreb": 4,
            "ast": 2,
            "stl": 0,
            "blk": 0,
            "to": 2,
            "game_date": "2026-01-01",
        },
        {
            "game_id": "G2",
            "team": "Alpha College",
            "player": "Player A1",
            "pts": 15,
            "fgm": 6,
            "fga": 12,
            "2pm": 5,
            "2pa": 8,
            "3pm": 1,
            "3pa": 4,
            "ftm": 2,
            "fta": 2,
            "reb": 5,
            "oreb": 1,
            "dreb": 4,
            "ast": 3,
            "stl": 1,
            "blk": 1,
            "to": 2,
            "game_date": "2026-01-05",
        },
        {
            "game_id": "G2",
            "team": "Alpha College",
            "player": "Player A2",
            "pts": 5,
            "fgm": 2,
            "fga": 5,
            "2pm": 2,
            "2pa": 3,
            "3pm": 0,
            "3pa": 2,
            "ftm": 1,
            "fta": 1,
            "reb": 4,
            "oreb": 1,
            "dreb": 3,
            "ast": 2,
            "stl": 0,
            "blk": 0,
            "to": 1,
            "game_date": "2026-01-05",
        },
        {
            "game_id": "G2",
            "team": "Beta State",
            "player": "Player B1",
            "pts": 10,
            "fgm": 4,
            "fga": 9,
            "2pm": 3,
            "2pa": 5,
            "3pm": 1,
            "3pa": 4,
            "ftm": 1,
            "fta": 2,
            "reb": 6,
            "oreb": 2,
            "dreb": 4,
            "ast": 2,
            "stl": 1,
            "blk": 0,
            "to": 1,
            "game_date": "2026-01-05",
        },
        {
            "game_id": "G2",
            "team": "Beta State",
            "player": "Player B2",
            "pts": 15,
            "fgm": 5,
            "fga": 11,
            "2pm": 4,
            "2pa": 7,
            "3pm": 1,
            "3pa": 4,
            "ftm": 4,
            "fta": 5,
            "reb": 7,
            "oreb": 2,
            "dreb": 5,
            "ast": 3,
            "stl": 0,
            "blk": 1,
            "to": 2,
            "game_date": "2026-01-05",
        },
    ]
    return pd.DataFrame(rows)


def test_build_team_features_from_boxscores_maps_and_aggregates():
    box_df = _sample_box_df()
    teams_df = pd.DataFrame(
        {
            "TeamID": [1, 2],
            "TeamName": ["Alpha College", "Beta State"],
        }
    )
    features, mapping_diag = build_team_features_from_boxscores(box_df, 2026, teams_df)
    assert not features.empty
    assert set(features.columns).issuperset(
        {
            "Season",
            "TeamID",
            "CBBpy_GamesPlayed",
            "CBBpy_WinPct",
            "CBBpy_PointsForPerGame",
            "CBBpy_PointsAgainstPerGame",
            "CBBpy_Last5_WinPct",
            "CBBpy_Last5_ScoringMargin",
        }
    )
    alpha_row = features.loc[features["TeamID"] == 1].iloc[0]
    assert alpha_row["CBBpy_GamesPlayed"] == 2
    assert 0.0 <= alpha_row["CBBpy_WinPct"] <= 1.0
    assert not pd.isna(alpha_row["CBBpy_Last5_WinPct"])
    assert mapping_diag["teams_matched"] == 2


def test_build_team_features_reports_unmatched_team():
    box_df = _sample_box_df()
    teams_df = pd.DataFrame(
        {
            "TeamID": [1],
            "TeamName": ["Alpha College"],
        }
    )
    features, mapping_diag = build_team_features_from_boxscores(box_df, 2026, teams_df)
    assert mapping_diag["teams_matched"] == 1
    assert "Beta State" in mapping_diag["teams_unmatched"]
