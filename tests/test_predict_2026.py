import json
from pathlib import Path

import pandas as pd

from src import predict_2026 as p


def test_resolve_field_team_ids_handles_play_in(monkeypatch, tmp_path):
    bracket_payload = {
        "season": 2026,
        "bracket_type": "official",
        "regions": [
            {
                "name": "Alpha",
                "round_of_64": [
                    {
                        "slot": "A1",
                        "round": "R64",
                        "team1": {"seed": 1, "team_key": "Alpha One", "display_name": "Alpha One"},
                        "team2": {
                            "seed": 16,
                            "team_key": "Alpha PlayIn",
                            "display_name": "TBD",
                            "options": [
                                {"team_key": "Beta", "display_name": "Beta"},
                                {"team_key": "Gamma", "display_name": "Gamma"},
                            ],
                        },
                    }
                ],
            }
        ],
        "final_four_pairings": [{"regions": ["Alpha", "Alpha"], "name": "FF1"}],
    }
    bracket_path = tmp_path / "bracket.json"
    bracket_path.write_text(json.dumps(bracket_payload), encoding="utf-8")

    teams_df = pd.DataFrame(
        [
            {"TeamID": 1, "TeamName": "Alpha One"},
            {"TeamID": 2, "TeamName": "Beta"},
            {"TeamID": 3, "TeamName": "Gamma"},
        ]
    )
    spellings_df = pd.DataFrame([{"TeamID": 3, "TeamNameSpelling": "Gamma"}])

    monkeypatch.setattr(p.data_loading, "load_teams", lambda data_dir: teams_df)
    monkeypatch.setattr(p.data_loading, "load_team_spellings", lambda data_dir: spellings_df)

    ids = p._resolve_field_team_ids(bracket_path, None)
    assert ids == {1, 2, 3}


def test_restrict_matchups_to_field_filters_rows():
    matchups = pd.DataFrame(
        [
            {"Team1ID": 1, "Team2ID": 2},
            {"Team1ID": 1, "Team2ID": 3},
            {"Team1ID": 4, "Team2ID": 5},
        ]
    )
    filtered = p._restrict_matchups_to_field(matchups, {1, 2, 3})
    assert len(filtered) == 2
    assert all(filtered["Team2ID"].isin({2, 3}))
