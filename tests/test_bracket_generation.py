import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.bracket_generation import BracketGenerator
from src.bracket_loader import load_bracket_definition


def build_test_bracket(tmp_path: Path) -> Path:
    regions = {
        "East": ["EastA", "EastB", "EastC", "EastD"],
        "West": ["WestA", "WestB", "WestC", "WestD"],
        "South": ["SouthA", "SouthB", "SouthC", "SouthD"],
        "Midwest": ["MidA", "MidB", "MidC", "MidD"],
    }
    payload = {
        "season": 2026,
        "bracket_type": "test",
        "regions": [],
        "final_four_pairings": [
            {"regions": ["East", "West"], "name": "FF1"},
            {"regions": ["South", "Midwest"], "name": "FF2"},
        ],
    }
    for region_name, teams in regions.items():
        payload["regions"].append(
            {
                "name": region_name,
                "round_of_64": [
                    {
                        "slot": f"{region_name}-1",
                        "round": "R64",
                        "team1": {"seed": 1, "team_key": teams[0], "display_name": teams[0]},
                        "team2": {"seed": 16, "team_key": teams[1], "display_name": teams[1]},
                    },
                    {
                        "slot": f"{region_name}-2",
                        "round": "R64",
                        "team1": {"seed": 8, "team_key": teams[2], "display_name": teams[2]},
                        "team2": {"seed": 9, "team_key": teams[3], "display_name": teams[3]},
                    },
                ],
            }
        )
    bracket_path = tmp_path / "bracket.json"
    bracket_path.write_text(json.dumps(payload), encoding="utf-8")
    return bracket_path


def build_prediction_csv(tmp_path: Path, teams: list[str]) -> Path:
    ratings = {team: idx for idx, team in enumerate(reversed(teams), start=1)}
    rows = []
    for team_a, team_b in combinations(sorted(teams), 2):
        rating_diff = ratings[team_a] - ratings[team_b]
        prob = max(0.05, min(0.95, 0.5 + 0.02 * rating_diff))
        rows.append(
            {
                "Season": 2026,
                "Team1Name": team_a,
                "Team2Name": team_b,
                "PredTeam1WinProb": prob,
                "PredTeam2WinProb": 1 - prob,
                "Team1ID": ratings[team_a],
                "Team2ID": ratings[team_b],
                "model_name": "logistic_regression",
                "feature_set": "core_test",
            }
        )
    df = pd.DataFrame(rows)
    path = tmp_path / "predictions.csv"
    df.to_csv(path, index=False)
    return path


def test_bracket_generation_outputs(tmp_path: Path) -> None:
    bracket_path = build_test_bracket(tmp_path)
    bracket = load_bracket_definition(bracket_path)
    all_teams = []
    for region in bracket.regions:
        for matchup in region.round_of_64:
            all_teams.append(matchup.team1.team_key)
            all_teams.append(matchup.team2.team_key)
    predictions_path = build_prediction_csv(tmp_path, all_teams)
    output_dir = tmp_path / "brackets"
    generator = BracketGenerator(predictions_path=predictions_path, output_dir=output_dir)
    outputs = generator.generate(bracket, season=2026)

    assert outputs["json"].exists()
    assert outputs["csv"].exists()
    assert outputs["summary"].exists()

    summary_text = outputs["summary"].read_text()
    assert "Champion" in summary_text

    json_payload = json.loads(outputs["json"].read_text())
    assert json_payload["champion"]["team_key"] in all_teams
    assert json_payload["metadata"]["model_name"] == "logistic_regression"
