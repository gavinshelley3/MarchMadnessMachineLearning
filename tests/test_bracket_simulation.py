import json
from pathlib import Path

import pandas as pd

from src.bracket_loader import load_bracket_definition
from src.bracket_simulation import BracketSimulator
from tests.test_bracket_generation import build_prediction_csv, build_test_bracket


def build_deterministic_predictions(tmp_path: Path, teams: list[str]) -> Path:
    rows = []
    for i, team_a in enumerate(teams):
        for j, team_b in enumerate(teams):
            if team_a == team_b or i > j:
                continue
            prob = 1.0 if i < j else 0.0
            rows.append(
                {
                    "Season": 2026,
                    "Team1Name": team_a,
                    "Team2Name": team_b,
                    "PredTeam1WinProb": prob,
                    "PredTeam2WinProb": 1 - prob,
                    "Team1ID": i + 1,
                    "Team2ID": j + 1,
                    "model_name": "logit",
                    "feature_set": "core",
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "deterministic_predictions.csv"
    df.to_csv(path, index=False)
    return path


def test_simulation_outputs(tmp_path: Path) -> None:
    bracket_path = build_test_bracket(tmp_path)
    bracket = load_bracket_definition(bracket_path)
    teams = []
    for region in bracket.regions:
        for matchup in region.round_of_64:
            teams.extend([matchup.team1.team_key, matchup.team2.team_key])
    predictions_path = build_prediction_csv(tmp_path, teams)
    simulator = BracketSimulator(
        bracket_def=bracket,
        predictions_path=predictions_path,
        n_sims=200,
        seed=7,
        output_dir=tmp_path / "out",
    )
    outputs = simulator.run()

    assert outputs.probabilities.exists()
    assert outputs.summary.exists()
    assert outputs.confidence.exists()

    summary = json.loads(outputs.summary.read_text())
    assert summary["simulations"]["n_sims"] == 200
    assert summary["simulations"]["seed"] == 7
    assert "top_championship_contenders" in summary


def test_simulation_handles_reversed_matchups(tmp_path: Path) -> None:
    bracket_path = build_test_bracket(tmp_path)
    bracket = load_bracket_definition(bracket_path)
    teams = []
    for region in bracket.regions:
        for matchup in region.round_of_64:
            teams.extend([matchup.team1.team_key, matchup.team2.team_key])
    predictions_path = build_deterministic_predictions(tmp_path, sorted(set(teams)))
    simulator = BracketSimulator(
        bracket_def=bracket,
        predictions_path=predictions_path,
        n_sims=50,
        seed=1,
        output_dir=tmp_path / "deterministic",
    )
    outputs = simulator.run()
    df = pd.read_csv(outputs.probabilities)
    # With deterministic probabilities, the top seed in each region should have champion probability 1.0
    assert df["champion_probability"].max() == 1.0
