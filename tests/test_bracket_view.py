import json
from pathlib import Path

from src.bracket_view import BracketViewRenderer


def _sample_game(slot: str, region: str, team1: str, team2: str, prob1: float, prob2: float, winner: str, seed1: int, seed2: int):
    return {
        "round": "R64",
        "slot": slot,
        "region": region,
        "team1": {"team_key": team1, "display_name": team1, "seed": seed1},
        "team2": {"team_key": team2, "display_name": team2, "seed": seed2},
        "prob_team1": prob1,
        "prob_team2": prob2,
        "winner": {"team_key": winner, "display_name": winner},
    }


def _write_bracket(path: Path) -> Path:
    data = {
        "metadata": {
            "season": 2026,
            "bracket_type": "projected",
            "model_name": "logistic_regression",
            "feature_set": "core_plus_opponent_adjustment",
            "regions": {"East": 1},
        },
        "rounds": {
            "R64": [
                _sample_game("E1", "East", "Duke", "Siena", 0.9, 0.1, "Duke", 1, 16),
                _sample_game("E8", "East", "Georgia", "TCU", 0.6, 0.4, "Georgia", 8, 9),
            ],
            "R32": [
                {
                    "round": "R32",
                    "slot": "East-R32-1",
                    "region": "East",
                    "team1": {"team_key": "Duke", "display_name": "Duke", "seed": 1},
                    "team2": {"team_key": "Georgia", "display_name": "Georgia", "seed": 8},
                    "prob_team1": 0.7,
                    "prob_team2": 0.3,
                    "winner": {"team_key": "Duke", "display_name": "Duke"},
                }
            ],
            "FinalFour": [
                {
                    "round": "FinalFour",
                    "slot": "Semifinal 1",
                    "region": "National",
                    "team1": {"team_key": "Duke", "display_name": "Duke", "seed": 1},
                    "team2": {"team_key": "Arizona", "display_name": "Arizona", "seed": 1},
                    "prob_team1": 0.55,
                    "prob_team2": 0.45,
                    "winner": {"team_key": "Duke", "display_name": "Duke"},
                }
            ],
            "Championship": [
                {
                    "round": "Championship",
                    "slot": "Title",
                    "region": "National",
                    "team1": {"team_key": "Duke", "display_name": "Duke", "seed": 1},
                    "team2": {"team_key": "Michigan", "display_name": "Michigan", "seed": 1},
                    "prob_team1": 0.52,
                    "prob_team2": 0.48,
                    "winner": {"team_key": "Duke", "display_name": "Duke"},
                }
            ],
        },
        "champion": {"team_key": "Duke", "display_name": "Duke"},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_bracket_view_generates_html(tmp_path: Path) -> None:
    bracket_file = _write_bracket(tmp_path / "bracket.json")
    output_path = tmp_path / "view.html"
    renderer = BracketViewRenderer(bracket_results_path=bracket_file, output_path=output_path)
    result_path = renderer.render()
    assert result_path.exists()
    html_text = result_path.read_text()
    assert "Duke" in html_text
    assert "90.0% win" in html_text  # winner probability label
    assert 'class="champion-card"' in html_text
