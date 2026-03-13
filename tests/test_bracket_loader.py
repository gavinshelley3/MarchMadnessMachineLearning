import json
from pathlib import Path

import pytest

from src.bracket_loader import BracketDefinition, load_bracket_definition


def _write_bracket(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "bracket.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_bracket_with_play_in_options(tmp_path: Path) -> None:
    bracket_payload = {
        "season": 2026,
        "bracket_type": "projected",
        "regions": [
            {
                "name": "Alpha",
                "round_of_64": [
                    {
                        "slot": "A1",
                        "round": "R64",
                        "team1": {"seed": 1, "team_key": "TeamA", "display_name": "Team A"},
                        "team2": {
                            "seed": 16,
                            "display_name": "Team B / Team C",
                            "options": [
                                {"team_key": "TeamB", "display_name": "Team B"},
                                {"team_key": "TeamC", "display_name": "Team C"},
                            ],
                        },
                    },
                    {
                        "slot": "A2",
                        "round": "R64",
                        "team1": {"seed": 8, "team_key": "TeamD"},
                        "team2": {"seed": 9, "team_key": "TeamE"},
                    },
                ],
            },
            {
                "name": "Beta",
                "round_of_64": [
                    {
                        "slot": "B1",
                        "round": "R64",
                        "team1": {"seed": 1, "team_key": "TeamF"},
                        "team2": {"seed": 16, "team_key": "TeamG"},
                    }
                ],
            },
        ],
        "final_four_pairings": [
            {"regions": ["Alpha", "Beta"], "name": "Semifinal A"},
            {"regions": ["Alpha", "Beta"], "name": "Semifinal B"},
        ],
    }
    path = _write_bracket(tmp_path, bracket_payload)
    definition = load_bracket_definition(path)

    assert isinstance(definition, BracketDefinition)
    assert definition.season == 2026
    slots = list(definition.iter_round_of_64())
    assert len(slots) == 3  # two Alpha, one Beta
    play_in_slot = next(slot for slot in slots if slot.slot == "A1")
    assert play_in_slot.team2.team_key == "TeamB"
    assert len(play_in_slot.team2.options) == 2
    assert len(definition.final_four_pairings) == 2


def test_invalid_bracket_missing_final_four(tmp_path: Path) -> None:
    bracket_payload = {
        "season": 2026,
        "regions": [
            {
                "name": "Alpha",
                "round_of_64": [
                    {
                        "slot": "A1",
                        "team1": {"seed": 1, "team_key": "TeamA"},
                        "team2": {"seed": 16, "team_key": "TeamB"},
                    }
                ],
            }
        ],
        "final_four_pairings": [],
    }
    path = _write_bracket(tmp_path, bracket_payload)
    with pytest.raises(ValueError):
        load_bracket_definition(path)
