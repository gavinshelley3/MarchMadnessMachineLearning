import json
from pathlib import Path

from src.compare_brackets import compare_brackets


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compare_brackets_detects_changes(tmp_path):
    baseline_bracket = {
        "champion": {"display_name": "Alpha"},
        "rounds": {
            "R64": [
                {
                    "round": "R64",
                    "slot": "E1",
                    "winner": {"display_name": "Alpha"},
                    "team1": {"display_name": "Alpha"},
                    "team2": {"display_name": "Delta"},
                }
            ],
            "Elite8": [
                {"winner": {"display_name": "Alpha"}},
                {"winner": {"display_name": "Gamma"}},
            ],
            "Championship": [
                {
                    "round": "Championship",
                    "slot": "Title",
                    "team1": {"display_name": "Alpha"},
                    "team2": {"display_name": "Gamma"},
                    "winner": {"display_name": "Alpha"},
                }
            ],
        },
    }
    enriched_bracket = {
        "champion": {"display_name": "Gamma"},
        "rounds": {
            "R64": [
                {
                    "round": "R64",
                    "slot": "E1",
                    "winner": {"display_name": "Delta"},
                    "team1": {"display_name": "Alpha"},
                    "team2": {"display_name": "Delta"},
                }
            ],
            "Elite8": [
                {"winner": {"display_name": "Gamma"}},
                {"winner": {"display_name": "Beta"}},
            ],
            "Championship": [
                {
                    "round": "Championship",
                    "slot": "Title",
                    "team1": {"display_name": "Gamma"},
                    "team2": {"display_name": "Beta"},
                    "winner": {"display_name": "Gamma"},
                }
            ],
        },
    }
    baseline_summary = {
        "top_championship_contenders": [
            {"team": "Alpha", "champion_probability": 0.6},
            {"team": "Gamma", "champion_probability": 0.2},
        ],
        "most_likely_champion": {"team": "Alpha"},
    }
    enriched_summary = {
        "top_championship_contenders": [
            {"team": "Alpha", "champion_probability": 0.3},
            {"team": "Gamma", "champion_probability": 0.5},
        ],
        "most_likely_champion": {"team": "Gamma"},
    }
    report = compare_brackets(
        _write(tmp_path / "baseline_bracket.json", baseline_bracket),
        _write(tmp_path / "enriched_bracket.json", enriched_bracket),
        _write(tmp_path / "baseline_summary.json", baseline_summary),
        _write(tmp_path / "enriched_summary.json", enriched_summary),
    )
    assert report["champion"]["changed"] is True
    assert report["championship_matchup"]["changed"] is True
    assert report["final_four"]["changed"] is True
    assert report["simulation"]["most_likely_champion_changed"] is True
    assert report["simulation"]["top_champion_probability_shifts"]
    assert any(diff["slot"] == "E1" for diff in report["deterministic_differences"])
