import json
from pathlib import Path

import pandas as pd

from src.results_report import ResultsReportGenerator


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def setup_sample_artifacts(base_path: Path) -> tuple[Path, Path, Path]:
    reports_dir = base_path / "reports"
    brackets_dir = base_path / "brackets"
    reports_dir.mkdir()
    brackets_dir.mkdir()

    _write_json(
        reports_dir / "final_model_summary.json",
        {
            "recommended_production_configuration": {
                "model_name": "logistic_regression",
                "feature_set": "core_plus_opponent_adjustment",
                "log_loss": 0.58,
                "accuracy": 0.71,
            },
            "best_model_main_split": {"log_loss": 0.57, "accuracy": 0.72},
            "best_feature_set_backtest_average": {"feature_set": "core_plus_opponent_adjustment", "avg_log_loss": 0.59},
            "feature_sets": {"core_plus_opponent_adjustment": "Core + opponent adjustment"},
        },
    )
    _write_csv(
        reports_dir / "final_model_comparison.csv",
        [
            {
                "split_name": "main_validation",
                "train_end_season": 2014,
                "validation_start_season": 2015,
                "model_name": "logistic_regression",
                "feature_set": "core_plus_opponent_adjustment",
                "log_loss": 0.57,
                "accuracy": 0.72,
                "brier_score": 0.19,
                "roc_auc": 0.77,
            },
            {
                "split_name": "backtest_train<= 2014_val>= 2015",
                "train_end_season": 2014,
                "validation_start_season": 2015,
                "model_name": "logistic_regression",
                "feature_set": "core_plus_opponent_adjustment",
                "log_loss": 0.59,
                "accuracy": 0.70,
                "brier_score": 0.2,
                "roc_auc": 0.76,
            },
        ],
    )
    _write_json(reports_dir / "backtest_summary.json", {"note": "unused placeholder"})

    _write_json(
        brackets_dir / "2026_test_simulation_summary.json",
        {
            "metadata": {"season": 2026, "bracket_type": "projected"},
            "simulations": {"n_sims": 100, "seed": 7},
            "most_likely_champion": {"team": "Duke", "champion_probability": 0.18},
            "top_championship_contenders": [
                {"team": "Duke", "champion_probability": 0.18},
                {"team": "Houston", "champion_probability": 0.15},
            ],
            "most_common_championship_matchup": {"teams": ["Duke", "Houston"], "probability": 0.11},
            "most_common_final_four": {"teams": ["Duke", "Houston", "Purdue", "UConn"], "probability": 0.07},
        },
    )
    _write_csv(
        brackets_dir / "2026_test_simulation_team_probabilities.csv",
        [
            {
                "team": "Duke",
                "team_key": "Duke",
                "region": "East",
                "seed": 1,
                "final_four_probability": 0.32,
                "champion_probability": 0.18,
            },
            {
                "team": "Houston",
                "team_key": "Houston",
                "region": "South",
                "seed": 1,
                "final_four_probability": 0.31,
                "champion_probability": 0.15,
            },
        ],
    )
    _write_csv(
        brackets_dir / "2026_test_pick_confidence.csv",
        [
            {
                "round": "R64",
                "region": "East",
                "slot": "E1",
                "team1": "Duke",
                "team2": "Siena",
                "deterministic_winner": "Duke",
                "winner_probability": 0.88,
                "confidence_tier": "high",
            },
            {
                "round": "R64",
                "region": "West",
                "slot": "W8",
                "team1": "UCLA",
                "team2": "Texas A&M",
                "deterministic_winner": "UCLA",
                "winner_probability": 0.55,
                "confidence_tier": "low",
            },
        ],
    )
    _write_csv(
        brackets_dir / "2026_test_upset_risk_report.csv",
        [
            {
                "team": "Texas A&M",
                "seed": 10,
                "region": "West",
                "focus_round": "Round of 32",
                "probability": 0.42,
                "note": "Could clip UCLA early",
            }
        ],
    )
    return reports_dir, brackets_dir, base_path / "final"


def test_results_report_creates_outputs(tmp_path: Path) -> None:
    reports_dir, brackets_dir, output_dir = setup_sample_artifacts(tmp_path)
    generator = ResultsReportGenerator(
        reports_dir=reports_dir,
        brackets_dir=brackets_dir,
        output_dir=output_dir,
    )
    files = generator.run()

    assert (output_dir / "project_summary.json").exists()
    assert (output_dir / "project_summary.md").exists()
    assert (output_dir / "project_summary.html").exists()
    assert (output_dir / "charts/model_comparison_log_loss.png").exists()
    assert "summary_json" in files


def test_results_report_handles_missing_optional_inputs(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports_missing"
    brackets_dir = tmp_path / "brackets_missing"
    reports_dir.mkdir()
    brackets_dir.mkdir()
    _write_json(
        reports_dir / "final_model_summary.json",
        {"recommended_production_configuration": {"model_name": "logistic_regression", "feature_set": "core"}},
    )
    generator = ResultsReportGenerator(
        reports_dir=reports_dir,
        brackets_dir=brackets_dir,
        output_dir=tmp_path / "final_missing",
    )
    generator.run()
    summary = json.loads((tmp_path / "final_missing" / "project_summary.json").read_text())
    assert "missing_inputs" in summary and summary["missing_inputs"]
