import pandas as pd

from src.experiments import (
    summarize_comparison_results,
    _default_feature_sets_for_mode,
    _merge_supplemental_feature_set,
)
from src.feature_metadata import select_diff_columns


def test_select_diff_columns_respects_feature_set():
    diff_cols = [
        "Diff_GamesPlayed",
        "Diff_SeedNum",
        "Diff_EffectiveFGPercent",
        "Diff_Recent5WinPct",
    ]
    core_cols = select_diff_columns(diff_cols, "core")
    assert "Diff_GamesPlayed" in core_cols
    assert "Diff_EffectiveFGPercent" not in core_cols

    advanced_cols = select_diff_columns(diff_cols, "advanced")
    assert set(advanced_cols) == set(diff_cols)


def test_summarize_results_builds_conclusions():
    rows = [
        {
            "split_name": "main_validation",
            "train_end_season": 2014,
            "validation_start_season": 2015,
            "feature_set": "core",
            "model_name": "logistic_regression",
            "feature_count": 10,
            "train_rows": 100,
            "validation_rows": 50,
            "log_loss": 0.70,
            "brier_score": 0.20,
            "roc_auc": 0.60,
            "accuracy": 0.60,
        },
        {
            "split_name": "main_validation",
            "train_end_season": 2014,
            "validation_start_season": 2015,
            "feature_set": "advanced",
            "model_name": "neural_net",
            "feature_count": 20,
            "train_rows": 100,
            "validation_rows": 50,
            "log_loss": 0.60,
            "brier_score": 0.18,
            "roc_auc": 0.70,
            "accuracy": 0.65,
        },
        {
            "split_name": "backtest_train<= 2014_val>= 2015",
            "train_end_season": 2014,
            "validation_start_season": 2015,
            "feature_set": "core",
            "model_name": "logistic_regression",
            "feature_count": 10,
            "train_rows": 100,
            "validation_rows": 50,
            "log_loss": 0.68,
            "brier_score": 0.21,
            "roc_auc": 0.62,
            "accuracy": 0.61,
        },
        {
            "split_name": "backtest_train<= 2014_val>= 2015",
            "train_end_season": 2014,
            "validation_start_season": 2015,
            "feature_set": "advanced",
            "model_name": "logistic_regression",
            "feature_count": 20,
            "train_rows": 100,
            "validation_rows": 50,
            "log_loss": 0.64,
            "brier_score": 0.19,
            "roc_auc": 0.68,
            "accuracy": 0.66,
        },
        {
            "split_name": "backtest_train<= 2014_val>= 2015",
            "train_end_season": 2014,
            "validation_start_season": 2015,
            "feature_set": "advanced",
            "model_name": "neural_net",
            "feature_count": 20,
            "train_rows": 100,
            "validation_rows": 50,
            "log_loss": 0.63,
            "brier_score": 0.18,
            "roc_auc": 0.70,
            "accuracy": 0.67,
        },
    ]
    df = pd.DataFrame(rows)
    summary = summarize_comparison_results(df)

    assert summary["best_validation_configuration"]["model"] == "neural_net"
    assert "advanced_vs_core_backtests" in summary
    assert summary["advanced_vs_core_backtests"]["logistic_regression"]["log_loss_improvement"] > 0
    assert "conclusions" in summary and summary["conclusions"]


def test_default_feature_sets_include_supplemental_when_requested():
    base_sets = _default_feature_sets_for_mode("comparison")
    augmented = _merge_supplemental_feature_set(base_sets)
    if base_sets == augmented:
        # In case supplemental set is already present, ensure no duplicates.
        assert len(augmented) == len(base_sets)
    else:
        assert "core_plus_supplemental_ncaa" in augmented
