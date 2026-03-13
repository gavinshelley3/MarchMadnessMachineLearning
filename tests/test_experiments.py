import pandas as pd

from src.experiments import (
    summarize_comparison_results,
    _default_feature_sets_for_mode,
    _ensure_feature_set,
    SUPPLEMENTAL_FEATURE_SET_NAME,
    CBBPY_FEATURE_SET_NAME,
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


def test_optional_feature_set_helper_adds_sets_once():
    base_sets = _default_feature_sets_for_mode("comparison")
    augmented = _ensure_feature_set(base_sets, SUPPLEMENTAL_FEATURE_SET_NAME)
    if SUPPLEMENTAL_FEATURE_SET_NAME in base_sets:
        assert augmented == base_sets
    else:
        assert augmented[-1] == SUPPLEMENTAL_FEATURE_SET_NAME

    augmented_again = _ensure_feature_set(augmented, CBBPY_FEATURE_SET_NAME)
    if CBBPY_FEATURE_SET_NAME in augmented:
        assert augmented_again == augmented
    else:
        assert augmented_again[-1] == CBBPY_FEATURE_SET_NAME
