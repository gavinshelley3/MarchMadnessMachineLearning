import pandas as pd

from src.experiments import summarize_ablation_results
from src.feature_metadata import default_ablation_feature_sets, select_diff_columns


def test_default_ablation_sets_present():
    feature_sets = default_ablation_feature_sets()
    assert feature_sets[0] == "core"
    assert "core_plus_efficiency" in feature_sets
    assert "core_plus_all_advanced" in feature_sets


def test_ablation_summary_detects_best_trimmed_set():
    rows = [
        # main validation rows
        {
            "split_name": "main_validation",
            "train_end_season": 2014,
            "validation_start_season": 2015,
            "feature_set": "core",
            "model_name": "logistic_regression",
            "feature_count": 10,
            "train_rows": 100,
            "validation_rows": 40,
            "log_loss": 0.70,
            "brier_score": 0.21,
            "roc_auc": 0.60,
            "accuracy": 0.60,
        },
        {
            "split_name": "main_validation",
            "train_end_season": 2014,
            "validation_start_season": 2015,
            "feature_set": "core_plus_efficiency",
            "model_name": "logistic_regression",
            "feature_count": 15,
            "train_rows": 100,
            "validation_rows": 40,
            "log_loss": 0.65,
            "brier_score": 0.20,
            "roc_auc": 0.65,
            "accuracy": 0.63,
        },
        {
            "split_name": "main_validation",
            "train_end_season": 2014,
            "validation_start_season": 2015,
            "feature_set": "core_plus_efficiency",
            "model_name": "neural_net",
            "feature_count": 15,
            "train_rows": 100,
            "validation_rows": 40,
            "log_loss": 0.62,
            "brier_score": 0.19,
            "roc_auc": 0.68,
            "accuracy": 0.66,
        },
        {
            "split_name": "main_validation",
            "train_end_season": 2014,
            "validation_start_season": 2015,
            "feature_set": "core",
            "model_name": "neural_net",
            "feature_count": 10,
            "train_rows": 100,
            "validation_rows": 40,
            "log_loss": 0.63,
            "brier_score": 0.19,
            "roc_auc": 0.67,
            "accuracy": 0.65,
        },
        # backtests
        {
            "split_name": "backtest_train<= 2014_val>= 2015",
            "train_end_season": 2014,
            "validation_start_season": 2015,
            "feature_set": "core",
            "model_name": "logistic_regression",
            "feature_count": 10,
            "train_rows": 90,
            "validation_rows": 40,
            "log_loss": 0.69,
            "brier_score": 0.22,
            "roc_auc": 0.61,
            "accuracy": 0.59,
        },
        {
            "split_name": "backtest_train<= 2014_val>= 2015",
            "train_end_season": 2014,
            "validation_start_season": 2015,
            "feature_set": "core_plus_efficiency",
            "model_name": "logistic_regression",
            "feature_count": 15,
            "train_rows": 90,
            "validation_rows": 40,
            "log_loss": 0.66,
            "brier_score": 0.20,
            "roc_auc": 0.66,
            "accuracy": 0.64,
        },
    ]
    df = pd.DataFrame(rows)
    summary = summarize_ablation_results(df)

    logistic_best = summary["main_validation"]["logistic_regression"]["feature_set"]
    assert logistic_best == "core_plus_efficiency"
    improvements = summary["improvements_vs_core"]["logistic_regression"]["main_validation"]
    assert abs(improvements["core_plus_efficiency"] - 0.05) < 1e-6
    assert summary["recommendations"]["logistic_regression"]["suggested_trimmed_set"] == "core_plus_efficiency"
