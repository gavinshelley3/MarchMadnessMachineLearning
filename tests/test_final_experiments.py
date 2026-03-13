import pandas as pd

from src.experiments import summarize_results
from src.feature_metadata import final_feature_sets, select_diff_columns


def test_final_feature_sets_list_expected():
    feature_sets = final_feature_sets()
    assert "core" in feature_sets
    assert "core_plus_opponent_adjustment" in feature_sets
    assert "core_plus_efficiency" in feature_sets


def test_final_feature_sets_cover_opponent_adjustment():
    diff_cols = ["Diff_SeedNum", "Diff_AdjustedScoringMargin", "Diff_EffectiveFGPercent"]
    cols = select_diff_columns(diff_cols, "core_plus_opponent_adjustment")
    assert "Diff_AdjustedScoringMargin" in cols
    assert "Diff_EffectiveFGPercent" not in cols


def test_summarize_final_results_produces_recommendation():
    rows = [
        {"split_name": "main_validation", "train_end_season": 2018, "validation_start_season": 2019,
         "feature_set": "core", "model_name": "neural_net", "feature_count": 5,
         "train_rows": 100, "validation_rows": 40, "log_loss": 0.68, "brier_score": 0.21,
         "roc_auc": 0.65, "accuracy": 0.61},
        {"split_name": "main_validation", "train_end_season": 2018, "validation_start_season": 2019,
         "feature_set": "core_plus_opponent_adjustment", "model_name": "logistic_regression",
         "feature_count": 6, "train_rows": 100, "validation_rows": 40, "log_loss": 0.60,
         "brier_score": 0.20, "roc_auc": 0.70, "accuracy": 0.66},
        {"split_name": "backtest_train<= 2014_val>= 2015", "train_end_season": 2014,
         "validation_start_season": 2015, "feature_set": "core", "model_name": "neural_net",
         "feature_count": 5, "train_rows": 90, "validation_rows": 30, "log_loss": 0.72,
         "brier_score": 0.22, "roc_auc": 0.63, "accuracy": 0.60},
        {"split_name": "backtest_train<= 2014_val>= 2015", "train_end_season": 2014,
         "validation_start_season": 2015, "feature_set": "core_plus_opponent_adjustment",
         "model_name": "logistic_regression", "feature_count": 6, "train_rows": 90,
         "validation_rows": 30, "log_loss": 0.65, "brier_score": 0.19, "roc_auc": 0.69,
         "accuracy": 0.67},
    ]
    df = pd.DataFrame(rows)
    summary = summarize_results(df, mode="final")
    assert summary["best_model_main_split"]["model_name"] == "logistic_regression"
    assert summary["recommended_production_configuration"]["feature_set"] == "core_plus_opponent_adjustment"
    assert summary["best_model_backtest_average"]["model_name"] == "logistic_regression"
