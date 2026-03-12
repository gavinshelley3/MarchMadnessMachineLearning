import numpy as np
import pandas as pd

from src.evaluate import (
    compute_metrics,
    compute_calibration_table,
    compute_seed_gap_metrics,
    compute_upset_metrics,
)


def test_compute_metrics_outputs_expected_values() -> None:
    y_true = np.array([0, 1, 1, 0], dtype=float)
    y_prob = np.array([0.1, 0.8, 0.6, 0.4], dtype=float)
    metrics, preds = compute_metrics(y_true, y_prob)
    assert set(["log_loss", "brier_score", "roc_auc", "accuracy"]).issubset(metrics.keys())
    assert metrics["accuracy"] == 1.0
    assert preds.tolist() == [0, 1, 1, 0]


def test_compute_calibration_table_bins_sum_to_samples() -> None:
    y_true = np.array([0, 1, 0, 1, 1], dtype=float)
    y_prob = np.array([0.05, 0.25, 0.55, 0.75, 0.95], dtype=float)
    table = compute_calibration_table(y_true, y_prob, n_bins=5)
    assert table["sample_count"].sum() == len(y_true)
    assert not table.empty


def test_seed_gap_metrics_bucket_counts() -> None:
    val_df = pd.DataFrame(
        {
            "Label": [1, 0, 1],
            "Team1_SeedNum": [1, 6, 12],
            "Team2_SeedNum": [2, 4, 5],
        }
    )
    preds = np.array([0.7, 0.4, 0.3])
    result = compute_seed_gap_metrics(val_df, preds)
    assert set(result.keys()) == {"even", "moderate", "large"}
    assert result["even"]["sample_count"] == 1
    assert result["large"]["sample_count"] == 1


def test_upset_metrics_handles_no_upsets() -> None:
    val_df = pd.DataFrame(
        {
            "Label": [1, 0],
            "Team1_SeedNum": [1, 2],
            "Team2_SeedNum": [4, 5],
        }
    )
    preds = np.array([0.9, 0.2])
    metrics = compute_upset_metrics(val_df, preds)
    assert metrics["sample_count"] == 0
    assert np.isnan(metrics["upset_accuracy"])
