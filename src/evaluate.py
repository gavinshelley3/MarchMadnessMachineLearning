from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
import torch
from torch.utils.data import DataLoader


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    clipped = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
    predicted_class = (clipped >= 0.5).astype(int)
    metrics = {
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "roc_auc": _safe_roc_auc(y_true, clipped),
        "accuracy": float(accuracy_score(y_true, predicted_class)),
    }
    return metrics, predicted_class


def compute_confusion(y_true: np.ndarray, predicted_class: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    matrix = confusion_matrix(y_true, predicted_class, labels=[0, 1])
    return tuple(map(tuple, matrix))


def compute_calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    if len(y_prob) == 0:
        return pd.DataFrame(columns=["bin", "avg_pred_prob", "actual_win_rate", "sample_count"])
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    bins = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["y_prob"], bins=bins, include_lowest=True)
    grouped = df.groupby("bin", observed=False)
    table = grouped.agg(
        avg_pred_prob=("y_prob", "mean"),
        actual_win_rate=("y_true", "mean"),
        sample_count=("y_true", "size"),
    ).reset_index()
    table["bin_lower"] = table["bin"].apply(lambda interval: interval.left if interval else np.nan)
    table["bin_upper"] = table["bin"].apply(lambda interval: interval.right if interval else np.nan)
    table = table[["bin", "bin_lower", "bin_upper", "avg_pred_prob", "actual_win_rate", "sample_count"]]
    return table


def compute_seed_gap_metrics(val_df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Dict[str, float]]:
    required = {"Team1_SeedNum", "Team2_SeedNum", "Label"}
    if not required.issubset(val_df.columns):
        return {}
    df = val_df[list(required)].copy()
    df["PredProb"] = predictions
    df = df.dropna(subset=["Team1_SeedNum", "Team2_SeedNum"])
    if df.empty:
        return {}
    df["SeedDiff"] = df["Team1_SeedNum"] - df["Team2_SeedNum"]

    def bucket(diff: float) -> str:
        abs_diff = abs(diff)
        if abs_diff <= 1:
            return "even"
        if abs_diff <= 5:
            return "moderate"
        return "large"

    df["Bucket"] = df["SeedDiff"].apply(bucket)
    results: Dict[str, Dict[str, float]] = {}
    for bucket_name, group in df.groupby("Bucket"):
        metrics, preds = compute_metrics(group["Label"].to_numpy(), group["PredProb"].to_numpy())
        results[bucket_name] = {
            "accuracy": metrics["accuracy"],
            "log_loss": metrics["log_loss"],
            "sample_count": int(len(group)),
        }
    return results


def compute_upset_metrics(val_df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
    if "Team1_SeedNum" not in val_df.columns or "Team2_SeedNum" not in val_df.columns:
        return {"upset_accuracy": float("nan"), "upset_log_loss": float("nan"), "sample_count": 0}
    df = val_df[["Label", "Team1_SeedNum", "Team2_SeedNum"]].copy()
    df["PredProb"] = predictions
    df = df.dropna(subset=["Team1_SeedNum", "Team2_SeedNum"])
    df["SeedDiff"] = df["Team1_SeedNum"] - df["Team2_SeedNum"]
    upset_df = df[df["SeedDiff"] > 0]
    if upset_df.empty:
        return {"upset_accuracy": float("nan"), "upset_log_loss": float("nan"), "sample_count": 0}
    metrics, _ = compute_metrics(upset_df["Label"].to_numpy(), upset_df["PredProb"].to_numpy())
    return {
        "upset_accuracy": metrics["accuracy"],
        "upset_log_loss": metrics["log_loss"],
        "sample_count": int(len(upset_df)),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    return_predictions: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray] | Dict[str, float]:
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
            targets.append(labels.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0) if preds else np.array([])
    y_true = np.concatenate(targets, axis=0) if targets else np.array([])
    if len(y_pred):
        metrics, predicted_class = compute_metrics(y_true, y_pred)
        metrics["confusion_matrix"] = compute_confusion(y_true, predicted_class)
    else:
        metrics = {"log_loss": 0.0, "accuracy": 0.0, "brier_score": 0.0, "roc_auc": float("nan"), "confusion_matrix": ((0, 0), (0, 0))}
    metrics["loss"] = float(np.mean(losses))
    if return_predictions:
        return metrics, y_true, y_pred
    return metrics


def print_evaluation_summary(metrics: Dict[str, float]) -> None:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    summary = ", ".join(parts)
    print(f"Evaluation -> {summary}")
