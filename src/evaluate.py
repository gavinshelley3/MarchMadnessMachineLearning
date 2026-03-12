from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import torch
from torch.utils.data import DataLoader


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    clipped = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
    return {
        "log_loss": float(log_loss(y_true, clipped)),
        "accuracy": float(
            accuracy_score(y_true, (clipped >= 0.5).astype(int))
        ),
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
    metrics = compute_metrics(y_true, y_pred) if len(y_pred) else {"log_loss": 0.0, "accuracy": 0.0}
    metrics["loss"] = float(np.mean(losses))
    if return_predictions:
        return metrics, y_true, y_pred
    return metrics


def print_evaluation_summary(metrics: Dict[str, float]) -> None:
    summary = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(f"Evaluation -> {summary}")
