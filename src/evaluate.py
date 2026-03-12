from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import torch
from torch.utils.data import DataLoader


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    return {
        "log_loss": float(log_loss(y_true, y_pred_proba, eps=1e-7)),
        "accuracy": float(
            accuracy_score(y_true, (y_pred_proba >= 0.5).astype(int))
        ),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> Dict[str, float]:
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
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def print_evaluation_summary(metrics: Dict[str, float]) -> None:
    summary = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(f"Evaluation -> {summary}")
