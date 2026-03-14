"""Train the team-level advancement neural network and compare optimizers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .advancement_dataset import ADVANCEMENT_LABELS, build_advancement_dataset
from .config import get_config
from .evaluate import compute_metrics
from .model import build_advancement_model
from .utils import ensure_parent_dir, save_pickle, set_random_seed


LABEL_NAMES = [label for label, _ in ADVANCEMENT_LABELS]


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tournament advancement neural network.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to Kaggle raw data (defaults to config).")
    parser.add_argument("--validation-start-season", type=int, default=None, help="First season used for validation.")
    parser.add_argument("--include-supplemental-kaggle", action="store_true", help="Include supplemental Kaggle features.")
    parser.add_argument("--include-cbbpy-current", action="store_true", help="Include cached CBBpy features.")
    parser.add_argument("--cbbpy-season", type=int, default=None, help="Season for cached CBBpy features.")
    parser.add_argument("--cbbpy-features", type=Path, default=None, help="Explicit path to cached CBBpy features CSV.")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer to run.")
    parser.add_argument("--compare-optimizers", action="store_true", help="Run both Adam and SGD sequentially.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay / L2 penalty.")
    parser.add_argument("--sgd-momentum", type=float, default=0.9, help="Momentum for SGD (ignored for Adam).")
    parser.add_argument("--max-epochs", type=int, default=150, help="Maximum number of epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument("--min-delta", type=float, default=0.0005, help="Minimum improvement for early stopping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Override outputs/reports directory for comparison artifacts.",
    )
    return parser.parse_args()


def time_based_split(dataset: pd.DataFrame, validation_start_season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = dataset[dataset["Season"] < validation_start_season]
    val_df = dataset[dataset["Season"] >= validation_start_season]
    if train_df.empty:
        raise ValueError("Training split is empty. Choose an earlier --validation-start-season.")
    if val_df.empty:
        latest = dataset["Season"].max()
        val_df = dataset[dataset["Season"] == latest]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def select_feature_columns(dataset: pd.DataFrame) -> List[str]:
    excluded = {"Season", "TeamID", "TeamName", "Seed", "Wins", "PlayInOffset"}
    excluded.update(LABEL_NAMES)
    feature_cols: List[str] = []
    for col, dtype in dataset.dtypes.items():
        if col in excluded:
            continue
        if dtype == object:
            continue
        feature_cols.append(col)
    return feature_cols


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    train_values = train_df[feature_cols].copy()
    train_means = train_values.mean()
    train_values = train_values.fillna(train_means)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_values)
    val_values = val_df[feature_cols].copy().fillna(train_means)
    X_val = scaler.transform(val_values)
    return X_train.astype(np.float32), X_val.astype(np.float32), scaler


def prepare_targets(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y_train = train_df[LABEL_NAMES].to_numpy(dtype=np.float32)
    y_val = val_df[LABEL_NAMES].to_numpy(dtype=np.float32)
    return y_train, y_val


def build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def evaluate_multi_output(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    model.eval()
    losses: List[float] = []
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
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
    y_pred = np.concatenate(preds, axis=0) if preds else np.zeros((0, len(LABEL_NAMES)))
    y_true = np.concatenate(targets, axis=0) if targets else np.zeros((0, len(LABEL_NAMES)))
    per_label: Dict[str, Dict[str, float]] = {}
    log_losses = []
    brier_scores = []
    accuracies = []
    for idx, label in enumerate(LABEL_NAMES):
        if len(y_pred):
            metrics, _ = compute_metrics(y_true[:, idx], y_pred[:, idx])
            per_label[label] = metrics
            log_losses.append(metrics["log_loss"])
            brier_scores.append(metrics["brier_score"])
            accuracies.append(metrics["accuracy"])
        else:
            per_label[label] = {"log_loss": float("nan"), "brier_score": float("nan"), "roc_auc": float("nan"), "accuracy": float("nan")}
    metrics: Dict[str, object] = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "mean_log_loss": float(np.nanmean(log_losses)) if log_losses else float("nan"),
        "mean_brier_score": float(np.nanmean(brier_scores)) if brier_scores else float("nan"),
        "mean_accuracy": float(np.nanmean(accuracies)) if accuracies else float("nan"),
        "per_label": per_label,
    }
    return metrics, y_true, y_pred


def train_with_optimizer(
    optimizer_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: List[str],
    scaler: StandardScaler,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    output_root: Path,
    predictions_dir: Path,
) -> Dict[str, object]:
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_advancement_model(input_dim=X_train.shape[1], output_dim=len(LABEL_NAMES))
    model = model.to(device)
    train_loader = build_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = build_loader(X_val, y_val, args.batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(optimizer_name, model, args.learning_rate, args.weight_decay, args.sgd_momentum)

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_losses = []
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics, _, _ = evaluate_multi_output(model, val_loader, device, criterion)
        print(
            f"[{optimizer_name}] Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | mean_log_loss={val_metrics['mean_log_loss']:.4f}"
        )
        if val_metrics["loss"] + args.min_delta < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[{optimizer_name}] Early stopping triggered.")
                break

    if best_state:
        model.load_state_dict(best_state)
    final_metrics, y_true, y_pred = evaluate_multi_output(model, val_loader, device, criterion)
    artifacts = save_artifacts(
        optimizer_name,
        model,
        scaler,
        feature_cols,
        args,
        output_root,
        predictions_dir,
        train_rows=len(train_df),
        val_rows=len(val_df),
        metrics=final_metrics,
        val_df=val_df,
        predictions=y_pred,
        y_true=y_true,
    )
    return {
        "optimizer": optimizer_name,
        "metrics": final_metrics,
        "artifacts": artifacts,
    }


def save_artifacts(
    optimizer_name: str,
    model: nn.Module,
    scaler: StandardScaler,
    feature_cols: List[str],
    args: argparse.Namespace,
    output_root: Path,
    predictions_dir: Path,
    train_rows: int,
    val_rows: int,
    metrics: Dict[str, object],
    val_df: pd.DataFrame,
    predictions: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Path]:
    model_dir = output_root / optimizer_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    scaler_path = model_dir / "scaler.pkl"
    save_pickle(scaler, scaler_path)
    feature_json = model_dir / "feature_columns.json"
    feature_json.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    save_pickle(feature_cols, model_dir / "feature_columns.pkl")
    metadata = {
        "optimizer": optimizer_name,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "sgd_momentum": args.sgd_momentum,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "train_rows": train_rows,
        "val_rows": val_rows,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")

    preds_path = predictions_dir / f"advancement_validation_{optimizer_name}.csv"
    ensure_parent_dir(preds_path)
    preds_df = val_df[["Season", "TeamID", "TeamName", "Seed", "SeedNum"]].copy()
    for idx, label in enumerate(LABEL_NAMES):
        preds_df[f"pred_{label}"] = predictions[:, idx]
        preds_df[f"true_{label}"] = y_true[:, idx]
    preds_df.to_csv(preds_path, index=False)
    return {"model": model_path, "scaler": scaler_path, "predictions": preds_path}


def save_comparison_report(
    results: List[Dict[str, object]],
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison = {}
    rows = []
    for entry in results:
        opt = entry["optimizer"]
        metrics = entry["metrics"]
        comparison[opt] = metrics
        per_label = metrics.get("per_label", {})
        for label, stats in per_label.items():
            rows.append(
                {
                    "optimizer": opt,
                    "label": label,
                    "log_loss": stats.get("log_loss"),
                    "brier_score": stats.get("brier_score"),
                    "roc_auc": stats.get("roc_auc"),
                    "accuracy": stats.get("accuracy"),
                }
            )
    (reports_dir / "advancement_optimizer_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=_json_default),
        encoding="utf-8",
    )
    if rows:
        pd.DataFrame(rows).to_csv(
            reports_dir / "advancement_optimizer_comparison.csv",
            index=False,
        )


def main() -> None:
    args = parse_args()
    config = get_config()
    reports_dir = args.reports_dir if args.reports_dir else config.paths.outputs_dir / "reports"
    if args.validation_start_season is not None:
        config.training.validation_start_season = args.validation_start_season

    dataset, _ = build_advancement_dataset(
        data_dir=args.data_dir,
        include_supplemental_kaggle=args.include_supplemental_kaggle,
        supplemental_dir=config.paths.supplemental_ncaa_dir,
        include_cbbpy_current=args.include_cbbpy_current,
        cbbpy_features_path=args.cbbpy_features,
        cbbpy_season=args.cbbpy_season,
        reports_dir=reports_dir,
    )
    processed_path = config.paths.processed_data_dir / "advancement_dataset.csv"
    ensure_parent_dir(processed_path)
    dataset.to_csv(processed_path, index=False)

    train_df, val_df = time_based_split(dataset, config.training.validation_start_season)
    feature_cols = select_feature_columns(dataset)
    X_train, X_val, scaler = prepare_features(train_df, val_df, feature_cols)
    y_train, y_val = prepare_targets(train_df, val_df)

    optimizers = ["adam", "sgd"] if args.compare_optimizers else [args.optimizer]
    results: List[Dict[str, object]] = []
    output_root = config.paths.models_dir / "advancement"
    predictions_dir = config.paths.outputs_dir / "predictions"
    for opt in optimizers:
        results.append(
            train_with_optimizer(
                opt,
                X_train,
                y_train,
                X_val,
                y_val,
                feature_cols,
                scaler,
                train_df,
                val_df,
                args,
                output_root,
                predictions_dir,
            )
        )
    save_comparison_report(results, reports_dir)
    print("Advancement training complete. Comparison report saved to", reports_dir)


if __name__ == "__main__":
    main()
