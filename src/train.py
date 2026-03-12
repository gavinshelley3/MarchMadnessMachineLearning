from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import get_config
from .data_pipeline import DatasetDiagnostics, load_and_build_dataset
from .evaluate import evaluate_model, print_evaluation_summary
from .model import build_model
from .utils import ensure_parent_dir, save_pickle, set_random_seed


def build_datasets(data_dir: Path | None = None):
    dataset, diagnostics = load_and_build_dataset(data_dir)
    return dataset, diagnostics


def time_based_split(
    dataset: pd.DataFrame, validation_start_season: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = dataset[dataset["Season"] < validation_start_season]
    val_df = dataset[dataset["Season"] >= validation_start_season]
    if train_df.empty:
        raise ValueError(
            "Training split is empty. Choose an earlier validation_start_season."
        )
    if val_df.empty:
        latest_season = dataset["Season"].max()
        val_df = dataset[dataset["Season"] == latest_season]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def prepare_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    feature_cols = [col for col in train_df.columns if col.startswith("Diff_")]
    if not feature_cols:
        raise ValueError("No feature columns found. Ensure feature engineering ran correctly.")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    y_train = train_df["Label"].values.astype(np.float32)
    y_val = val_df["Label"].values.astype(np.float32)
    return X_train, y_train, X_val, y_val, feature_cols, scaler


def build_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
) -> Tuple[nn.Module, dict, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.training.max_epochs + 1):
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
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] + config.training.min_delta < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                print("Early stopping triggered.")
                break

    if best_state:
        model.load_state_dict(best_state)
    final_metrics, y_true, y_pred = evaluate_model(
        model, val_loader, device, criterion, return_predictions=True
    )
    print_evaluation_summary(final_metrics)
    return model, final_metrics, y_true, y_pred


def save_artifacts(model, scaler, feature_cols, config) -> None:
    model_path = config.paths.models_dir / "tabular_nn.pt"
    ensure_parent_dir(model_path)
    torch.save(model.state_dict(), model_path)

    scaler_path = config.paths.models_dir / "scaler.pkl"
    save_pickle(scaler, scaler_path)

    features_path = config.paths.models_dir / "feature_columns.pkl"
    save_pickle(feature_cols, features_path)


def save_reports(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    predictions: np.ndarray,
    metrics: dict,
    feature_cols: List[str],
    config,
) -> None:
    reports_dir = config.paths.outputs_dir / "reports"
    predictions_path = reports_dir / "val_predictions.csv"
    ensure_parent_dir(predictions_path)

    val_metadata_cols = ["MatchupID", "Season", "Team1ID", "Team2ID", "Label"]
    missing_cols = [col for col in val_metadata_cols if col not in val_df.columns]
    if missing_cols:
        raise ValueError(f"Validation DataFrame missing columns: {missing_cols}")
    if len(predictions) != len(val_df):
        raise ValueError("Prediction count does not match validation rows.")
    preds_df = val_df[val_metadata_cols].copy()
    preds_df["PredProb"] = predictions
    preds_df["PredClass"] = (predictions >= 0.5).astype(int)
    preds_df.to_csv(predictions_path, index=False)

    metrics_payload = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "log_loss": metrics["log_loss"],
        "accuracy": metrics["accuracy"],
        "average_bce_loss": metrics.get("loss"),
    }
    metrics_path = reports_dir / "val_metrics.json"
    ensure_parent_dir(metrics_path)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    features_json_path = reports_dir / "feature_columns.json"
    ensure_parent_dir(features_json_path)
    features_json_path.write_text(json.dumps(feature_cols, indent=2))


def log_dataset_diagnostics(diag: DatasetDiagnostics) -> None:
    print("=== Dataset build diagnostics ===")
    print(
        f"Tournament games: {diag.total_tourney_games} -> symmetric rows after filtering: {diag.rows_after_filter}"
    )
    print(f"Dropped rows: {diag.dropped_rows}")
    if diag.dropped_seasons:
        print(f"Seasons removed due to missing features: {diag.dropped_seasons}")
    print(f"Team feature join coverage: {diag.team_feature_join_coverage:.2%}")
    print(f"Seed parse coverage: {diag.seed_parse_coverage:.2%}")
    print(f"Seed numeric coverage (modeling set): {diag.seed_numeric_coverage:.2%}")
    print(f"Massey coverage (modeling set): {diag.massey_coverage:.2%}")
    print(f"Feature columns: {diag.feature_count}")
    print("Label distribution:", diag.class_balance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the March Madness model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing Kaggle CSVs. Defaults to config paths.",
    )
    parser.add_argument(
        "--validation-start-season",
        type=int,
        default=None,
        help="First season to use for validation (inclusive). Overrides config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    if args.validation_start_season is not None:
        config.training.validation_start_season = args.validation_start_season
    set_random_seed(config.training.random_seed)

    dataset, diagnostics = build_datasets(args.data_dir)
    log_dataset_diagnostics(diagnostics)
    processed_path = config.paths.processed_data_dir / "matchup_dataset.csv"
    ensure_parent_dir(processed_path)
    dataset.to_csv(processed_path, index=False)

    train_df, val_df = time_based_split(dataset, config.training.validation_start_season)
    X_train, y_train, X_val, y_val, feature_cols, scaler = prepare_features(
        train_df, val_df
    )

    print(
        f"Training seasons: {train_df['Season'].min()} - {train_df['Season'].max()} ({len(train_df)} rows)"
    )
    print(
        f"Validation seasons: {val_df['Season'].min()} - {val_df['Season'].max()} ({len(val_df)} rows)"
    )
    print(f"Feature columns used: {len(feature_cols)}")

    train_loader = build_dataloader(X_train, y_train, config.training.batch_size, True)
    val_loader = build_dataloader(X_val, y_val, config.training.batch_size, False)

    model = build_model(input_dim=X_train.shape[1])
    trained_model, final_metrics, _, val_pred = train_model(model, train_loader, val_loader, config)
    save_artifacts(trained_model, scaler, feature_cols, config)
    save_reports(train_df, val_df, val_pred, final_metrics, feature_cols, config)
    print("Training complete. Artifacts saved to", config.paths.models_dir)


if __name__ == "__main__":
    main()
