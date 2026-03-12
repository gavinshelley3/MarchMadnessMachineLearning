from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import get_config
from . import data_loading
from .dataset_builder import build_matchup_dataset
from .evaluate import evaluate_model, print_evaluation_summary
from .model import build_model
from .utils import ensure_parent_dir, save_pickle, set_random_seed


def build_datasets(data_dir: Path | None = None) -> pd.DataFrame:
    regular = data_loading.load_regular_season_detailed_results(data_dir)
    tourney = data_loading.load_tourney_compact_results(data_dir)
    seeds = data_loading.load_tourney_seeds(data_dir)
    massey = data_loading.load_massey_ordinals(data_dir)
    dataset = build_matchup_dataset(regular, tourney, seeds, massey)
    return dataset


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
    return train_df, val_df


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
) -> Tuple[nn.Module, dict]:
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
    final_metrics = evaluate_model(model, val_loader, device, criterion)
    print_evaluation_summary(final_metrics)
    return model, final_metrics


def save_artifacts(model, scaler, feature_cols, config) -> None:
    model_path = config.paths.models_dir / "tabular_nn.pt"
    ensure_parent_dir(model_path)
    torch.save(model.state_dict(), model_path)

    scaler_path = config.paths.models_dir / "scaler.pkl"
    save_pickle(scaler, scaler_path)

    features_path = config.paths.models_dir / "feature_columns.pkl"
    save_pickle(feature_cols, features_path)


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

    dataset = build_datasets(args.data_dir)
    processed_path = config.paths.processed_data_dir / "matchup_dataset.csv"
    ensure_parent_dir(processed_path)
    dataset.to_csv(processed_path, index=False)

    train_df, val_df = time_based_split(dataset, config.training.validation_start_season)
    X_train, y_train, X_val, y_val, feature_cols, scaler = prepare_features(
        train_df, val_df
    )

    train_loader = build_dataloader(X_train, y_train, config.training.batch_size, True)
    val_loader = build_dataloader(X_val, y_val, config.training.batch_size, False)

    model = build_model(input_dim=X_train.shape[1])
    trained_model, _ = train_model(model, train_loader, val_loader, config)
    save_artifacts(trained_model, scaler, feature_cols, config)
    print("Training complete. Artifacts saved to", config.paths.models_dir)


if __name__ == "__main__":
    main()
