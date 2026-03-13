from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import Config
from .data_pipeline import load_and_build_dataset
from .feature_metadata import select_diff_columns
from .production_config import (
    PRODUCTION_FEATURE_SET,
    PRODUCTION_MODEL_NAME,
    get_production_paths,
)
from .train import fit_logistic_regression_model
from .utils import ensure_parent_dir, load_pickle, save_pickle


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=str)


def _json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def artifacts_available(config: Config) -> bool:
    """Return True if all production artifacts are present."""

    paths = get_production_paths(config)
    required = [
        paths.model_path,
        paths.scaler_path,
        paths.features_path,
        paths.metadata_path,
    ]
    return all(path.exists() for path in required)


def train_and_save_production_model(
    config: Config,
    data_dir: Path | None = None,
    feature_set: str = PRODUCTION_FEATURE_SET,
) -> Dict[str, Any]:
    """Train the locked production logistic regression model and persist artifacts."""

    dataset, diagnostics = load_and_build_dataset(data_dir)
    diff_cols = [col for col in dataset.columns if col.startswith("Diff_")]
    selected_cols = select_diff_columns(diff_cols, feature_set)
    if not selected_cols:
        raise ValueError(f"No columns found for feature set '{feature_set}'.")

    scaler = StandardScaler()
    X = scaler.fit_transform(dataset[selected_cols])
    y = dataset["Label"].to_numpy(dtype=np.float32)
    model = fit_logistic_regression_model(X, y)

    paths = get_production_paths(config)
    ensure_parent_dir(paths.model_path)
    save_pickle(model, paths.model_path)
    save_pickle(scaler, paths.scaler_path)
    paths.features_path.write_text(json.dumps(selected_cols, indent=2))

    metadata = {
        "model_name": PRODUCTION_MODEL_NAME,
        "feature_set": feature_set,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(dataset)),
        "season_min": int(dataset["Season"].min()),
        "season_max": int(dataset["Season"].max()),
        "diff_feature_count": len(selected_cols),
        "team_feature_columns": diagnostics.feature_columns,
    }
    _json_dump(paths.metadata_path, metadata)
    return metadata


def load_production_artifacts(
    config: Config,
) -> Tuple[Any, StandardScaler, list[str], Dict[str, Any]]:
    """Load persisted artifacts for inference."""

    paths = get_production_paths(config)
    model = load_pickle(paths.model_path)
    scaler = load_pickle(paths.scaler_path)
    feature_cols = json.loads(paths.features_path.read_text())
    metadata = _json_load(paths.metadata_path)
    return model, scaler, feature_cols, metadata


def ensure_production_artifacts(
    config: Config,
    data_dir: Path | None = None,
    feature_set: str = PRODUCTION_FEATURE_SET,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Guarantee that artifacts exist, training them if needed."""

    if not force_retrain and artifacts_available(config):
        _, _, _, metadata = load_production_artifacts(config)
        return metadata
    return train_and_save_production_model(config, data_dir=data_dir, feature_set=feature_set)
