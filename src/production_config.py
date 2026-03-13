from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config


PRODUCTION_MODEL_NAME = "logistic_regression"
PRODUCTION_FEATURE_SET = "core_plus_opponent_adjustment"
PRODUCTION_SEASON = 2026


@dataclass(frozen=True)
class ProductionArtifactPaths:
    directory: Path
    model_path: Path
    scaler_path: Path
    features_path: Path
    metadata_path: Path


def get_production_dir(config: Config) -> Path:
    """Return the directory that stores production-ready artifacts."""

    return config.paths.models_dir / "production"


def get_predictions_dir(config: Config) -> Path:
    """Return the directory used for saved inference outputs."""

    return config.paths.outputs_dir / "predictions"


def get_production_paths(config: Config) -> ProductionArtifactPaths:
    """Materialize canonical artifact paths for the locked production model."""

    base = get_production_dir(config)
    return ProductionArtifactPaths(
        directory=base,
        model_path=base / "logistic_model.pkl",
        scaler_path=base / "scaler.pkl",
        features_path=base / "feature_columns.json",
        metadata_path=base / "metadata.json",
    )
