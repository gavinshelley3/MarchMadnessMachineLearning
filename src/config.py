from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class PathsConfig:
    """Holds canonical repository paths used throughout the project."""

    base_dir: Path = BASE_DIR
    data_dir: Path = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", ""))
    processed_data_dir: Path = Path(os.getenv("PROCESSED_DATA_DIR", ""))
    outputs_dir: Path = Path(os.getenv("OUTPUTS_DIR", BASE_DIR / "outputs"))
    models_dir: Path = Path(os.getenv("MODELS_DIR", ""))
    supplemental_ncaa_dir: Path = Path(os.getenv("SUPPLEMENTAL_NCAA_DIR", ""))

    def __post_init__(self) -> None:
        if not self.raw_data_dir or str(self.raw_data_dir) == ".":
            self.raw_data_dir = self.data_dir / "raw"
        if not self.processed_data_dir or str(self.processed_data_dir) == ".":
            self.processed_data_dir = self.data_dir / "processed"
        if not self.models_dir or str(self.models_dir) == ".":
            self.models_dir = self.outputs_dir / "models"
        if not self.supplemental_ncaa_dir or str(self.supplemental_ncaa_dir) == ".":
            self.supplemental_ncaa_dir = self.raw_data_dir / "ncaa_basketball"

    def ensure_exists(self) -> None:
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.outputs_dir,
            self.models_dir,
            self.supplemental_ncaa_dir,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    """Training hyperparameters and experiment defaults."""

    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    validation_start_season: int = int(os.getenv("VALIDATION_START_SEASON", "2015"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "128"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.001"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.0001"))
    max_epochs: int = int(os.getenv("MAX_EPOCHS", "200"))
    patience: int = int(os.getenv("PATIENCE", "20"))
    min_delta: float = float(os.getenv("MIN_DELTA", "0.0001"))


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_config() -> Config:
    """Return a ready-to-use config instance with ensured directories."""

    config = Config()
    config.paths.ensure_exists()
    return config
