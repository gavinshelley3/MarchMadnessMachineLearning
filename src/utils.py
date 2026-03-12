from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Seed python, numpy, and torch (cpu + cuda when available)."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory of `path` if it is missing."""

    path.parent.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("wb") as fp:
        pickle.dump(obj, fp)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fp:
        return pickle.load(fp)
