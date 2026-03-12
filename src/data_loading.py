from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import get_config

CORE_FILES = {
    "teams": "MTeams.csv",
    "regular_season_detailed": "MRegularSeasonDetailedResults.csv",
    "tourney_compact": "MNCAATourneyCompactResults.csv",
    "tourney_seeds": "MNCAATourneySeeds.csv",
    "massey_ordinals": "MMasseyOrdinals.csv",
}


def _resolve_path(filename: str, data_dir: Optional[Path] = None) -> Path:
    config = get_config()
    base_dir = Path(data_dir) if data_dir else config.paths.raw_data_dir
    path = base_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file '{filename}' under {base_dir}. Place Kaggle CSVs there."
        )
    return path


def _load_csv(filename: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    path = _resolve_path(filename, data_dir)
    return pd.read_csv(path)


def load_teams(data_dir: Optional[Path] = None) -> pd.DataFrame:
    return _load_csv(CORE_FILES["teams"], data_dir)


def load_regular_season_detailed_results(data_dir: Optional[Path] = None) -> pd.DataFrame:
    return _load_csv(CORE_FILES["regular_season_detailed"], data_dir)


def load_tourney_compact_results(data_dir: Optional[Path] = None) -> pd.DataFrame:
    return _load_csv(CORE_FILES["tourney_compact"], data_dir)


def load_tourney_seeds(data_dir: Optional[Path] = None) -> pd.DataFrame:
    return _load_csv(CORE_FILES["tourney_seeds"], data_dir)


def load_massey_ordinals(data_dir: Optional[Path] = None) -> pd.DataFrame:
    return _load_csv(CORE_FILES["massey_ordinals"], data_dir)


def load_all_core_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Convenience helper that loads all key Kaggle CSVs at once."""

    return {
        "teams": load_teams(data_dir),
        "regular_season_detailed": load_regular_season_detailed_results(data_dir),
        "tourney_compact": load_tourney_compact_results(data_dir),
        "tourney_seeds": load_tourney_seeds(data_dir),
        "massey_ordinals": load_massey_ordinals(data_dir),
    }
