from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from .config import get_config

CORE_FILES = {
    "teams": "MTeams.csv",
    "regular_season_detailed": "MRegularSeasonDetailedResults.csv",
    "tourney_compact": "MNCAATourneyCompactResults.csv",
    "tourney_seeds": "MNCAATourneySeeds.csv",
    "massey_ordinals": "MMasseyOrdinals.csv",
    "team_spellings": "MTeamSpellings.csv",
}


def _resolve_path(filename: str, data_dir: Optional[Path] = None) -> Path:
    config = get_config()
    base_dir = Path(data_dir) if data_dir else config.paths.raw_data_dir
    candidate_dirs: Sequence[Path] = (
        base_dir,
        base_dir / "march_machine_learning_mania_2026",
        base_dir / "march-machine-learning-mania-2026",
    )
    matches = []
    for directory in candidate_dirs:
        path = directory / filename
        if path.exists():
            matches.append(path)
    if not matches:
        # Fall back to searching the raw data tree so we can support arbitrary Kaggle folder layouts.
        matches = list(base_dir.rglob(filename))
    if matches:
        # Prefer the largest file in case lightweight samples were left at the root.
        return max(matches, key=lambda p: p.stat().st_size)
    raise FileNotFoundError(
        f"Expected file '{filename}' under {base_dir} (or nested directories). "
        "Place the Kaggle CSVs under data/raw/march_machine_learning_mania_2026/."
    )


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


def load_team_spellings(data_dir: Optional[Path] = None) -> pd.DataFrame:
    return _load_csv(CORE_FILES["team_spellings"], data_dir)


def load_all_core_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Convenience helper that loads all key Kaggle CSVs at once."""

    return {
        "teams": load_teams(data_dir),
        "regular_season_detailed": load_regular_season_detailed_results(data_dir),
        "tourney_compact": load_tourney_compact_results(data_dir),
        "tourney_seeds": load_tourney_seeds(data_dir),
        "massey_ordinals": load_massey_ordinals(data_dir),
    }
