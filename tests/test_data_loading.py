from pathlib import Path

import pandas as pd

from src import data_loading


def _write_csv(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_resolve_path_prefers_nested_full_dataset(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    nested_dir = raw_dir / "march_machine_learning_mania_2026"
    nested_dir.mkdir(parents=True)

    # Small placeholder file at root.
    root_file = raw_dir / "MTeams.csv"
    _write_csv(root_file, [{"TeamID": 1, "TeamName": "Sample"}])

    # Real file (larger) inside nested Kaggle folder.
    nested_file = nested_dir / "MTeams.csv"
    _write_csv(
        nested_file,
        [{"TeamID": 1, "TeamName": "Sample"}, {"TeamID": 2, "TeamName": "Real"}],
    )

    resolved = data_loading._resolve_path("MTeams.csv", data_dir=raw_dir)
    assert resolved == nested_file


def test_resolve_path_falls_back_to_direct_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_path = raw_dir / "MTeams.csv"
    _write_csv(csv_path, [{"TeamID": 1, "TeamName": "Only"}])

    resolved = data_loading._resolve_path("MTeams.csv", data_dir=raw_dir)
    assert resolved == csv_path
