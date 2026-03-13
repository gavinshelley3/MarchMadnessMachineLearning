from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

MEN_FILE_HINTS = ("mbb", "men", "cbb", "m_", "/m", "\\m")
WOMEN_FILE_HINTS = ("wbb", "women", "wncaa", "w_", "/w", "\\w")


def _classify_gender_from_name(name: str) -> str:
    lowered = name.lower()
    if any(hint in lowered for hint in MEN_FILE_HINTS):
        return "men"
    if any(hint in lowered for hint in WOMEN_FILE_HINTS):
        return "women"
    # March Mania naming uses leading M/W
    if lowered.startswith("m"):
        return "men"
    if lowered.startswith("w"):
        return "women"
    return "unknown"


@dataclass
class InventoryEntry:
    dataset: str
    relative_path: str
    size_bytes: int
    gender: str
    selected_for_use: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _scan_directory(
    root: Path,
    dataset_label: str,
    men_predicate,
) -> List[InventoryEntry]:
    entries: List[InventoryEntry] = []
    if not root.exists():
        return entries
    for path in sorted(root.rglob("*.csv")):
        if path.is_file():
            gender = men_predicate(path.name)
            entries.append(
                InventoryEntry(
                    dataset=dataset_label,
                    relative_path=str(path.relative_to(root)),
                    size_bytes=path.stat().st_size,
                    gender=gender,
                    selected_for_use=(gender == "men"),
                )
            )
    return entries


def build_local_dataset_inventory(
    base_data_dir: Path,
    supplemental_dir: Path,
) -> Dict[str, object]:
    base_entries = _scan_directory(
        base_data_dir,
        "march_machine_learning_mania",
        _classify_gender_from_name,
    )
    supplemental_entries = _scan_directory(
        supplemental_dir,
        "ncaa_basketball",
        _classify_gender_from_name,
    )
    return {
        "base_data_dir": str(base_data_dir),
        "supplemental_data_dir": str(supplemental_dir),
        "inventory": [entry.to_dict() for entry in base_entries + supplemental_entries],
    }


def write_local_dataset_inventory(
    base_data_dir: Path,
    supplemental_dir: Path,
    output_path: Path,
) -> None:
    payload = build_local_dataset_inventory(base_data_dir, supplemental_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
