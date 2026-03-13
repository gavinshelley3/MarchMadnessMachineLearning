from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MEN_SUFFIXES = ("mbb", "cbb")
WOMEN_SUFFIXES = ("wbb", "wncaa")


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return normalized


def _classify_gender(file_name: str) -> str:
    lowered = file_name.lower()
    if any(token in lowered for token in WOMEN_SUFFIXES):
        return "women"
    if any(token in lowered for token in MEN_SUFFIXES):
        return "men"
    if lowered.startswith("m"):
        return "men"
    if lowered.startswith("w"):
        return "women"
    return "unknown"


def _infer_year_from_filename(path: Path) -> Optional[int]:
    match = re.search(r"(\d{2,4})", path.stem)
    if not match:
        return None
    digits = match.group(1)
    year = int(digits)
    if year < 100:
        year += 2000
    return year


@dataclass
class MappingDiagnostics:
    table: str
    total_rows: int
    matched_rows: int
    coverage: float
    unmatched_examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class SupplementalDiagnostics:
    tables_used: List[str] = field(default_factory=list)
    tables_skipped: List[str] = field(default_factory=list)
    gender_counts: Dict[str, int] = field(default_factory=dict)
    mapping: List[MappingDiagnostics] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    season_coverage: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "tables_used": self.tables_used,
            "tables_skipped": self.tables_skipped,
            "gender_counts": self.gender_counts,
            "mapping": [entry.to_dict() for entry in self.mapping],
            "feature_columns": self.feature_columns,
            "season_coverage": self.season_coverage,
            "notes": self.notes,
        }


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]
    return df


def _load_cbb_file(path: Path, diagnostics: SupplementalDiagnostics) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - safety net for large files
        diagnostics.notes.append(f"Failed to read {path.name}: {exc}")
        return pd.DataFrame()
    df = _standardize_columns(df)
    if "team" not in df.columns:
        diagnostics.notes.append(f"{path.name} missing 'Team' column.")
        return pd.DataFrame()
    if "year" in df.columns:
        df["season"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    elif "season" not in df.columns:
        inferred = _infer_year_from_filename(path)
        df["season"] = inferred
    df = df.dropna(subset=["season"])
    df["season"] = df["season"].astype(int)
    df["team_name"] = df["team"].astype(str)
    df["source"] = path.name
    return df


def _load_cbb_tables(directory: Path, diagnostics: SupplementalDiagnostics) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(directory.glob("*.csv")):
        gender = _classify_gender(csv_path.name)
        diagnostics.gender_counts[gender] = diagnostics.gender_counts.get(gender, 0) + 1
        if not csv_path.name.lower().startswith("cbb"):
            continue
        if gender != "men":
            diagnostics.tables_skipped.append(csv_path.name)
            continue
        diagnostics.tables_used.append(csv_path.name)
        frame = _load_cbb_file(csv_path, diagnostics)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.dropna(subset=["team_name", "season"])
    combined = combined.sort_values(["season", "team_name"])
    combined = combined.drop_duplicates(subset=["season", "team_name"], keep="last")
    return combined


def _map_to_team_ids(
    source_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    diagnostics: SupplementalDiagnostics,
) -> pd.DataFrame:
    if teams_df is None or teams_df.empty:
        diagnostics.notes.append("Teams dataframe missing; cannot map supplemental data.")
        return pd.DataFrame()
    lookup = teams_df[["TeamID", "TeamName"]].copy()
    lookup["name_key"] = lookup["TeamName"].map(_normalize_name)
    merged = source_df.copy()
    merged["name_key"] = merged["team_name"].map(_normalize_name)
    merged = merged.merge(lookup[["TeamID", "name_key"]], on="name_key", how="left")
    total_rows = len(merged)
    matched_rows = merged["TeamID"].notna().sum()
    coverage = float(matched_rows / total_rows) if total_rows else 0.0
    unmatched_examples = (
        merged.loc[merged["TeamID"].isna(), "team_name"].drop_duplicates().head(10).tolist()
    )
    diagnostics.mapping.append(
        MappingDiagnostics(
            table="cbb",
            total_rows=total_rows,
            matched_rows=int(matched_rows),
            coverage=coverage,
            unmatched_examples=unmatched_examples,
        )
    )
    merged = merged.dropna(subset=["TeamID"])
    merged["TeamID"] = merged["TeamID"].astype(int)
    merged.rename(columns={"season": "Season"}, inplace=True)
    return merged


def _select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SuppAdjTempo"] = df.get("adj_t")
    df["SuppAdjEfficiencyMargin"] = df.get("adjoe") - df.get("adjde")
    df["SuppBarthag"] = df.get("barthag")
    df["SuppWab"] = df.get("wab")
    efg_o = df.get("efg_o")
    efg_d = df.get("efg_d")
    df["SuppEffectiveFgDiff"] = efg_o - efg_d if efg_o is not None and efg_d is not None else np.nan
    columns = [
        "Season",
        "TeamID",
        "SuppAdjTempo",
        "SuppAdjEfficiencyMargin",
        "SuppBarthag",
        "SuppWab",
        "SuppEffectiveFgDiff",
    ]
    df = df[columns]
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(0.0)
    df = df.sort_values(["Season", "TeamID"]).reset_index(drop=True)
    return df


def build_supplemental_features(
    supplemental_dir: Path,
    teams_df: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, SupplementalDiagnostics]:
    diagnostics = SupplementalDiagnostics()
    if not supplemental_dir.exists():
        diagnostics.notes.append(f"Supplemental directory {supplemental_dir} missing.")
        return pd.DataFrame(columns=["Season", "TeamID"]), diagnostics

    cbb_df = _load_cbb_tables(supplemental_dir, diagnostics)
    if cbb_df.empty:
        diagnostics.notes.append("No usable cbb tables found for supplemental features.")
        return pd.DataFrame(columns=["Season", "TeamID"]), diagnostics

    mapped = _map_to_team_ids(cbb_df, teams_df, diagnostics)
    if mapped.empty:
        diagnostics.notes.append("Supplemental tables present but no teams matched.")
        return pd.DataFrame(columns=["Season", "TeamID"]), diagnostics

    features = _select_feature_columns(mapped)
    diagnostics.feature_columns = [col for col in features.columns if col not in {"Season", "TeamID"}]
    diagnostics.season_coverage = {
        "min_season": int(features["Season"].min()),
        "max_season": int(features["Season"].max()),
        "row_count": int(len(features)),
    }
    return features, diagnostics


def save_supplemental_report(diagnostics: SupplementalDiagnostics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostics.to_dict(), indent=2))
