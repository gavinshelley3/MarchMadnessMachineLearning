"""Generate advancement probabilities for the current tournament field."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .advancement_dataset import ADVANCEMENT_LABELS, build_team_context_with_features
from .bracket_generation import PredictionLookup
from .bracket_loader import BracketDefinition, TeamSlot, load_bracket_definition
from .config import get_config
from .data_loading import (
    load_massey_ordinals,
    load_regular_season_detailed_results,
    load_teams,
)
from .dataset_builder import summarize_massey_ordinals
from .model import build_advancement_model
from .supplemental_ncaa import build_supplemental_features
from .utils import ensure_parent_dir, load_pickle


LABEL_NAMES = [label for label, _wins in ADVANCEMENT_LABELS]
ROUND_COLUMN_MAP = {
    "reached_round_of_32": "round_of_32_probability",
    "reached_sweet_16": "sweet_16_probability",
    "reached_elite_8": "elite_8_probability",
    "reached_final_four": "final_four_probability",
    "reached_championship_game": "championship_game_probability",
    "won_championship": "champion_probability",
}


@dataclass(frozen=True)
class FieldEntry:
    season: int
    team_id: int
    team_name: str
    team_key: str
    bracket_display: str
    region: str
    seed: int


def _normalize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]", "", value.lower())
    return cleaned


def _iter_team_payloads(slot: TeamSlot) -> Iterable[Tuple[str, str]]:
    yield slot.team_key, slot.display_name
    for option in slot.options:
        key = option.get("team_key")
        name = option.get("display_name", key)
        if key:
            yield str(key), str(name or key)


def collect_field_entries(
    bracket: BracketDefinition,
    lookup: PredictionLookup,
    teams_df: pd.DataFrame,
) -> Tuple[List[FieldEntry], List[str]]:
    """Resolve the projected field to TeamIDs using predictions + team table."""

    team_map = {
        _normalize_name(str(row.TeamName)): int(row.TeamID)
        for row in teams_df.itertuples(index=False)
        if pd.notna(row.TeamName)
    }
    id_to_name = {
        int(row.TeamID): str(row.TeamName)
        for row in teams_df.itertuples(index=False)
    }
    resolved: Dict[int, FieldEntry] = {}
    missing: List[str] = []

    def resolve_team_id(candidates: Sequence[str]) -> Optional[int]:
        for candidate in candidates:
            if not candidate:
                continue
            team_id = lookup.team_id(candidate)
            if team_id:
                return team_id
            normalized = _normalize_name(candidate)
            if normalized in team_map:
                return team_map[normalized]
        return None

    for matchup in bracket.iter_round_of_64():
        for slot_team in (matchup.team1, matchup.team2):
            for team_key, name in _iter_team_payloads(slot_team):
                candidates = [team_key, name]
                team_id = resolve_team_id(candidates)
                if team_id is None:
                    missing.append(name or team_key)
                    continue
                if team_id in resolved:
                    continue
                official_name = id_to_name.get(team_id, name)
                resolved[team_id] = FieldEntry(
                    season=bracket.season,
                    team_id=team_id,
                    team_name=official_name,
                    team_key=team_key,
                    bracket_display=name,
                    region=matchup.region,
                    seed=int(slot_team.seed),
                )
    return list(resolved.values()), missing


def _seed_code(region: str, seed: int) -> str:
    prefix = "".join(ch for ch in region if ch.isalpha())
    prefix = (prefix[:1] or "R").upper()
    return f"{prefix}{seed:02d}"


def _build_seed_frame(entries: List[FieldEntry]) -> pd.DataFrame:
    rows = []
    seen: set[Tuple[int, int]] = set()
    for entry in entries:
        key = (entry.season, entry.team_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "Season": entry.season,
                "TeamID": entry.team_id,
                "Seed": _seed_code(entry.region, entry.seed),
                "SeedNum": entry.seed,
            }
        )
    return pd.DataFrame(rows)


def _load_cbbpy_features(path: Optional[Path], season: int) -> Optional[pd.DataFrame]:
    if path and path.exists():
        return pd.read_csv(path)
    config = get_config()
    inferred = config.paths.cbbpy_cache_dir / f"cbbpy_team_features_{season}.csv"
    if inferred.exists():
        return pd.read_csv(inferred)
    return None


def _load_supplemental_features(include: bool, teams: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not include:
        return None
    config = get_config()
    supplemental_path = config.paths.supplemental_ncaa_dir
    if not supplemental_path.exists():
        return None
    features, _diagnostics = build_supplemental_features(supplemental_path, teams)
    return features


def prepare_feature_frame(
    season: int,
    entries: List[FieldEntry],
    seeds_df: pd.DataFrame,
    include_supplemental_kaggle: bool,
    include_cbbpy_current: bool,
    cbbpy_features_path: Optional[Path],
    data_dir: Optional[Path],
) -> pd.DataFrame:
    regular = load_regular_season_detailed_results(data_dir)
    season_regular = regular[regular["Season"] == season]
    if season_regular.empty:
        raise ValueError(f"No regular-season data found for season {season}.")
    teams = load_teams(data_dir)
    massey = load_massey_ordinals(data_dir)
    massey_summary = summarize_massey_ordinals(massey, None) if not massey.empty else None
    supplemental = _load_supplemental_features(include_supplemental_kaggle, teams)
    cbbpy_features = _load_cbbpy_features(cbbpy_features_path, season) if include_cbbpy_current else None
    context = build_team_context_with_features(
        season_regular,
        seeds_df,
        massey_summary=massey_summary,
        supplemental_features=supplemental,
        cbbpy_features=cbbpy_features,
    )
    team_frame = pd.DataFrame([entry.__dict__ for entry in entries])
    merged = team_frame.merge(
        context,
        left_on=["season", "team_id"],
        right_on=["Season", "TeamID"],
        how="left",
    )
    merged = merged.merge(
        teams[["TeamID", "TeamName"]],
        left_on="team_id",
        right_on="TeamID",
        how="left",
        suffixes=("", "_official"),
    )
    if "TeamName_official" in merged.columns:
        merged["TeamName"] = merged["TeamName_official"].fillna(merged.get("team_name"))
        merged = merged.drop(columns=["TeamName_official"], errors="ignore")
    else:
        merged["TeamName"] = merged.get("team_name")
    merged = merged.drop(columns=["Season_y", "TeamID_y"], errors="ignore")
    merged = merged.rename(columns={"Season_x": "Season", "TeamID_x": "TeamID"})
    return merged


def _load_feature_columns(feature_path: Path) -> List[str]:
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature column file {feature_path} not found.")
    with feature_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Feature column file {feature_path} must be a JSON list.")
    return [str(col) for col in data]


def _fill_with_scaler_means(df: pd.DataFrame, feature_cols: List[str], scaler: object) -> pd.DataFrame:
    values = df[feature_cols].copy()
    if hasattr(scaler, "mean_") and isinstance(scaler.mean_, np.ndarray):
        means = pd.Series(scaler.mean_, index=feature_cols)
        values = values.fillna(means)
    else:
        values = values.fillna(0.0)
    return values


def run_inference(args: argparse.Namespace) -> Dict[str, object]:
    config = get_config()
    bracket = load_bracket_definition(args.bracket_file)
    if args.season and args.season != bracket.season:
        print(f"[advancement_inference] Overriding bracket season {bracket.season} with CLI season {args.season}.")
        bracket = BracketDefinition(
            season=args.season,
            bracket_type=bracket.bracket_type,
            source_note=bracket.source_note,
            bracket_file=bracket.bracket_file,
            updated_at=bracket.updated_at,
            regions=bracket.regions,
            final_four_pairings=bracket.final_four_pairings,
        )
    baseline_lookup = PredictionLookup(args.baseline_predictions, bracket.season)
    teams_df = load_teams(args.data_dir)
    entries, missing_names = collect_field_entries(bracket, baseline_lookup, teams_df)
    if not entries:
        raise ValueError("No teams were resolved from the bracket file.")
    seeds_df = _build_seed_frame(entries)
    feature_frame = prepare_feature_frame(
        season=bracket.season,
        entries=entries,
        seeds_df=seeds_df,
        include_supplemental_kaggle=args.include_supplemental_kaggle,
        include_cbbpy_current=args.include_cbbpy_current,
        cbbpy_features_path=args.cbbpy_features,
        data_dir=args.data_dir,
    )

    model_path = args.model_path or (config.paths.models_dir / "advancement" / "sgd" / "model.pt")
    scaler_path = args.scaler_path or (config.paths.models_dir / "advancement" / "sgd" / "scaler.pkl")
    feature_cols_path = args.feature_columns or (config.paths.models_dir / "advancement" / "sgd" / "feature_columns.json")
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Advancement model artifacts were not found. Train the model before running inference.")
    feature_cols = _load_feature_columns(feature_cols_path)
    for col in feature_cols:
        if col not in feature_frame.columns:
            feature_frame[col] = np.nan

    raw_values = feature_frame[feature_cols].copy()
    scaler = load_pickle(scaler_path)
    scaler_feature_names = getattr(scaler, "feature_names_in_", None)
    if scaler_feature_names is not None and len(scaler_feature_names) == len(getattr(scaler, "mean_", [])):
        scaler_feature_list = [str(col) for col in scaler_feature_names]
        if len(scaler_feature_list) != len(feature_cols):
            print(
                "[advancement_inference] Overriding feature column list with scaler.feature_names_in_ "
                f"(expected {len(scaler_feature_list)} features, found {len(feature_cols)})."
            )
            feature_cols = scaler_feature_list
            for col in feature_cols:
                if col not in feature_frame.columns:
                    feature_frame[col] = np.nan
            raw_values = feature_frame[feature_cols].copy()
    filled_values = _fill_with_scaler_means(feature_frame, feature_cols, scaler)
    transformed = scaler.transform(filled_values.astype(np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    model = build_advancement_model(input_dim=transformed.shape[1], output_dim=len(LABEL_NAMES))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(transformed, dtype=torch.float32, device=device))
        probabilities = torch.sigmoid(logits).cpu().numpy()

    output_df = feature_frame[
        ["season", "team_id", "team_name", "team_key", "bracket_display", "region", "seed"]
    ].copy()
    for idx, label in enumerate(LABEL_NAMES):
        column = ROUND_COLUMN_MAP[label]
        output_df[column] = probabilities[:, idx]
    output_df = output_df.rename(
        columns={
            "season": "Season",
            "team_id": "TeamID",
            "team_name": "TeamName",
            "bracket_display": "BracketName",
            "seed": "Seed",
        }
    )
    output_df = output_df.sort_values(["region", "Seed"]).reset_index(drop=True)
    ensure_parent_dir(args.output_csv)
    output_df.to_csv(args.output_csv, index=False)

    missing_features = output_df.loc[
        raw_values.isna().any(axis=1), ["TeamName", "TeamID"]
    ].to_dict(orient="records")
    summary = {
        "season": bracket.season,
        "total_teams": int(len(output_df)),
        "missing_team_names": missing_names,
        "missing_feature_rows": missing_features,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_columns": feature_cols,
        "bracket_file": str(args.bracket_file),
        "baseline_predictions": str(args.baseline_predictions),
        "output_csv": str(args.output_csv),
    }
    ensure_parent_dir(args.summary_json)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    config = get_config()
    outputs_dir = config.paths.outputs_dir
    predictions_dir = outputs_dir / "predictions"
    brackets_dir = outputs_dir / "brackets"
    parser = argparse.ArgumentParser(description="Infer advancement probabilities for the projected tournament field.")
    parser.add_argument("--season", type=int, default=2026, help="Tournament season to infer.")
    parser.add_argument("--data-dir", type=Path, default=config.paths.raw_data_dir, help="Root Kaggle data directory.")
    parser.add_argument(
        "--bracket-file",
        type=Path,
        default=Path("data/brackets/projected_2026_bracket.json"),
        help="Bracket JSON describing the projected or official field.",
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=predictions_dir / "2026_matchup_predictions_projected_field_baseline.csv",
        help="Baseline field-only predictions CSV (for resolving TeamIDs).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override path to the trained advancement model weights.",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=None,
        help="Override path to the saved scaler pickle.",
    )
    parser.add_argument(
        "--feature-columns",
        type=Path,
        default=None,
        help="Override path to the JSON list of feature columns.",
    )
    parser.add_argument(
        "--include-supplemental-kaggle",
        action="store_true",
        help="Merge supplemental Kaggle NCAA features before inference.",
    )
    parser.add_argument(
        "--include-cbbpy-current",
        action="store_true",
        help="Merge cached CBBpy current-season features before inference.",
    )
    parser.add_argument(
        "--cbbpy-features",
        type=Path,
        default=None,
        help="Explicit path to cached CBBpy team features CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=predictions_dir / "2026_advancement_probabilities.csv",
        help="Destination CSV for advancement probabilities.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=outputs_dir / "reports" / "advancement_inference_summary.json",
        help="Where to write the inference summary JSON.",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU inference even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_inference(args)
    print(f"[advancement_inference] Saved probabilities to {args.output_csv}")
    print(f"[advancement_inference] Summary written to {args.summary_json}")
    if summary.get("missing_team_names"):
        print(f"[advancement_inference] Missing teams: {summary['missing_team_names']}")


if __name__ == "__main__":
    main()
