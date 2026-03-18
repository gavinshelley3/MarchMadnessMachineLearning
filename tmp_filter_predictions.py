import json
import difflib
from pathlib import Path

import pandas as pd

from src import data_loading
from src.advancement_inference import _normalize_name as _normalize

BRACKET_PATH = Path("data/brackets/official_2026_bracket.json")
BASELINE_INPUT = Path("outputs/predictions/2026_matchup_predictions_baseline.csv")
CBBPY_INPUT = Path("outputs/predictions/2026_matchup_predictions_cbbpy.csv")
BASELINE_OUTPUT = Path("outputs/predictions/2026_matchup_predictions_official_field_baseline.csv")
CBBPY_OUTPUT = Path("outputs/predictions/2026_matchup_predictions_official_field_cbbpy.csv")
ALIASES = {"miami": "miamifl"}


def build_team_lookup() -> dict[str, int]:
    teams = data_loading.load_teams()
    mapping: dict[str, int] = {}
    for row in teams.itertuples(index=False):
        mapping[_normalize(row.TeamName)] = int(row.TeamID)
    spellings = pd.read_csv(data_loading._resolve_path('MTeamSpellings.csv'))  # type: ignore[attr-defined]
    for row in spellings.itertuples(index=False):
        mapping[_normalize(row.TeamNameSpelling)] = int(row.TeamID)
    return mapping


def extract_team_entries(bracket: dict) -> list[dict]:
    entries: list[dict] = []
    for region in bracket.get("regions", []):
        for matchup in region.get("round_of_64", []):
            for entry in (matchup.get("team1", {}), matchup.get("team2", {})):
                if entry.get("options"):
                    entries.extend(entry.get("options", []))
                else:
                    entries.append(entry)
            entries.extend(matchup.get("team1", {}).get("options", []))
            entries.extend(matchup.get("team2", {}).get("options", []))
    return entries


def resolve_team_ids(entries: list[dict], lookup: dict[str, int]) -> set[int]:
    team_ids: set[int] = set()
    missing: list[str] = []
    keys = list(lookup.keys())
    for entry in entries:
        candidates = [entry.get("team_key"), entry.get("display_name")]
        resolved = False
        for candidate in candidates:
            norm = _normalize(candidate) if candidate else ""
            norm = ALIASES.get(norm, norm)
            if norm and norm in lookup:
                team_ids.add(lookup[norm])
                resolved = True
                break
            if norm:
                close = difflib.get_close_matches(norm, keys, n=1, cutoff=0.85)
                if close:
                    team_ids.add(lookup[close[0]])
                    resolved = True
                    break
        if not resolved and candidates and candidates[0]:
            missing.append(candidates[0])
    if missing:
        raise SystemExit(f"Unable to resolve TeamIDs for: {sorted(set(missing))}")
    return team_ids


def filter_predictions(input_path: Path, output_path: Path, team_ids: set[int]) -> None:
    df = pd.read_csv(input_path)
    mask = df["Team1ID"].isin(team_ids) & df["Team2ID"].isin(team_ids)
    filtered = df.loc[mask].copy()
    filtered.to_csv(output_path, index=False)
    print(f"Wrote {len(filtered)} rows to {output_path}")


def main() -> None:
    lookup = build_team_lookup()
    bracket = json.loads(BRACKET_PATH.read_text())
    entries = extract_team_entries(bracket)
    team_ids = resolve_team_ids(entries, lookup)
    print(f"Resolved {len(team_ids)} teams from official bracket.")
    filter_predictions(BASELINE_INPUT, BASELINE_OUTPUT, team_ids)
    filter_predictions(CBBPY_INPUT, CBBPY_OUTPUT, team_ids)


if __name__ == "__main__":
    main()
