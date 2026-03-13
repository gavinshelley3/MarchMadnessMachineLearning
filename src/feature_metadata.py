from __future__ import annotations

from typing import Dict, List

# Core stat columns captured before the advanced feature work.
CORE_FEATURE_COLUMNS: List[str] = [
    "GamesPlayed",
    "PointsForPerGame",
    "PointsAgainstPerGame",
    "ScoringMargin",
    "WinPercentage",
    "FieldGoalPercent",
    "ThreePointPercent",
    "FreeThrowPercent",
    "OffReboundsPerGame",
    "DefReboundsPerGame",
    "TotalReboundsPerGame",
    "AssistsPerGame",
    "TurnoversPerGame",
    "StealsPerGame",
    "BlocksPerGame",
    "FoulsPerGame",
    "AssistToTurnoverRatio",
    "SeedNum",
    "MasseyOrdinal",
]

ADVANCED_FEATURE_GROUPS: Dict[str, List[str]] = {
    "efficiency": [
        "EffectiveFGPercent",
        "TrueShootingPercent",
        "ThreePointRate",
        "FreeThrowRate",
    ],
    "possession_ratings": [
        "OffensiveRating",
        "DefensiveRating",
        "NetRating",
    ],
    "rebound_turnover": [
        "ReboundMargin",
        "OffensiveReboundRate",
        "TurnoverMargin",
        "AssistRate",
    ],
    "opponent_adjustment": [
        "AdjustedScoringMargin",
    ],
    "recent_form_5": [
        "Recent5WinPct",
        "Recent5ScoringMargin",
        "Recent5OffRating",
        "Recent5DefRating",
        "Recent5EffectiveFG",
    ],
    "recent_form_10": [
        "Recent10WinPct",
        "Recent10ScoringMargin",
        "Recent10OffRating",
        "Recent10DefRating",
        "Recent10EffectiveFG",
    ],
    "neutral_context": [
        "NeutralWinPercentage",
        "NeutralScoringMargin",
    ],
}

ADVANCED_FEATURE_COLUMNS: List[str] = [
    column for columns in ADVANCED_FEATURE_GROUPS.values() for column in columns
]

FEATURE_GROUPS: Dict[str, List[str]] = {"core": CORE_FEATURE_COLUMNS}
FEATURE_GROUPS.update({f"advanced_{name}": cols for name, cols in ADVANCED_FEATURE_GROUPS.items()})

BASE_FEATURE_SETS: Dict[str, List[str]] = {
    "core": ["core"],
    "advanced": list(FEATURE_GROUPS.keys()),
}

ABLATION_FEATURE_SETS: Dict[str, List[str]] = {
    "core_plus_efficiency": ["core", "advanced_efficiency"],
    "core_plus_possession_ratings": ["core", "advanced_possession_ratings"],
    "core_plus_recent_form": [
        "core",
        "advanced_recent_form_5",
        "advanced_recent_form_10",
    ],
    "core_plus_opponent_adjustment": ["core", "advanced_opponent_adjustment"],
    "core_plus_neutral_context": ["core", "advanced_neutral_context"],
    "core_plus_all_advanced": list(FEATURE_GROUPS.keys()),
}

FINAL_FEATURE_SETS: Dict[str, List[str]] = {
    "core": list(BASE_FEATURE_SETS["core"]),
    "core_plus_opponent_adjustment": list(ABLATION_FEATURE_SETS["core_plus_opponent_adjustment"]),
    "core_plus_efficiency": list(ABLATION_FEATURE_SETS["core_plus_efficiency"]),
}

FEATURE_SET_DEFINITIONS: Dict[str, List[str]] = {}
for mapping in (BASE_FEATURE_SETS, ABLATION_FEATURE_SETS, FINAL_FEATURE_SETS):
    FEATURE_SET_DEFINITIONS.update(mapping)

FEATURE_SET_DESCRIPTIONS: Dict[str, str] = {
    "core": "Seed/context features plus historical per-team season averages.",
    "advanced": "Core features plus every advanced feature family (efficiency, possession, recent form, opponent adjustments, and neutral context).",
    "core_plus_efficiency": "Core feature slate plus shooting efficiency ratios (eFG%, TS%, rates).",
    "core_plus_possession_ratings": "Core features plus offensive/defensive/net rating estimates.",
    "core_plus_recent_form": "Core features plus 5/10-game form windows.",
    "core_plus_opponent_adjustment": "Core features plus opponent-adjusted scoring margin.",
    "core_plus_neutral_context": "Core features plus neutral-site win rate and margin.",
    "core_plus_all_advanced": "Core features with the entire advanced feature bundle.",
}

FEATURE_TO_GROUP: Dict[str, str] = {}
for group_name, columns in FEATURE_GROUPS.items():
    for column in columns:
        FEATURE_TO_GROUP[column] = group_name


def available_feature_sets() -> List[str]:
    return list(FEATURE_SET_DEFINITIONS.keys())


def ablation_feature_sets() -> List[str]:
    return list(ABLATION_FEATURE_SETS.keys())


def final_feature_sets() -> List[str]:
    return list(FINAL_FEATURE_SETS.keys())


def feature_group_for_column(column: str) -> str:
    return FEATURE_TO_GROUP.get(column, "core")


def diff_column_base(column: str) -> str:
    return column.replace("Diff_", "", 1)


def select_diff_columns(diff_columns: List[str], feature_set: str) -> List[str]:
    if feature_set not in FEATURE_SET_DEFINITIONS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Options: {available_feature_sets()}")
    allowed_groups = set(FEATURE_SET_DEFINITIONS[feature_set])
    selected = [
        col
        for col in diff_columns
        if feature_group_for_column(diff_column_base(col)) in allowed_groups
    ]
    return selected


def describe_feature_sets() -> Dict[str, str]:
    return FEATURE_SET_DESCRIPTIONS.copy()
