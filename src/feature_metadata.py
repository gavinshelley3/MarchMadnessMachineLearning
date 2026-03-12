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
    "shooting_efficiency": [
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
    "rebound_turnover_margins": [
        "ReboundMargin",
        "OffensiveReboundRate",
        "TurnoverMargin",
        "AssistRate",
    ],
    "adjusted_margin": [
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
    "neutral_site": [
        "NeutralWinPercentage",
        "NeutralScoringMargin",
    ],
}

ADVANCED_FEATURE_COLUMNS: List[str] = [
    column for columns in ADVANCED_FEATURE_GROUPS.values() for column in columns
]

FEATURE_GROUPS: Dict[str, List[str]] = {"core": CORE_FEATURE_COLUMNS}
FEATURE_GROUPS.update({f"advanced_{name}": cols for name, cols in ADVANCED_FEATURE_GROUPS.items()})

FEATURE_SET_DEFINITIONS: Dict[str, List[str]] = {
    "core": ["core"],
    "advanced": list(FEATURE_GROUPS.keys()),
}

FEATURE_SET_DESCRIPTIONS: Dict[str, str] = {
    "core": "Seed/context features plus historical per-team season averages.",
    "advanced": "Core features plus efficiency, possession-based ratings, recent form, opponent adjustments, and neutral-site stats.",
}

FEATURE_TO_GROUP: Dict[str, str] = {}
for group_name, columns in FEATURE_GROUPS.items():
    for column in columns:
        FEATURE_TO_GROUP[column] = group_name


def available_feature_sets() -> List[str]:
    return list(FEATURE_SET_DEFINITIONS.keys())


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
