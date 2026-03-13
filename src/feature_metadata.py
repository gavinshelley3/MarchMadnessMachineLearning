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

EFFICIENCY_FEATURES: List[str] = [
    "EffectiveFGPercent",
    "TrueShootingPercent",
    "ThreePointRate",
    "FreeThrowRate",
]

POSSESSION_RATING_FEATURES: List[str] = [
    "OffensiveRating",
    "DefensiveRating",
    "NetRating",
]

REBOUND_TURNOVER_FEATURES: List[str] = [
    "ReboundMargin",
    "OffensiveReboundRate",
    "TurnoverMargin",
    "AssistRate",
]

RECENT_FORM_FEATURES: List[str] = [
    "Recent5WinPct",
    "Recent5ScoringMargin",
    "Recent5OffRating",
    "Recent5DefRating",
    "Recent5EffectiveFG",
    "Recent10WinPct",
    "Recent10ScoringMargin",
    "Recent10OffRating",
    "Recent10DefRating",
    "Recent10EffectiveFG",
]

OPPONENT_ADJUSTMENT_FEATURES: List[str] = [
    "AdjustedScoringMargin",
]

NEUTRAL_CONTEXT_FEATURES: List[str] = [
    "NeutralWinPercentage",
    "NeutralScoringMargin",
]

SUPPLEMENTAL_NCAA_FEATURES: List[str] = [
    "SuppAdjTempo",
    "SuppAdjEfficiencyMargin",
    "SuppBarthag",
    "SuppWab",
    "SuppEffectiveFgDiff",
]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "core": CORE_FEATURE_COLUMNS,
    "efficiency": EFFICIENCY_FEATURES,
    "possession_ratings": POSSESSION_RATING_FEATURES,
    "rebound_turnover": REBOUND_TURNOVER_FEATURES,
    "recent_form": RECENT_FORM_FEATURES,
    "opponent_adjustment": OPPONENT_ADJUSTMENT_FEATURES,
    "neutral_context": NEUTRAL_CONTEXT_FEATURES,
    "supplemental_ncaa": SUPPLEMENTAL_NCAA_FEATURES,
}

ADVANCED_GROUP_NAMES: List[str] = [name for name in FEATURE_GROUPS if name != "core"]

ADVANCED_FEATURE_COLUMNS: List[str] = [
    column for name in ADVANCED_GROUP_NAMES for column in FEATURE_GROUPS[name]
]

def _all_groups_except_core() -> List[str]:
    return ["core"] + ADVANCED_GROUP_NAMES


FEATURE_SET_DEFINITIONS: Dict[str, List[str]] = {
    "core": ["core"],
    "core_plus_efficiency": ["core", "efficiency"],
    "core_plus_possession_ratings": ["core", "possession_ratings"],
    "core_plus_recent_form": ["core", "recent_form"],
    "core_plus_opponent_adjustment": ["core", "opponent_adjustment"],
    "core_plus_neutral_context": ["core", "neutral_context"],
    "core_plus_rebound_turnover": ["core", "rebound_turnover"],
    "core_plus_supplemental_ncaa": ["core", "supplemental_ncaa"],
    "core_plus_all_advanced": _all_groups_except_core(),
    "advanced": _all_groups_except_core(),
}

FEATURE_SET_DESCRIPTIONS: Dict[str, str] = {
    "core": "Seed/context features plus historical per-team season averages.",
    "core_plus_efficiency": "Core + shooting/true-shooting efficiency rates.",
    "core_plus_possession_ratings": "Core + offensive/defensive/net ratings derived from possessions.",
    "core_plus_recent_form": "Core + 5- and 10-game momentum metrics.",
    "core_plus_opponent_adjustment": "Core + opponent-adjusted scoring margin.",
    "core_plus_neutral_context": "Core + neutral-site win rate/margin.",
    "core_plus_rebound_turnover": "Core + rebound/turnover margin features.",
    "core_plus_supplemental_ncaa": "Core + NCAA Basketball supplemental tempo/efficiency metrics.",
    "core_plus_all_advanced": "Core + every advanced feature group currently available.",
    "advanced": "Core features plus the entire advanced bundle.",
}

ABLATION_DEFAULT_SETS: List[str] = [
    "core",
    "core_plus_efficiency",
    "core_plus_possession_ratings",
    "core_plus_recent_form",
    "core_plus_opponent_adjustment",
    "core_plus_neutral_context",
    "core_plus_all_advanced",
]

FEATURE_TO_GROUP: Dict[str, str] = {}
for group_name, columns in FEATURE_GROUPS.items():
    for column in columns:
        FEATURE_TO_GROUP[column] = group_name


def available_feature_sets() -> List[str]:
    return list(FEATURE_SET_DEFINITIONS.keys())


def default_ablation_feature_sets() -> List[str]:
    return [name for name in ABLATION_DEFAULT_SETS if name in FEATURE_SET_DEFINITIONS]


def default_comparison_feature_sets() -> List[str]:
    preferred = ["core", "advanced"]
    return [name for name in preferred if name in FEATURE_SET_DEFINITIONS]


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
