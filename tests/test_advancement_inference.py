from pathlib import Path

import pandas as pd

from src.advancement_inference import _build_seed_frame, collect_field_entries
from src.bracket_loader import BracketDefinition, FinalFourPairing, MatchupSlot, RegionDefinition, TeamSlot


class _FakeLookup:
    def __init__(self, mapping: dict[str, int]) -> None:
        self.mapping = mapping

    def team_id(self, name: str) -> int | None:
        return self.mapping.get(name)


def _build_bracket_definition() -> BracketDefinition:
    slot_a = MatchupSlot(
        slot="A1",
        round_name="R64",
        region="Alpha",
        team1=TeamSlot(team_key="Alpha_1", display_name="Alpha One", seed=1),
        team2=TeamSlot(team_key="Alpha_16", display_name="Alpha Sixteen", seed=16),
    )
    slot_b = MatchupSlot(
        slot="B1",
        round_name="R64",
        region="Beta",
        team1=TeamSlot(
            team_key="Beta_11",
            display_name="Beta Eleven",
            seed=11,
            options=[{"team_key": "Beta_11", "display_name": "Beta Eleven"}, {"team_key": "Gamma", "display_name": "Gamma"}],
        ),
        team2=TeamSlot(team_key="Beta_6", display_name="Beta Six", seed=6),
    )
    regions = [
        RegionDefinition(name="Alpha", locations=[], round_of_64=[slot_a]),
        RegionDefinition(name="Beta", locations=[], round_of_64=[slot_b]),
    ]
    final_four = [FinalFourPairing(regions=["Alpha", "Beta"], name="FF1")]
    return BracketDefinition(
        season=2026,
        bracket_type="test",
        source_note="unit",
        bracket_file=Path("dummy"),
        updated_at=None,
        regions=regions,
        final_four_pairings=final_four,
    )


def test_collect_field_entries_resolves_options() -> None:
    bracket = _build_bracket_definition()
    lookup = _FakeLookup({"Alpha One": 1, "Alpha Sixteen": 2, "Beta Eleven": 3, "Beta Six": 4})
    teams_df = pd.DataFrame(
        [
            {"TeamID": 1, "TeamName": "Alpha One"},
            {"TeamID": 2, "TeamName": "Alpha Sixteen"},
            {"TeamID": 3, "TeamName": "Beta Eleven"},
            {"TeamID": 4, "TeamName": "Beta Six"},
        ]
    )
    entries, missing = collect_field_entries(bracket, lookup, teams_df)
    assert missing == ["Gamma"]
    assert len(entries) == 4
    seeds = _build_seed_frame(entries)
    assert set(seeds["SeedNum"]) == {1, 6, 11, 16}
    assert all(seeds["Seed"].str.startswith(("A", "B")))
