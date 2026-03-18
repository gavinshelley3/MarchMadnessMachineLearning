import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.advancement_inference import (
    MILESTONES,
    _build_seed_frame,
    apply_calibration,
    collect_field_entries,
    load_calibrators,
)
from src.bracket_loader import (
    BracketDefinition,
    FinalFourPairing,
    MatchupSlot,
    RegionDefinition,
    TeamSlot,
)


class _FakeLookup:
    def __init__(self, mapping: dict[str, int]) -> None:
        self.mapping = mapping

    def team_id(self, name: str) -> int | None:
        return self.mapping.get(name)


@pytest.fixture
def dummy_calibration_metadata(tmp_path):
    # Create dummy calibrators plus metadata for each milestone.
    import pickle
    from sklearn.isotonic import IsotonicRegression

    metadata = {"milestones": {}}
    calibrators = {}
    for milestone in MILESTONES:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit([0, 1], [0, 1])  # identity transformation
        pkl_path = tmp_path / f"{milestone}_isotonic.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(calibrator, f)
        metadata["milestones"][milestone] = {
            "calibrator_path": str(pkl_path),
            "method": "isotonic",
            "blend_applied": False,
            "blend_weight": None,
        }
        calibrators[milestone] = {
            "model": calibrator,
            "method": "isotonic",
            "blend_applied": False,
            "blend_weight": None,
        }
    metadata_path = tmp_path / "calibration_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path, metadata_path, calibrators


def test_load_calibrators(dummy_calibration_metadata, monkeypatch):
    tmp_path, metadata_path, _ = dummy_calibration_metadata
    import src.advancement_inference as ai

    monkeypatch.setattr(ai, "CALIBRATOR_DIR", tmp_path)
    calibrators, metadata = ai.load_calibrators(metadata_path, fallback_method="isotonic")
    assert metadata is not None
    assert set(calibrators.keys()) == set(MILESTONES)
    assert all(entry["method"] == "isotonic" for entry in calibrators.values())


def test_apply_calibration(dummy_calibration_metadata):
    _, _, calibrators = dummy_calibration_metadata
    probs = {milestone: 0.5 for milestone in MILESTONES}
    calibrated = apply_calibration(calibrators, probs)
    for value in calibrated.values():
        assert 0 <= value <= 1


def test_apply_calibration_blend_fallback():
    class _ZeroCalibrator:
        def transform(self, arr):
            return np.zeros_like(arr)

    calibrators = {
        "champion": {
            "model": _ZeroCalibrator(),
            "method": "isotonic",
            "blend_applied": True,
            "blend_weight": 0.5,
        }
    }
    for milestone in MILESTONES:
        calibrators.setdefault(
            milestone,
            {"model": _ZeroCalibrator(), "method": "isotonic", "blend_applied": False, "blend_weight": None},
        )
    raw_probs = {milestone: 0.6 for milestone in MILESTONES}
    blended = apply_calibration(calibrators, raw_probs)
    assert blended["champion"] == pytest.approx(0.3)


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
            options=[
                {"team_key": "Beta_11", "display_name": "Beta Eleven"},
                {"team_key": "Gamma", "display_name": "Gamma"},
            ],
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
