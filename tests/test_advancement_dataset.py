import pandas as pd

from src.advancement_dataset import build_advancement_dataset_from_frames


def test_play_in_labels_are_offset() -> None:
    seeds = pd.DataFrame(
        [
            {"Season": 2020, "Seed": "W01", "TeamID": 1},
            {"Season": 2020, "Seed": "W16a", "TeamID": 2},
            {"Season": 2020, "Seed": "W16b", "TeamID": 3},
        ]
    )
    context = pd.DataFrame(
        [
            {"Season": 2020, "TeamID": 1, "SeedNum": 1, "MetricA": 5.0},
            {"Season": 2020, "TeamID": 2, "SeedNum": 16, "MetricA": 1.0},
            {"Season": 2020, "TeamID": 3, "SeedNum": 16, "MetricA": 0.5},
        ]
    )
    teams = pd.DataFrame(
        [
            {"TeamID": 1, "TeamName": "Alpha"},
            {"TeamID": 2, "TeamName": "Bravo"},
            {"TeamID": 3, "TeamName": "Charlie"},
        ]
    )
    tourney = pd.DataFrame(
        [
            {"Season": 2020, "WTeamID": 2, "LTeamID": 3},
            {"Season": 2020, "WTeamID": 1, "LTeamID": 2},
            {"Season": 2020, "WTeamID": 1, "LTeamID": 4},
        ]
    )

    dataset, diagnostics = build_advancement_dataset_from_frames(
        seeds=seeds,
        tourney_results=tourney,
        context=context,
        teams=teams,
        min_season=2010,
    )

    assert diagnostics.total_rows == 3
    play_in_row = dataset.loc[dataset["TeamID"] == 2].iloc[0]
    assert play_in_row["Wins"] == 1
    assert play_in_row["reached_round_of_32"] == 0
    top_seed_row = dataset.loc[dataset["TeamID"] == 1].iloc[0]
    assert top_seed_row["reached_round_of_32"] == 1
    assert top_seed_row["reached_sweet_16"] == 1
