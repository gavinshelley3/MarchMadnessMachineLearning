from pathlib import Path

import pandas as pd
import pytest

from src.generate_advancement_probabilities_table import generate_advancement_probabilities_table


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_path_probabilities_reflect_simulation(tmp_path: Path) -> None:
    adv_df = pd.DataFrame(
        [
            {
                "TeamKey": "Alpha",
                "TeamName": "Alpha",
                "Region": "East",
                "Seed": 1,
                "calibrated_prob_round_of_32": 0.9,
                "calibrated_prob_sweet_16": 0.8,
                "calibrated_prob_elite_8": 0.7,
                "calibrated_prob_final_four": 0.6,
                "calibrated_prob_championship_game": 0.5,
                "calibrated_prob_champion": 0.4,
            },
            {
                "TeamKey": "Beta",
                "TeamName": "Beta",
                "Region": "East",
                "Seed": 1,
                "calibrated_prob_round_of_32": 0.3,
                "calibrated_prob_sweet_16": 0.25,
                "calibrated_prob_elite_8": 0.2,
                "calibrated_prob_final_four": 0.15,
                "calibrated_prob_championship_game": 0.1,
                "calibrated_prob_champion": 0.05,
            },
        ]
    )
    sim_df = pd.DataFrame(
        [
            {
                "team_key": "Alpha",
                "round_of_32_probability": 0.95,
                "sweet_16_probability": 0.85,
                "elite_8_probability": 0.75,
                "final_four_probability": 0.65,
                "championship_game_probability": 0.55,
                "champion_probability": 0.45,
            },
            {
                "team_key": "Beta",
                "round_of_32_probability": 0.5,
                "sweet_16_probability": 0.4,
                "elite_8_probability": 0.3,
                "final_four_probability": 0.2,
                "championship_game_probability": 0.1,
                "champion_probability": 0.05,
            },
        ]
    )

    adv_path = tmp_path / "adv.csv"
    sim_path = tmp_path / "sim.csv"
    html_path = tmp_path / "table.html"
    csv_output = tmp_path / "table.csv"
    _write_csv(adv_path, adv_df)
    _write_csv(sim_path, sim_df)

    generate_advancement_probabilities_table(
        adv_csv=adv_path,
        html_path=html_path,
        simulation_csv=sim_path,
        csv_output=csv_output,
        prior_weight=0.0,
    )

    result = pd.read_csv(csv_output)
    alpha_row = result.loc[result["TeamName"] == "Alpha"].iloc[0]
    beta_row = result.loc[result["TeamName"] == "Beta"].iloc[0]
    assert alpha_row["Path Round of 32"] == pytest.approx(0.95)
    assert beta_row["Path Round of 32"] == pytest.approx(0.5)
    assert alpha_row["Path Round of 32"] != beta_row["Path Round of 32"]


def test_fallback_to_prior_without_simulation(tmp_path: Path) -> None:
    adv_df = pd.DataFrame(
        [
            {
                "TeamKey": "Alpha",
                "TeamName": "Alpha",
                "Region": "East",
                "Seed": 1,
                "calibrated_prob_round_of_32": 0.42,
                "calibrated_prob_sweet_16": 0.32,
                "calibrated_prob_elite_8": 0.22,
                "calibrated_prob_final_four": 0.12,
                "calibrated_prob_championship_game": 0.08,
                "calibrated_prob_champion": 0.04,
            }
        ]
    )
    adv_path = tmp_path / "adv.csv"
    html_path = tmp_path / "table.html"
    csv_output = tmp_path / "table.csv"
    _write_csv(adv_path, adv_df)

    generate_advancement_probabilities_table(
        adv_csv=adv_path,
        html_path=html_path,
        simulation_csv=None,
        csv_output=csv_output,
        prior_weight=0.5,
    )

    result = pd.read_csv(csv_output)
    row = result.iloc[0]
    assert row["Path Round of 32"] == pytest.approx(0.42)
