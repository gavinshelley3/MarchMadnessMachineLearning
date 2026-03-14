import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.bracket_generation import BracketGenerator, TeamState
from src.bracket_loader import load_bracket_definition
from src.bracket_selection import (
    AdvancementInformedPolicy,
    AdvancementProbabilities,
    BlendedPredictionLookup,
    SimulationInformedPolicy,
    SimulationProbabilities,
    run_selection,
)


def _build_bracket(tmp_path: Path) -> Path:
    payload = {
        "season": 2026,
        "bracket_type": "test",
        "regions": [],
        "final_four_pairings": [
            {"regions": ["Alpha", "Beta"], "name": "FF1"},
            {"regions": ["Gamma", "Delta"], "name": "FF2"},
        ],
    }
    seeds = [1, 2, 3, 4]
    for region_idx, region_name in enumerate(["Alpha", "Beta", "Gamma", "Delta"], start=1):
        teams = [f"{region_name}_{i}" for i in range(1, 5)]
        payload["regions"].append(
            {
                "name": region_name,
                "round_of_64": [
                    {
                        "slot": f"{region_name}-1",
                        "round": "R64",
                        "team1": {"seed": seeds[0], "team_key": teams[0], "display_name": teams[0]},
                        "team2": {"seed": seeds[3], "team_key": teams[3], "display_name": teams[3]},
                    },
                    {
                        "slot": f"{region_name}-2",
                        "round": "R64",
                        "team1": {"seed": seeds[1], "team_key": teams[1], "display_name": teams[1]},
                        "team2": {"seed": seeds[2], "team_key": teams[2], "display_name": teams[2]},
                    },
                ],
            }
        )
    bracket_path = tmp_path / "bracket.json"
    bracket_path.write_text(json.dumps(payload), encoding="utf-8")
    return bracket_path


def _write_predictions_csv(path: Path, teams: list[str], bias: float) -> None:
    ratings = {team: idx + 1 for idx, team in enumerate(sorted(teams))}
    rows = []
    for team_a, team_b in combinations(sorted(teams), 2):
        rating_diff = bias * (ratings[team_a] - ratings[team_b])
        prob = max(0.05, min(0.95, 0.5 + 0.02 * rating_diff))
        rows.append(
            {
                "Season": 2026,
                "Team1Name": team_a,
                "Team2Name": team_b,
                "PredTeam1WinProb": prob,
                "PredTeam2WinProb": 1 - prob,
                "Team1ID": ratings[team_a],
                "Team2ID": ratings[team_b],
                "model_name": "test_model",
                "feature_set": "core",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_simulation_csv(path: Path, teams: list[str]) -> None:
    rows = []
    for idx, team in enumerate(teams):
        base = max(0.05, 0.35 - idx * 0.03)
        rows.append(
            {
                "team_key": team,
                "round_of_32_probability": max(0.1, min(0.9, 0.55 + base)),
                "sweet_16_probability": max(0.05, min(0.9, 0.45 + base)),
                "elite_8_probability": max(0.02, min(0.8, 0.35 + base)),
                "final_four_probability": max(0.01, min(0.7, 0.25 + base)),
                "championship_game_probability": max(0.005, min(0.6, 0.15 + base)),
                "champion_probability": max(0.001, min(0.5, 0.08 + base)),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_blended_prediction_lookup_weights_probabilities(tmp_path: Path) -> None:
    teams = ["A", "B", "C"]
    baseline_path = tmp_path / "baseline.csv"
    enriched_path = tmp_path / "enriched.csv"
    _write_predictions_csv(baseline_path, teams, bias=0.5)
    _write_predictions_csv(enriched_path, teams, bias=-0.5)
    baseline_lookup = BlendedPredictionLookup(baseline_path, baseline_path, season=2026, baseline_weight=1.0)
    enriched_lookup = BlendedPredictionLookup(enriched_path, enriched_path, season=2026, baseline_weight=1.0)
    lookup = BlendedPredictionLookup(baseline_path, enriched_path, season=2026, baseline_weight=0.75)
    p1, p2 = lookup.probability("A", "B")
    base_prob, _ = baseline_lookup.probability("A", "B")
    enr_prob, _ = enriched_lookup.probability("A", "B")
    assert abs(p1 + p2 - 1.0) < 1e-6
    assert min(base_prob, enr_prob) <= p1 <= max(base_prob, enr_prob)
    assert abs(p1 - base_prob) < abs(enr_prob - base_prob)


def test_simulation_informed_policy_prefers_simulation(tmp_path: Path) -> None:
    sim_path = tmp_path / "sim.csv"
    teams = ["Alpha", "Beta"]
    _write_simulation_csv(sim_path, teams)
    sim_probs = SimulationProbabilities(sim_path)
    policy = SimulationInformedPolicy(sim_probs, close_margin=0.5, stage_weight=0.8)
    team1 = TeamState(team_key="Alpha", display_name="Alpha", seed=5, region="East", source_slot="slot")
    team2 = TeamState(team_key="Beta", display_name="Beta", seed=2, region="East", source_slot="slot")
    winner = policy(team1, team2, 0.51, 0.49, "R64", "East-1", "East")
    assert winner is team1 or winner is team2
    assert winner is team1


def test_advancement_informed_policy_consults_advancement(tmp_path: Path) -> None:
    adv_path = tmp_path / "adv.csv"
    pd.DataFrame(
        [
            {
                "team_key": "Fav",
                "round_of_32_probability": 0.4,
                "sweet_16_probability": 0.3,
                "elite_8_probability": 0.2,
                "final_four_probability": 0.15,
                "championship_game_probability": 0.1,
                "champion_probability": 0.05,
            },
            {
                "team_key": "Dog",
                "round_of_32_probability": 0.8,
                "sweet_16_probability": 0.6,
                "elite_8_probability": 0.4,
                "final_four_probability": 0.3,
                "championship_game_probability": 0.2,
                "champion_probability": 0.1,
            },
        ]
    ).to_csv(adv_path, index=False)
    adv_lookup = AdvancementProbabilities(adv_path)
    policy = AdvancementInformedPolicy(
        advancement_probs=adv_lookup,
        simulation_probs=None,
        close_margin=0.5,
        simulation_weight=0.0,
        advancement_weight=0.4,
    )
    team1 = TeamState(team_key="Fav", display_name="Fav", seed=1, region="R", source_slot="")
    team2 = TeamState(team_key="Dog", display_name="Dog", seed=10, region="R", source_slot="")
    winner = policy(team1, team2, 0.52, 0.48, "R64", "slot", "R")
    assert winner is team2


def test_run_selection_outputs(tmp_path: Path) -> None:
    bracket_path = _build_bracket(tmp_path)
    bracket = load_bracket_definition(bracket_path)
    teams = []
    for region in bracket.regions:
        for matchup in region.round_of_64:
            teams.extend([matchup.team1.team_key, matchup.team2.team_key])
    baseline_path = tmp_path / "baseline.csv"
    enriched_path = tmp_path / "enriched.csv"
    _write_predictions_csv(baseline_path, teams, bias=0.4)
    _write_predictions_csv(enriched_path, teams, bias=-0.2)

    baseline_dir = tmp_path / "baseline_out"
    enriched_dir = tmp_path / "enriched_out"
    generator = BracketGenerator(baseline_path, output_dir=baseline_dir)
    baseline_outputs = generator.generate(bracket, season=2026, label_suffix="baseline")
    generator_enriched = BracketGenerator(enriched_path, output_dir=enriched_dir)
    enriched_outputs = generator_enriched.generate(bracket, season=2026, label_suffix="enriched")

    sim_path = tmp_path / "sim_probs.csv"
    _write_simulation_csv(sim_path, teams)

    result = run_selection(
        strategy="blended_probabilities",
        bracket_file=bracket_path,
        season=2026,
        baseline_predictions=baseline_path,
        enriched_predictions=enriched_path,
        baseline_weight=0.6,
        simulation_probabilities=None,
        advancement_probabilities=None,
        advancement_weight=0.3,
        label_suffix="final",
        output_dir=tmp_path / "final_brackets",
        comparison_json=tmp_path / "comparison.json",
        comparison_md=None,
        baseline_bracket_json=baseline_outputs["json"],
        enriched_bracket_json=enriched_outputs["json"],
        render_html=False,
        html_output=tmp_path / "final.html",
        close_margin=0.1,
        simulation_weight=0.5,
    )
    assert result["outputs"]["json"].exists()
    assert (tmp_path / "comparison.json").exists()

    adv_path = tmp_path / "adv_probs.csv"
    _write_simulation_csv(adv_path, teams)
    sim_path = tmp_path / "sim_probs.csv"
    _write_simulation_csv(sim_path, teams)
    adv_result = run_selection(
        strategy="advancement_informed",
        bracket_file=bracket_path,
        season=2026,
        baseline_predictions=baseline_path,
        enriched_predictions=enriched_path,
        baseline_weight=0.6,
        simulation_probabilities=sim_path,
        advancement_probabilities=adv_path,
        advancement_weight=0.25,
        label_suffix="adv",
        output_dir=tmp_path / "adv_brackets",
        comparison_json=tmp_path / "adv_comparison.json",
        comparison_md=None,
        baseline_bracket_json=baseline_outputs["json"],
        enriched_bracket_json=enriched_outputs["json"],
        render_html=False,
        html_output=tmp_path / "adv.html",
        close_margin=0.1,
        simulation_weight=0.5,
    )
    assert adv_result["outputs"]["json"].exists()
    assert (tmp_path / "adv_comparison.json").exists()
