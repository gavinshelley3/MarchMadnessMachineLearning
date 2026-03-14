
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_winner_map(bracket_payload: Dict[str, object]) -> Dict[Tuple[str, str], str]:
    winners: Dict[Tuple[str, str], str] = {}
    rounds = bracket_payload.get("rounds", {})
    for round_name, games in rounds.items():
        for game in games:
            slot = game.get("slot")
            winner = game.get("winner", {})
            winner_name = winner.get("display_name")
            if not slot or not winner_name:
                continue
            winners[(round_name, slot)] = str(winner_name)
    return winners


def _extract_final_four(bracket_payload: Dict[str, object]) -> List[str]:
    elite_round = bracket_payload.get("rounds", {}).get("Elite8", [])
    teams = [game.get("winner", {}).get("display_name") for game in elite_round]
    return sorted({name for name in teams if name})


def _extract_championship_matchup(bracket_payload: Dict[str, object]) -> List[str]:
    games = bracket_payload.get("rounds", {}).get("Championship", [])
    if not games:
        return []
    game = games[0]
    team1 = game.get("team1", {}).get("display_name")
    team2 = game.get("team2", {}).get("display_name")
    return [name for name in [team1, team2] if name]


def _probability_map(summary_payload: Dict[str, object]) -> Dict[str, float]:
    entries = summary_payload.get("top_championship_contenders", [])
    probs: Dict[str, float] = {}
    for entry in entries:
        team = entry.get("team")
        prob = entry.get("champion_probability")
        if team is None or prob is None:
            continue
        probs[str(team)] = float(prob)
    return probs


def _compare_probabilities(
    baseline: Dict[str, float],
    enriched: Dict[str, float],
) -> List[Dict[str, object]]:
    teams = sorted(set(baseline) | set(enriched))
    rows: List[Dict[str, object]] = []
    for team in teams:
        delta = enriched.get(team, 0.0) - baseline.get(team, 0.0)
        if delta == 0:
            continue
        rows.append(
            {
                "team": team,
                "baseline_probability": baseline.get(team, 0.0),
                "enriched_probability": enriched.get(team, 0.0),
                "delta": delta,
            }
        )
    rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
    return rows[:10]


def compare_brackets(
    baseline_bracket_path: Path,
    enriched_bracket_path: Path,
    baseline_summary_path: Path,
    enriched_summary_path: Path,
) -> Dict[str, object]:
    baseline_bracket = _load_json(baseline_bracket_path)
    enriched_bracket = _load_json(enriched_bracket_path)
    baseline_summary = _load_json(baseline_summary_path)
    enriched_summary = _load_json(enriched_summary_path)

    baseline_champion = baseline_bracket.get("champion", {}).get("display_name")
    enriched_champion = enriched_bracket.get("champion", {}).get("display_name")
    championship_changed = baseline_champion != enriched_champion

    baseline_final_four = _extract_final_four(baseline_bracket)
    enriched_final_four = _extract_final_four(enriched_bracket)
    final_four_changed = sorted(baseline_final_four) != sorted(enriched_final_four)

    baseline_matchup = _extract_championship_matchup(baseline_bracket)
    enriched_matchup = _extract_championship_matchup(enriched_bracket)
    matchup_changed = baseline_matchup != enriched_matchup

    baseline_winners = _extract_winner_map(baseline_bracket)
    enriched_winners = _extract_winner_map(enriched_bracket)
    all_slots = sorted(set(baseline_winners) | set(enriched_winners))
    slot_differences: List[Dict[str, object]] = []
    for round_name, slot in all_slots:
        base_pick = baseline_winners.get((round_name, slot))
        enriched_pick = enriched_winners.get((round_name, slot))
        if base_pick == enriched_pick:
            continue
        slot_differences.append(
            {
                "round": round_name,
                "slot": slot,
                "baseline": base_pick,
                "enriched": enriched_pick,
            }
        )

    baseline_probs = _probability_map(baseline_summary)
    enriched_probs = _probability_map(enriched_summary)
    probability_shifts = _compare_probabilities(baseline_probs, enriched_probs)

    most_likely_champion_changed = (
        baseline_summary.get("most_likely_champion", {}).get("team")
        != enriched_summary.get("most_likely_champion", {}).get("team")
    )

    return {
        "champion": {
            "baseline": baseline_champion,
            "enriched": enriched_champion,
            "changed": championship_changed,
        },
        "championship_matchup": {
            "baseline": baseline_matchup,
            "enriched": enriched_matchup,
            "changed": matchup_changed,
        },
        "final_four": {
            "baseline": baseline_final_four,
            "enriched": enriched_final_four,
            "changed": final_four_changed,
        },
        "deterministic_differences": slot_differences,
        "simulation": {
            "most_likely_champion_changed": most_likely_champion_changed,
            "top_champion_probability_shifts": probability_shifts,
        },
    }


def _write_markdown(report: Dict[str, object], path: Path) -> None:
    lines = [
        "# Baseline vs. CBBpy Bracket Comparison",
        "",
        f"- Champion changed: {'Yes' if report['champion']['changed'] else 'No'} "
        f"({report['champion']['baseline']} → {report['champion']['enriched']})",
        f"- Championship matchup changed: {'Yes' if report['championship_matchup']['changed'] else 'No'}",
        f"- Final Four changed: {'Yes' if report['final_four']['changed'] else 'No'}",
        "",
        "## Largest Champion Probability Shifts",
    ]
    for row in report["simulation"]["top_champion_probability_shifts"]:
        lines.append(
            f"- {row['team']}: {row['baseline_probability']:.3f} → "
            f"{row['enriched_probability']:.3f} (Δ {row['delta']:+.3f})"
        )
    lines.append("")
    lines.append("## Deterministic Pick Differences")
    if report["deterministic_differences"]:
        for diff in report["deterministic_differences"]:
            lines.append(
                f"- {diff['round']} / {diff['slot']}: "
                f"{diff['baseline']} → {diff['enriched']}"
            )
    else:
        lines.append("- None")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs. CBBpy bracket outputs.")
    parser.add_argument("--baseline-bracket", type=Path, required=True, help="Baseline *_bracket_results*.json file.")
    parser.add_argument("--enriched-bracket", type=Path, required=True, help="Enriched *_bracket_results*.json file.")
    parser.add_argument("--baseline-summary", type=Path, required=True, help="Baseline simulation summary JSON.")
    parser.add_argument("--enriched-summary", type=Path, required=True, help="Enriched simulation summary JSON.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/reports/cbbpy_bracket_comparison.json"),
        help="Destination for JSON comparison report.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional Markdown summary path.",
    )
    return parser.parse_args()


def main() -> Dict[str, object]:
    args = parse_args()
    report = compare_brackets(
        args.baseline_bracket,
        args.enriched_bracket,
        args.baseline_summary,
        args.enriched_summary,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote comparison JSON to {args.output_json}")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(report, args.output_md)
        print(f"Wrote Markdown summary to {args.output_md}")
    return report


if __name__ == "__main__":
    main()
