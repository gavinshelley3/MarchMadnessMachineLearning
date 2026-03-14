from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def compute_team_probabilities(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rows.append({"TeamID": row["Team1ID"], "TeamName": row["Team1Name"], "win_prob": row[prob_col]})
        rows.append({"TeamID": row["Team2ID"], "TeamName": row["Team2Name"], "win_prob": 1.0 - row[prob_col]})
    team_df = pd.DataFrame(rows)
    grouped = team_df.groupby(["TeamID", "TeamName"]).win_prob.mean().reset_index()
    return grouped


def compare_predictions(baseline_path: Path, enriched_path: Path, output_path: Path) -> None:
    baseline = pd.read_csv(baseline_path)
    enriched = pd.read_csv(enriched_path)
    merged = baseline.merge(
        enriched,
        on=["MatchupID", "Season", "Team1ID", "Team2ID"],
        suffixes=("_baseline", "_enriched"),
    )
    merged["prob_shift"] = merged["PredTeam1WinProb_enriched"] - merged["PredTeam1WinProb_baseline"]
    merged["winner_changed"] = merged["PredictedWinnerID_baseline"] != merged["PredictedWinnerID_enriched"]
    winner_changes = int(merged["winner_changed"].sum())
    total_matchups = int(len(merged))

    top_changes = (
        merged.reindex(merged["prob_shift"].abs().sort_values(ascending=False).index)
        .head(20)
        .loc[:, ["MatchupID", "Team1Name_baseline", "Team2Name_baseline", "PredTeam1WinProb_baseline", "PredTeam1WinProb_enriched", "prob_shift", "winner_changed"]]
        .to_dict(orient="records")
    )

    baseline_team = compute_team_probabilities(baseline, "PredTeam1WinProb")
    enriched_team = compute_team_probabilities(enriched, "PredTeam1WinProb")
    team_merged = baseline_team.merge(
        enriched_team,
        on=["TeamID", "TeamName"],
        suffixes=("_baseline", "_enriched"),
        how="outer",
    ).fillna(0.0)
    team_merged["avg_prob_shift"] = team_merged["win_prob_enriched"] - team_merged["win_prob_baseline"]
    team_shifts = (
        team_merged.reindex(team_merged["avg_prob_shift"].abs().sort_values(ascending=False).index)
        .head(20)
        .to_dict(orient="records")
    )

    summary = {
        "baseline_file": str(baseline_path),
        "enriched_file": str(enriched_path),
        "total_matchups_compared": total_matchups,
        "winner_changes": winner_changes,
        "winner_change_rate": float(winner_changes / total_matchups) if total_matchups else 0.0,
        "top_matchup_shifts": top_changes,
        "top_team_average_shifts": team_shifts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two matchup prediction files.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline predictions CSV.")
    parser.add_argument("--enriched", type=Path, required=True, help="Enriched predictions CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/cbbpy_2026_inference_comparison.json"),
        help="Path to comparison JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_predictions(args.baseline, args.enriched, args.output)
    print(f"Comparison saved to {args.output}")


if __name__ == "__main__":
    main()
