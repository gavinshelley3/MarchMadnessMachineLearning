import argparse
from pathlib import Path

import pandas as pd


def generate_advancement_probabilities_table(csv_path: Path, html_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    milestones = [
        "calibrated_prob_round_of_32",
        "calibrated_prob_sweet_16",
        "calibrated_prob_elite_8",
        "calibrated_prob_final_four",
        "calibrated_prob_championship_game",
        "calibrated_prob_champion",
    ]
    columns = ["TeamName", "Seed"] + milestones
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for advancement probabilities table: {missing}")
    table_html = df[columns].rename(columns={"TeamName": "Team"}).to_html(index=False)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(table_html, encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML table for calibrated advancement probabilities.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to calibrated advancement probabilities CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Destination HTML file.")
    args = parser.parse_args()
    html_path = generate_advancement_probabilities_table(args.csv, args.output)
    print(f"Wrote advancement probabilities table to {html_path}")


if __name__ == "__main__":
    main()
