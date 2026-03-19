import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


MILESTONES: List[Tuple[str, str]] = [
    ("round_of_32", "Round of 32"),
    ("sweet_16", "Sweet 16"),
    ("elite_8", "Elite 8"),
    ("final_four", "Final Four"),
    ("championship_game", "Championship"),
    ("champion", "Champion"),
]


def _load_advancement_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "TeamKey" not in df.columns and "TeamName" in df.columns:
        df["TeamKey"] = df["TeamName"]
    if "Region" not in df.columns:
        df["Region"] = "Unknown"
    required = {"TeamKey", "TeamName", "Region", "Seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Advancement CSV {csv_path} missing columns: {missing}")
    return df


def _load_simulation_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"team_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Simulation CSV {csv_path} missing columns: {missing}")
    return df


def _blend_prob(sim_val: Optional[float], adv_val: Optional[float], prior_weight: float) -> Optional[float]:
    if pd.isna(sim_val) and pd.isna(adv_val):
        return None
    if pd.isna(sim_val):
        return adv_val
    if pd.isna(adv_val):
        return sim_val
    return (1.0 - prior_weight) * sim_val + prior_weight * adv_val


def generate_advancement_probabilities_table(
    adv_csv: Path,
    html_path: Path,
    simulation_csv: Optional[Path] = None,
    csv_output: Optional[Path] = None,
    prior_weight: float = 0.25,
) -> Path:
    if not 0.0 <= prior_weight <= 1.0:
        raise ValueError("--prior-weight must be between 0 and 1.")

    adv_df = _load_advancement_frame(adv_csv)
    table_df = adv_df[["TeamKey", "TeamName", "Region", "Seed"]].copy()

    for slug, _title in MILESTONES:
        prior_col = f"calibrated_prob_{slug}"
        if prior_col not in adv_df.columns:
            raise ValueError(f"Column {prior_col} not found in {adv_csv}")
        table_df[f"prior_{slug}"] = adv_df[prior_col]

    if simulation_csv:
        sim_df = _load_simulation_frame(simulation_csv)
        rename_map: Dict[str, str] = {
            "team_key": "TeamKey",
            "round_of_32_probability": "sim_round_of_32",
            "sweet_16_probability": "sim_sweet_16",
            "elite_8_probability": "sim_elite_8",
            "final_four_probability": "sim_final_four",
            "championship_game_probability": "sim_championship_game",
            "champion_probability": "sim_champion",
        }
        missing_cols = [col for col in rename_map if col not in sim_df.columns]
        if missing_cols:
            raise ValueError(f"Simulation CSV {simulation_csv} missing columns: {missing_cols}")
        sim_df = sim_df.rename(columns=rename_map)
        table_df = table_df.merge(sim_df[list(rename_map.values())], on="TeamKey", how="left")
    else:
        for slug, _ in MILESTONES:
            table_df[f"sim_{slug}"] = pd.NA

    for slug, _title in MILESTONES:
        table_df[f"path_{slug}"] = table_df.apply(
            lambda row, s=slug: _blend_prob(row[f"sim_{s}"], row[f"prior_{s}"], prior_weight),
            axis=1,
        )

    display_cols = ["TeamName", "Region", "Seed"]
    renamed_cols: Dict[str, str] = {}
    for slug, title in MILESTONES:
        path_col = f"path_{slug}"
        prior_col = f"prior_{slug}"
        renamed_cols[path_col] = f"Path {title}"
        renamed_cols[prior_col] = f"calibrated_prob_{slug}"
        display_cols.extend([path_col, prior_col])

    output_df = table_df[display_cols].rename(columns=renamed_cols)
    output_df = output_df.sort_values(["Region", "Seed", "TeamName"]).reset_index(drop=True)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(output_df.to_html(index=False, float_format=lambda x: f"{x:.4f}"), encoding="utf-8")

    if csv_output:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(csv_output, index=False)

    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bracket-aware advancement probability tables.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to calibrated advancement probabilities CSV.")
    parser.add_argument(
        "--simulation",
        type=Path,
        default=None,
        help="Optional simulation team-probability CSV for path-aware blending.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination HTML file.")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV output containing the same table data.",
    )
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=0.25,
        help="Blend weight for the advancement NN prior (0 = pure simulation, 1 = pure prior).",
    )
    args = parser.parse_args()
    html_path = generate_advancement_probabilities_table(
        adv_csv=args.csv,
        html_path=args.output,
        simulation_csv=args.simulation,
        csv_output=args.csv_output,
        prior_weight=args.prior_weight,
    )
    print(f"Wrote advancement probabilities table to {html_path}")


if __name__ == "__main__":
    main()
