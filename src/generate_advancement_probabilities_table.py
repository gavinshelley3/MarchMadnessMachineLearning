"""Generate polished, bracket-ordered advancement probability tables."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:  # pragma: no cover - allow running as script or module
    from .bracket_loader import load_bracket_definition
except ImportError:  # pragma: no cover
    from bracket_loader import load_bracket_definition  # type: ignore

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
    return (1.0 - prior_weight) * float(sim_val) + prior_weight * float(adv_val)


def _build_bracket_index(bracket_path: Optional[Path]) -> Tuple[Dict[str, Tuple[int, int]], List[str]]:
    if bracket_path is None:
        return {}, []
    definition = load_bracket_definition(Path(bracket_path))
    slot_map: Dict[str, Tuple[int, int]] = {}
    for region_idx, region in enumerate(definition.regions):
        slot_idx = 0
        for matchup in region.round_of_64:
            for team_slot in (matchup.team1, matchup.team2):
                slot_map[team_slot.team_key] = (region_idx, slot_idx)
                slot_idx += 1
    region_sequence = [region.name for region in definition.regions]
    return slot_map, region_sequence


def _format_percent(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.1f}%"


def _prob_cell(value: Optional[float], label: str) -> str:
    pct = 0.0 if value is None or pd.isna(value) else max(0.0, min(1.0, float(value)))
    return (
        f'<td class="prob-cell" style="--pct:{pct:.4f}" title="{html.escape(label)}">'
        f"<span>{_format_percent(value)}</span>"
        "</td>"
    )


def _render_table_html(df: pd.DataFrame, html_path: Path, region_sequence: List[str]) -> Path:
    columns = ["Team", "Seed"] + [label for _, label in MILESTONES]
    header_html = "".join(f"<th>{html.escape(label)}</th>" for label in columns)
    rows: List[str] = []
    for region in region_sequence:
        chunk = df[df["Region"] == region]
        if chunk.empty:
            continue
        rows.append(
            f'<tr class="region-row"><td colspan="{len(columns)}">{html.escape(region)} Region</td></tr>'
        )
        for _, row in chunk.iterrows():
            team = html.escape(str(row["TeamName"]))
            seed = html.escape(str(row["Seed"]))
            badge = html.escape(str(row["Region"]))
            prob_cells = "".join(_prob_cell(row.get(f"path_{slug}"), title) for slug, title in MILESTONES)
            rows.append(
                "<tr class=\"team-row\">"
                f'<td class="team-cell"><span class="team-name">{team}</span>'
                f'<span class="region-pill">{badge}</span></td>'
                f'<td class="seed-cell">{seed}</td>'
                f"{prob_cells}"
                "</tr>"
            )
    css = """
    :root {{
      --surface: #f5f7fb;
      --card: #ffffff;
      --ink: #0b2244;
      --muted: #5e6c86;
      --accent: #0b63ce;
      --border: #dbe1ef;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: clamp(1rem, 2vw, 1.5rem);
      font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--surface);
      color: var(--ink);
    }}
    .table-shell {{
      max-width: 1180px;
      margin: 0 auto;
      background: radial-gradient(circle at top left, rgba(11, 99, 206, 0.08), transparent 45%) var(--surface);
      padding: clamp(1rem, 2.5vw, 1.75rem);
      border-radius: 28px;
      box-shadow: 0 24px 60px rgba(5, 26, 66, 0.12);
    }}
    .table-shell h1 {{
      font-size: clamp(1.6rem, 3vw, 2rem);
      margin: 0 0 0.35rem;
    }}
    .table-shell p {{
      margin: 0 0 1rem;
      color: var(--muted);
    }}
    .adv-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      background: var(--card);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 12px 32px rgba(5, 26, 66, 0.14);
      border: 1px solid var(--border);
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: #e8ecf8;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.72rem;
      padding: 0.75rem 0.6rem;
    }}
    tbody td {{
      padding: 0.45rem 0.55rem;
      border-bottom: 1px solid rgba(9, 31, 67, 0.08);
      font-size: 0.92rem;
    }}
    .team-cell {{
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .region-pill {{
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--muted);
    }}
    .seed-cell {{
      text-align: center;
      font-variant-numeric: tabular-nums;
      color: var(--muted);
      width: 56px;
    }}
    .team-row:nth-of-type(odd) td {{
      background: rgba(11, 34, 68, 0.015);
    }}
    .team-row td:first-child {{
      font-weight: 600;
    }}
    .team-row td:last-child {{
      font-weight: 600;
      color: var(--accent);
    }}
    .prob-cell {{
      position: relative;
      font-variant-numeric: tabular-nums;
      text-align: right;
      color: var(--ink);
      overflow: hidden;
    }}
    .prob-cell::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, rgba(11, 99, 206, 0.18) calc(var(--pct) * 100%), transparent 0);
      opacity: 0.65;
    }}
    .prob-cell span {{
      position: relative;
    }}
    .region-row td {{
      background: rgba(11, 34, 68, 0.06);
      font-size: 0.72rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      padding: 0.45rem 0.6rem;
    }}
    tbody tr:hover td {{
      background: rgba(11, 99, 206, 0.08);
    }}
    """
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Advancement Probabilities — Official Order</title>
  <style>
{css}
  </style>
</head>
<body>
  <div class="table-shell">
    <h1>Advancement Probabilities</h1>
    <p>Bracket-ordered view blending simulation paths with calibrated advancement priors.</p>
    <table class="adv-table">
      <thead><tr>{header_html}</tr></thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")
    return html_path


def generate_advancement_probabilities_table(
    adv_csv: Path,
    html_path: Path,
    simulation_csv: Optional[Path] = None,
    csv_output: Optional[Path] = None,
    prior_weight: float = 0.25,
    bracket_file: Optional[Path] = None,
) -> Path:
    if not 0.0 <= prior_weight <= 1.0:
        raise ValueError("--prior-weight must be between 0 and 1.")

    adv_df = _load_advancement_frame(adv_csv)
    table_df = adv_df[["TeamKey", "TeamName", "Region", "Seed"]].copy()

    for slug, _ in MILESTONES:
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

    for slug, _ in MILESTONES:
        table_df[f"path_{slug}"] = table_df.apply(
            lambda row, s=slug: _blend_prob(row.get(f"sim_{s}"), row.get(f"prior_{s}"), prior_weight),
            axis=1,
        )

    slot_map, region_sequence = _build_bracket_index(bracket_file)
    if not region_sequence:
        region_sequence = sorted(table_df["Region"].unique())
    region_rank = {region: idx for idx, region in enumerate(region_sequence)}
    table_df["__region_rank"] = table_df["Region"].map(lambda r: region_rank.get(r, len(region_rank)))
    table_df["__slot_rank"] = table_df.groupby("Region").cumcount()
    mapped = table_df["TeamKey"].map(slot_map)
    mask = mapped.notna()
    if mask.any():
        table_df.loc[mask, "__region_rank"] = mapped[mask].apply(lambda pair: pair[0])
        table_df.loc[mask, "__slot_rank"] = mapped[mask].apply(lambda pair: pair[1])

    ordered_df = table_df.sort_values(["__region_rank", "__slot_rank"]).reset_index(drop=True)
    display_df = ordered_df[
        ["TeamName", "Region", "Seed"] + [f"path_{slug}" for slug, _ in MILESTONES]
    ].copy()

    _render_table_html(display_df, html_path, region_sequence)

    if csv_output:
        rename_cols = {f"path_{slug}": title for slug, title in MILESTONES}
        csv_df = display_df.rename(columns=rename_cols)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        csv_df.to_csv(csv_output, index=False)

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
    parser.add_argument(
        "--bracket-file",
        type=Path,
        default=None,
        help="Bracket JSON used to derive official team order.",
    )
    args = parser.parse_args()
    html_path = generate_advancement_probabilities_table(
        adv_csv=args.csv,
        html_path=args.output,
        simulation_csv=args.simulation,
        csv_output=args.csv_output,
        prior_weight=args.prior_weight,
        bracket_file=args.bracket_file,
    )
    print(f"Wrote advancement probabilities table to {html_path}")


if __name__ == "__main__":
    main()
