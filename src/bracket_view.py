"""Render a human-friendly HTML bracket view from deterministic results."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import get_config


ROUND_ORDER = ["R64", "R32", "Sweet16", "Elite8"]
NATIONAL_ROUNDS = ["FinalFour", "Championship"]
ROUND_LABELS = {
    "R64": "Round of 64",
    "R32": "Round of 32",
    "Sweet16": "Sweet 16",
    "Elite8": "Elite Eight",
    "FinalFour": "Final Four",
    "Championship": "Championship",
}


class BracketViewRenderer:
    """Creates a visual HTML bracket from stored deterministic results."""

    def __init__(
        self,
        bracket_results_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> None:
        config = get_config()
        self.outputs_dir = Path(config.paths.outputs_dir)
        self.brackets_dir = self.outputs_dir / "brackets"
        default_output = self.outputs_dir / "final_report" / "bracket_view.html"
        self.output_path = Path(output_path) if output_path else default_output
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.bracket_results_path = (
            Path(bracket_results_path) if bracket_results_path else self._find_latest_bracket_results()
        )

    def _find_latest_bracket_results(self) -> Path:
        candidates = sorted(self.brackets_dir.glob("*_bracket_results.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No bracket results found under {self.brackets_dir}. "
                "Run the deterministic bracket generator first."
            )
        return candidates[-1]

    def render(self) -> Path:
        data = self._load_json(self.bracket_results_path)
        html_text = self._build_html(data)
        self.output_path.write_text(html_text, encoding="utf-8")
        return self.output_path

    def _load_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    # ------------------------------------------------------------------ #
    # HTML assembly
    # ------------------------------------------------------------------ #

    def _build_html(self, data: dict) -> str:
        metadata = data.get("metadata", {})
        region_sections = "\n".join(
            section for region in self._region_list(data) if (section := self._render_region(data, region))
        )
        national_section = self._render_national(data)
        title = f"{metadata.get('season', '')} {metadata.get('bracket_type', 'Bracket').title()} Bracket"
        champion = data.get("champion", {}).get("display_name", "TBD")

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: "Inter", "Segoe UI", system-ui, sans-serif;
      margin: 1.5rem;
      background: #f7f9fc;
      color: #0b2a55;
    }}
    h1 {{
      margin-bottom: 0.2rem;
    }}
    .metadata {{
      margin-bottom: 1.5rem;
      color: #415a77;
    }}
    .regions {{
      display: flex;
      flex-wrap: wrap;
      gap: 1.5rem;
    }}
    .region {{
      flex: 1 1 480px;
      background: #ffffff;
      border-radius: 10px;
      padding: 1rem;
      box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
    }}
    .round-columns {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 0.75rem;
    }}
    .round {{
      border-left: 3px solid #d1d9e6;
      padding-left: 0.5rem;
    }}
    .round h3 {{
      font-size: 0.95rem;
      margin-top: 0;
    }}
    .game {{
      margin-bottom: 0.65rem;
      padding: 0.4rem 0.5rem;
      border-radius: 6px;
      background: #f1f5fb;
      border: 1px solid #dfe6f3;
    }}
    .team {{
      display: flex;
      justify-content: space-between;
      font-size: 0.9rem;
    }}
    .team.winner {{
      font-weight: 600;
      color: #0b4f8a;
    }}
    .prob {{
      color: #627597;
      font-size: 0.85rem;
      margin-left: 0.5rem;
    }}
    .game-note {{
      font-size: 0.75rem;
      margin-top: 0.2rem;
      color: #5c6f91;
    }}
    .national {{
      margin-top: 2rem;
      background: #ffffff;
      border-radius: 10px;
      padding: 1rem;
      box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
    }}
    .title-chip {{
      display: inline-block;
      padding: 0.2rem 0.7rem;
      background: #0b4f8a;
      color: #fff;
      border-radius: 999px;
      font-size: 0.85rem;
      margin-left: 0.6rem;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="metadata">
    Model: {html.escape(str(metadata.get("model_name", "unknown")))} ·
    Feature set: {html.escape(str(metadata.get("feature_set", "unknown")))} ·
    Champion: <strong>{html.escape(champion)}</strong>
  </div>
  <div class="regions">
    {region_sections}
  </div>
  {national_section}
</body>
</html>
"""

    def _region_list(self, data: dict) -> List[str]:
        metadata = data.get("metadata", {})
        if "regions" in metadata:
            return list(metadata["regions"].keys())
        # fallback: infer from rounds
        inferred = set(metadata.get("regions", {}).keys())
        for round_games in data.get("rounds", {}).values():
            for game in round_games:
                inferred.add(game.get("region"))
        return sorted(inferred)

    def _render_region(self, data: dict, region: str) -> str:
        rounds_html = []
        for round_name in ROUND_ORDER:
            games = self._collect_games(data, round_name, region)
            if not games:
                continue
            games_html = "\n".join(self._render_game(game) for game in games)
            rounds_html.append(
                f'<div class="round"><h3>{ROUND_LABELS[round_name]}</h3>{games_html}</div>'
            )
        if not rounds_html:
            return ""
        return f'<section class="region"><h2>{html.escape(region)} Region</h2><div class="round-columns">{"".join(rounds_html)}</div></section>'

    def _collect_games(self, data: dict, round_name: str, region: Optional[str] = None) -> List[dict]:
        games = data.get("rounds", {}).get(round_name, [])
        filtered = [g for g in games if region is None or g.get("region") == region]
        return sorted(filtered, key=lambda g: g.get("slot", ""))

    def _render_game(self, game: dict) -> str:
        winner_key = game.get("winner", {}).get("team_key")
        t1 = self._format_team_line(game.get("team1", {}), game.get("prob_team1"), winner_key)
        t2 = self._format_team_line(game.get("team2", {}), game.get("prob_team2"), winner_key)
        winner = game.get("winner", {}).get("display_name", "TBD")
        winner_prob = self._winner_probability(game)
        return f'<div class="game">{t1}{t2}<div class="game-note">Winner: {html.escape(winner)} ({winner_prob*100:.1f}%)</div></div>'

    def _format_team_line(self, team: dict, prob: Optional[float], winner_key: Optional[str]) -> str:
        name = html.escape(str(team.get("display_name", "TBD")))
        seed = team.get("seed")
        label = f"{seed} · {name}" if seed is not None else name
        prob_text = f"{float(prob)*100:.1f}%" if isinstance(prob, (int, float)) else "–"
        is_winner = winner_key is not None and team.get("team_key") == winner_key
        return f'<div class="team{" winner" if is_winner else ""}"><span>{label}</span><span class="prob">{prob_text}</span></div>'

    def _winner_probability(self, game: dict) -> float:
        probs = [p for p in (game.get("prob_team1"), game.get("prob_team2")) if isinstance(p, (int, float))]
        return max(probs) if probs else 0.0

    def _render_national(self, data: dict) -> str:
        sections = []
        for round_name in NATIONAL_ROUNDS:
            games = self._collect_games(data, round_name, None)
            if not games:
                continue
            games_html = "\n".join(self._render_game(game) for game in games)
            sections.append(f'<div class="round"><h3>{ROUND_LABELS[round_name]}</h3>{games_html}</div>')
        if not sections:
            return ""
        return f'<section class="national"><h2>Final Four & Championship<span class="title-chip">National</span></h2><div class="round-columns">{"".join(sections)}</div></section>'
