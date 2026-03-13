"""Render a polished NCAA-style bracket HTML view."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .config import get_config

ROUND_ORDER = ["R64", "R32", "Sweet16", "Elite8"]
ROUND_LABELS = {
    "R64": "Round of 64",
    "R32": "Round of 32",
    "Sweet16": "Sweet 16",
    "Elite8": "Elite Eight",
    "FinalFour": "Final Four",
    "Championship": "Championship",
}
NATIONAL_ROUNDS = ["FinalFour", "Championship"]
DEFAULT_REGION_ORDER = ["East", "West", "South", "Midwest"]


class BracketViewRenderer:
    """Creates a styled HTML bracket from deterministic results."""

    def __init__(
        self,
        bracket_results_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> None:
        config = get_config()
        outputs_dir = Path(config.paths.outputs_dir)
        self.brackets_dir = outputs_dir / "brackets"
        default_output = outputs_dir / "final_report" / "bracket_view.html"
        self.output_path = Path(output_path) if output_path else default_output
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.bracket_results_path = (
            Path(bracket_results_path) if bracket_results_path else self._find_latest_bracket_results()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self) -> Path:
        data = self._load_json(self.bracket_results_path)
        html_text = self._build_html(data)
        self.output_path.write_text(html_text, encoding="utf-8")
        return self.output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_latest_bracket_results(self) -> Path:
        candidates = sorted(self.brackets_dir.glob("*_bracket_results.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No bracket results found under {self.brackets_dir}. "
                "Run the deterministic bracket generator first."
            )
        return candidates[-1]

    def _load_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    # ------------------------------------------------------------------
    # HTML assembly
    # ------------------------------------------------------------------

    def _build_html(self, data: dict) -> str:
        metadata = data.get("metadata", {})
        title = f"{metadata.get('season', 'Season')} NCAA Tournament Projection"
        regions_markup = self._render_regions(data)
        national_markup = self._render_final_four(data)

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --ink: #0b274a;
      --muted: #5c6f90;
      --border: #d7e0f1;
      --winner: #0b63ce;
      --winner-bg: #e7f1ff;
      --loser: #9aa9c0;
      --accent: linear-gradient(135deg, #1b91ff, #7c4dff);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 2rem clamp(1rem, 4vw, 3rem);
      font-family: "Inter", "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    .page-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      margin-bottom: 1rem;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.2em;
      font-size: 0.75rem;
      color: var(--muted);
      margin: 0;
    }}
    .page-title {{
      margin: 0.1rem 0 0;
      font-size: clamp(2rem, 4vw, 2.75rem);
    }}
    .meta-list {{
      display: flex;
      flex-direction: column;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1.5rem;
      font-size: 0.85rem;
      color: var(--muted);
    }}
    .legend .chip {{
      width: 16px;
      height: 16px;
      border-radius: 4px;
      display: inline-block;
      margin-right: 0.4rem;
      vertical-align: middle;
    }}
    .chip.winner {{ background: var(--winner); }}
    .chip.loser {{ background: #c9d3e4; }}
    .chip.prob {{ background: #ffd166; }}
    .regions {{
      display: flex;
      flex-direction: column;
      gap: 2rem;
    }}
    .region-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 1.5rem;
      justify-content: center;
    }}
    .bracket-region {{
      flex: 1 1 480px;
      background: var(--card);
      border-radius: 16px;
      padding: 1.25rem;
      box-shadow: 0 10px 30px rgba(15, 31, 62, 0.08);
      border: 1px solid var(--border);
    }}
    .region-header {{
      font-size: 1.15rem;
      font-weight: 600;
      margin-bottom: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .rounds-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 0.75rem;
    }}
    .round-column {{
      position: relative;
      padding-right: 0.25rem;
    }}
    .round-column::after {{
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      right: -0.35rem;
      border-right: 1px solid var(--border);
      opacity: 0.5;
    }}
    .round-column:last-child::after {{
      display: none;
    }}
    .round-column h3 {{
      margin: 0 0 0.4rem;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}
    .game-card {{
      background: #f6f8fd;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.55rem 0.75rem;
      margin-bottom: 0.7rem;
      position: relative;
      overflow: hidden;
    }}
    .game-card::after {{
      content: "";
      position: absolute;
      top: 50%;
      right: -0.35rem;
      width: 0.35rem;
      border-top: 1px solid #d4def2;
    }}
    .round-column:last-child .game-card::after {{
      display: none;
    }}
    .team-line {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.92rem;
      padding: 0.1rem 0;
    }}
    .team-line .seed {{
      width: 26px;
      height: 26px;
      border-radius: 50%;
      border: 1px solid var(--border);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 600;
      font-size: 0.85rem;
      color: var(--muted);
      background: #fff;
    }}
    .team-line .team-name {{
      font-weight: 500;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .team-line .prob {{
      font-size: 0.8rem;
      color: var(--muted);
    }}
    .team-line.winner {{
      color: var(--winner);
      font-weight: 600;
    }}
    .team-line.winner .seed {{
      border-color: var(--winner);
      color: var(--winner);
    }}
    .team-line.winner .prob {{
      color: var(--winner);
    }}
    .team-line.loser {{
      color: var(--loser);
    }}
    .game-meta {{
      font-size: 0.7rem;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0.15em;
      margin-top: 0.35rem;
    }}
    .national {{
      margin-top: 2.5rem;
      padding: 1.5rem;
      background: var(--card);
      border-radius: 20px;
      border: 1px solid var(--border);
      box-shadow: 0 12px 40px rgba(15, 31, 62, 0.12);
    }}
    .national h2 {{
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 1rem;
      color: var(--muted);
      margin: 0 0 1rem;
    }}
    .national-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.5rem;
      align-items: stretch;
    }}
    .champion-card {{
      border: 2px solid transparent;
      border-radius: 16px;
      padding: 1rem;
      position: relative;
      background: var(--card);
      box-shadow: inset 0 0 0 1px var(--border);
    }}
    .champion-card::before {{
      content: "";
      position: absolute;
      inset: 0;
      border-radius: 16px;
      padding: 1px;
      background: var(--accent);
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
    }}
    .champion-name {{
      font-size: 1.4rem;
      font-weight: 700;
      margin: 0.4rem 0 0;
    }}
    .champion-meta {{
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .champion-badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.3rem 0.7rem;
      border-radius: 999px;
      background: #0b63ce;
      color: #fff;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }}
    @media (max-width: 900px) {{
      .rounds-grid {{
        grid-template-columns: repeat(2, minmax(140px, 1fr));
      }}
      .round-column::after,
      .game-card::after {{
        display: none;
      }}
    }}
  </style>
</head>
<body>
  {self._build_header(metadata, data.get("champion"))}
  {self._build_legend()}
  <section class="regions">
    {regions_markup}
  </section>
  {national_markup}
</body>
</html>
"""

    def _build_header(self, metadata: dict, champion: Optional[dict]) -> str:
        model = metadata.get("model_name", "unknown")
        feature_set = metadata.get("feature_set", "unknown")
        bracket_type = metadata.get("bracket_type", "projected").title()
        season = metadata.get("season", "Season")
        champion_name = champion.get("display_name", "TBD") if champion else "TBD"
        return f"""
  <header class="page-header">
    <div>
      <p class="eyebrow">{html.escape(bracket_type)} Bracket · {html.escape(str(season))}</p>
      <h1 class="page-title">March Madness Bracket View</h1>
    </div>
    <div class="meta-list">
      <span>Model: {html.escape(str(model))}</span>
      <span>Feature set: {html.escape(str(feature_set))}</span>
      <span>Champion: {html.escape(champion_name)}</span>
    </div>
  </header>
  """

    def _build_legend(self) -> str:
        return """
  <div class="legend">
    <span><span class="chip winner"></span> Winner (highlighted)</span>
    <span><span class="chip loser"></span> Eliminated team</span>
    <span><span class="chip prob"></span> Probability = projected win chance</span>
  </div>
  """

    def _render_regions(self, data: dict) -> str:
        blocks: List[str] = []
        regions = self._region_list(data)
        for chunk in self._chunk_regions(regions, 2):
            row_blocks = [self._render_region(data, region) for region in chunk if region]
            if row_blocks:
                blocks.append(f'<div class="region-row">{"".join(row_blocks)}</div>')
        return "\n".join(blocks)

    def _render_region(self, data: dict, region: str) -> str:
        columns: List[str] = []
        for round_name in ROUND_ORDER:
            games = self._collect_games(data, round_name, region)
            games_html = "".join(self._render_game_card(game, round_name) for game in games) or "<div class=\"game-card placeholder\"></div>"
            columns.append(
                f'<div class="round-column"><h3>{html.escape(ROUND_LABELS.get(round_name, round_name))}</h3>{games_html}</div>'
            )
        return f'<section class="bracket-region" data-region="{html.escape(region)}"><div class="region-header">{html.escape(region)} Region</div><div class="rounds-grid">{"".join(columns)}</div></section>'

    def _collect_games(self, data: dict, round_name: str, region: Optional[str] = None) -> List[dict]:
        games = data.get("rounds", {}).get(round_name, [])
        filtered = [g for g in games if region is None or g.get("region") == region]
        return sorted(filtered, key=lambda g: g.get("slot", ""))

    def _render_game_card(self, game: dict, round_name: str) -> str:
        winner_key = game.get("winner", {}).get("team_key")
        team1 = self._team_line(game.get("team1", {}), game.get("prob_team1"), winner_key)
        team2 = self._team_line(game.get("team2", {}), game.get("prob_team2"), winner_key)
        slot = game.get("slot", "")
        winner_prob = self._winner_probability(game)
        return (
            f'<div class="game-card">'
            f"{team1}{team2}"
            f'<div class="game-meta">{html.escape(slot)} · {self._format_percent(winner_prob)} win</div>'
            f"</div>"
        )

    def _team_line(self, team: dict, prob: Optional[float], winner_key: Optional[str]) -> str:
        name = html.escape(str(team.get("display_name", "TBD")))
        seed = html.escape(str(team.get("seed", "—")))
        prob_text = self._format_percent(prob)
        is_winner = winner_key is not None and team.get("team_key") == winner_key
        classes = "team-line winner" if is_winner else "team-line loser"
        return f'<div class="{classes}"><span class="seed">{seed}</span><span class="team-name">{name}</span><span class="prob">{prob_text}</span></div>'

    def _winner_probability(self, game: dict) -> float:
        probs = [p for p in (game.get("prob_team1"), game.get("prob_team2")) if isinstance(p, (int, float))]
        return max(probs) if probs else 0.0

    def _format_percent(self, value: Optional[float]) -> str:
        return f"{float(value)*100:.1f}%" if isinstance(value, (int, float)) else "—"

    def _render_final_four(self, data: dict) -> str:
        semifinals = self._collect_games(data, "FinalFour")
        championship = self._collect_games(data, "Championship")
        if not semifinals and not championship:
            return ""
        semis_html = "".join(self._render_game_card(game, "FinalFour") for game in semifinals)
        champion_card = self._render_champion_card(championship[0]) if championship else ""
        return f"""
  <section class="national">
    <h2>Final Four & Championship</h2>
    <div class="national-grid">
      <div class="semis">{semis_html}</div>
      {champion_card}
    </div>
  </section>
  """

    def _render_champion_card(self, game: dict) -> str:
        winner = game.get("winner", {})
        champ_name = html.escape(winner.get("display_name", "TBD"))
        seed = winner.get("seed")
        region = winner.get("region", "")
        win_prob = self._winner_probability(game)
        seed_text = f"Seed {seed}" if seed is not None else "Seed TBD"
        region_text = f"{region} region" if region else ""
        return f"""
      <div class="champion-card">
        <span class="champion-badge">Champion</span>
        <p class="champion-name">{champ_name}</p>
        <p class="champion-meta">{seed_text} · {region_text}</p>
        <p class="champion-meta">Title probability: {self._format_percent(win_prob)}</p>
      </div>
    """

    def _region_list(self, data: dict) -> List[str]:
        metadata = data.get("metadata", {})
        if "regions" in metadata:
            available = list(metadata["regions"].keys())
        else:
            available = []
            for games in data.get("rounds", {}).values():
                for game in games:
                    region = game.get("region")
                    if region:
                        available.append(region)
            available = list(dict.fromkeys(available))
        ordered = [region for region in DEFAULT_REGION_ORDER if region in available]
        remaining = [region for region in available if region not in ordered]
        return ordered + remaining

    def _chunk_regions(self, regions: Sequence[str], size: int) -> Iterable[List[str]]:
        chunk: List[str] = []
        for region in regions:
            chunk.append(region)
            if len(chunk) == size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
