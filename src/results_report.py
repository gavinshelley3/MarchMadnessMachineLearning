"""Results reporting utilities for presentation-ready artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import get_config


@dataclass
class ReportInputs:
    final_model_summary: Optional[dict]
    final_model_comparison: Optional[pd.DataFrame]
    backtest_summary: Optional[pd.DataFrame]
    simulation_summary: Optional[dict]
    simulation_probs: Optional[pd.DataFrame]
    pick_confidence: Optional[pd.DataFrame]
    upset_report: Optional[pd.DataFrame]
    missing: List[str]


class ResultsReportGenerator:
    """Transforms saved project artifacts into polished summary files."""

    def __init__(
        self,
        reports_dir: Optional[Path] = None,
        predictions_dir: Optional[Path] = None,
        brackets_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        config = get_config()
        base_outputs = config.paths.outputs_dir
        self.reports_dir = Path(reports_dir or (base_outputs / "reports"))
        self.predictions_dir = Path(predictions_dir or (base_outputs / "predictions"))
        self.brackets_dir = Path(brackets_dir or (base_outputs / "brackets"))
        self.output_dir = Path(output_dir or (base_outputs / "final_report"))
        self.charts_dir = self.output_dir / "charts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Path]:
        inputs = self._collect_inputs()
        summary = self._build_summary(inputs)
        chart_paths = self._generate_charts(inputs, summary)
        files = self._write_reports(summary, chart_paths, inputs)
        self._print_console_summary(summary, files)
        return files

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def _collect_inputs(self) -> ReportInputs:
        missing: List[str] = []
        final_model_summary = self._read_json(self.reports_dir / "final_model_summary.json", "final_model_summary.json", missing)
        final_model_comparison = self._read_csv(self.reports_dir / "final_model_comparison.csv", "final_model_comparison.csv", missing)
        backtest_summary = self._read_csv(self.reports_dir / "backtest_summary.csv", "backtest_summary.csv", missing)

        simulation_summary = self._read_from_glob(
            pattern="*_simulation_summary.json",
            reader=self._read_json,
            description="simulation_summary",
            missing_list=missing,
            directory=self.brackets_dir,
        )
        simulation_probs = self._read_from_glob(
            pattern="*_simulation_team_probabilities.csv",
            reader=self._read_csv,
            description="simulation_team_probabilities",
            missing_list=missing,
            directory=self.brackets_dir,
        )
        pick_confidence = self._read_from_glob(
            pattern="*_pick_confidence.csv",
            reader=self._read_csv,
            description="pick_confidence",
            missing_list=missing,
            directory=self.brackets_dir,
        )
        upset_report = self._read_from_glob(
            pattern="*_upset_risk_report.csv",
            reader=self._read_csv,
            description="upset_risk_report",
            missing_list=missing,
            directory=self.brackets_dir,
        )

        return ReportInputs(
            final_model_summary=final_model_summary,
            final_model_comparison=final_model_comparison,
            backtest_summary=backtest_summary,
            simulation_summary=simulation_summary,
            simulation_probs=simulation_probs,
            pick_confidence=pick_confidence,
            upset_report=upset_report,
            missing=missing,
        )

    def _read_json(self, path: Path, label: str, missing_list: List[str]) -> Optional[dict]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            missing_list.append(label)
            return None

    def _read_csv(self, path: Path, label: str, missing_list: List[str]) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            missing_list.append(label)
            return None

    def _read_from_glob(
        self,
        pattern: str,
        reader,
        description: str,
        missing_list: List[str],
        directory: Path,
    ):
        matches = sorted(directory.glob(pattern))
        if not matches:
            missing_list.append(description)
            return None
        return reader(matches[-1], description, missing_list) if reader == self._read_csv or reader == self._read_json else reader(matches[-1])

    # ------------------------------------------------------------------ #
    # Summary assembly
    # ------------------------------------------------------------------ #

    def _build_summary(self, inputs: ReportInputs) -> Dict[str, object]:
        recommended = {}
        best_validation = {}
        best_backtest = {}
        model_desc = {}
        if inputs.final_model_summary:
            summary = inputs.final_model_summary
            recommended = summary.get("recommended_production_configuration", {})
            best_validation = summary.get("best_model_main_split", {})
            best_backtest = summary.get("best_feature_set_backtest_average", {})
            model_desc = summary.get("feature_sets", {})

        backtest_rows = []
        if inputs.final_model_comparison is not None and recommended:
            df = inputs.final_model_comparison.copy()
            df["split_rank"] = df["split_name"].apply(lambda x: 0 if x == "main_validation" else 1)
            mask = (
                (df["model_name"] == recommended.get("model_name"))
                & (df["feature_set"] == recommended.get("feature_set"))
                & (df["split_name"] != "main_validation")
            )
            filtered = df.loc[mask].sort_values("train_end_season")
            backtest_rows = filtered[
                ["split_name", "train_end_season", "validation_start_season", "log_loss", "accuracy"]
            ].to_dict(orient="records")

        simulation_info = self._extract_simulation_info(inputs)
        confidence_info = self._extract_confidence_info(inputs)
        upset_highlights = self._extract_upset_highlights(inputs)

        summary_payload = {
            "model": {
                "recommended_configuration": recommended,
                "best_validation": best_validation,
                "best_backtest": best_backtest,
                "feature_descriptions": model_desc,
            },
            "backtests": backtest_rows,
            "simulation": simulation_info,
            "confidence": confidence_info,
            "upset_alerts": upset_highlights,
            "missing_inputs": inputs.missing,
        }
        return summary_payload

    def _extract_simulation_info(self, inputs: ReportInputs) -> Dict[str, object]:
        simulation_summary = inputs.simulation_summary or {}
        metadata = simulation_summary.get("metadata", {})
        simulations = simulation_summary.get("simulations", {})
        most_likely = simulation_summary.get("most_likely_champion", {})
        top_contenders = simulation_summary.get("top_championship_contenders", [])
        championship_matchup = simulation_summary.get("most_common_championship_matchup", {})
        most_common_final_four = simulation_summary.get("most_common_final_four", {})

        if not top_contenders and inputs.simulation_probs is not None:
            df = inputs.simulation_probs.nlargest(5, "champion_probability")
            top_contenders = df[["team", "champion_probability"]].to_dict(orient="records")
            if not most_likely and not df.empty:
                most_likely = df.iloc[0].to_dict()

        return {
            "metadata": metadata,
            "simulations": simulations,
            "most_likely_champion": most_likely,
            "top_championship_contenders": top_contenders,
            "most_common_championship_matchup": championship_matchup,
            "most_common_final_four": most_common_final_four,
        }

    def _extract_confidence_info(self, inputs: ReportInputs) -> Dict[str, object]:
        if inputs.pick_confidence is None:
            return {}
        df = inputs.pick_confidence.copy()
        counts = df["confidence_tier"].value_counts().to_dict()
        low_conf = (
            df.sort_values("winner_probability")
            .head(5)[
                [
                    "round",
                    "region",
                    "slot",
                    "team1",
                    "team2",
                    "deterministic_winner",
                    "winner_probability",
                    "confidence_tier",
                ]
            ]
            .to_dict(orient="records")
        )
        return {"counts": counts, "lowest_confidence_games": low_conf}

    def _extract_upset_highlights(self, inputs: ReportInputs) -> List[Dict[str, object]]:
        if inputs.upset_report is None:
            return []
        df = inputs.upset_report.sort_values("probability", ascending=False).head(5)
        return df.to_dict(orient="records")

    # ------------------------------------------------------------------ #
    # Charts
    # ------------------------------------------------------------------ #

    def _generate_charts(
        self,
        inputs: ReportInputs,
        summary: Dict[str, object],
    ) -> Dict[str, Optional[Path]]:
        chart_paths: Dict[str, Optional[Path]] = {}
        model_info = summary["model"]["recommended_configuration"]
        final_model_comparison = inputs.final_model_comparison

        if final_model_comparison is not None:
            main_df = final_model_comparison[final_model_comparison["split_name"] == "main_validation"].copy()
            if not main_df.empty:
                path = self.charts_dir / "model_comparison_log_loss.png"
                self._plot_model_comparison(main_df, path)
                chart_paths["model_comparison"] = path
            if model_info:
                mask = (
                    (final_model_comparison["model_name"] == model_info.get("model_name"))
                    & (final_model_comparison["feature_set"] == model_info.get("feature_set"))
                    & (final_model_comparison["split_name"] != "main_validation")
                )
                subset = final_model_comparison.loc[mask]
                if not subset.empty:
                    path = self.charts_dir / "backtest_log_loss.png"
                    self._plot_backtests(subset, path)
                    chart_paths["backtests"] = path

        if inputs.simulation_probs is not None:
            champ_path = self.charts_dir / "championship_probabilities.png"
            final_four_path = self.charts_dir / "final_four_probabilities.png"
            self._plot_probabilities(inputs.simulation_probs, "champion_probability", "Championship Probability", champ_path)
            self._plot_probabilities(inputs.simulation_probs, "final_four_probability", "Final Four Probability", final_four_path)
            chart_paths["championship_probabilities"] = champ_path
            chart_paths["final_four_probabilities"] = final_four_path

        if summary["confidence"]:
            path = self.charts_dir / "pick_confidence_counts.png"
            self._plot_confidence(summary["confidence"]["counts"], path)
            chart_paths["confidence_counts"] = path

        return chart_paths

    def _plot_model_comparison(self, df: pd.DataFrame, path: Path) -> None:
        df = df.copy()
        df["label"] = df["model_name"] + " / " + df["feature_set"]
        df.sort_values("log_loss", inplace=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(df["label"], df["log_loss"], color="#1f77b4")
        ax.set_xlabel("Log Loss (lower is better)")
        ax.set_title("Main Validation Model Comparison")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_backtests(self, df: pd.DataFrame, path: Path) -> None:
        df = df.sort_values("train_end_season")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["train_end_season"], df["log_loss"], marker="o")
        ax.set_xlabel("Train End Season")
        ax.set_ylabel("Log Loss")
        ax.set_title("Rolling Backtest Performance (Log Loss)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_probabilities(self, df: pd.DataFrame, column: str, title: str, path: Path, top_n: int = 8) -> None:
        if column not in df.columns:
            path.unlink(missing_ok=True)
            return
        subset = df.nlargest(top_n, column)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(subset["team"], subset[column], color="#2ca02c")
        ax.set_xlabel("Probability")
        ax.set_title(title)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_confidence(self, counts: Dict[str, int], path: Path) -> None:
        tiers = ["high", "medium", "low"]
        values = [counts.get(tier, 0) for tier in tiers]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(tiers, values, color=["#1f77b4", "#ff7f0e", "#d62728"])
        ax.set_ylabel("Number of Games")
        ax.set_title("Deterministic Pick Confidence Tiers")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def _write_reports(
        self,
        summary: Dict[str, object],
        charts: Dict[str, Optional[Path]],
        inputs: ReportInputs,
    ) -> Dict[str, Path]:
        summary_json_path = self.output_dir / "project_summary.json"
        with summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        markdown_path = self.output_dir / "project_summary.md"
        markdown_content = self._build_markdown(summary, charts)
        markdown_path.write_text(markdown_content, encoding="utf-8")

        html_path = self.output_dir / "project_summary.html"
        html_path.write_text(self._build_html(summary, charts), encoding="utf-8")

        bracket_md_path = self.output_dir / "bracket_highlights.md"
        bracket_md_path.write_text(self._build_bracket_highlights(summary), encoding="utf-8")

        bracket_html_path = self.output_dir / "bracket_highlights.html"
        bracket_html_path.write_text(self._build_bracket_html(summary), encoding="utf-8")

        return {
            "summary_json": summary_json_path,
            "summary_md": markdown_path,
            "summary_html": html_path,
            "bracket_md": bracket_md_path,
            "bracket_html": bracket_html_path,
        }

    def _build_markdown(self, summary: Dict[str, object], charts: Dict[str, Optional[Path]]) -> str:
        model = summary["model"]["recommended_configuration"] or {}
        best_validation = summary["model"]["best_validation"] or {}
        sim = summary["simulation"] or {}
        top_contenders = sim.get("top_championship_contenders", [])
        confidence = summary["confidence"] or {}
        low_conf = confidence.get("lowest_confidence_games", [])
        upset_alerts = summary["upset_alerts"] or []
        missing = summary.get("missing_inputs", [])

        lines = [
            "# March Madness Project Summary",
            "## Project Overview",
            f"- **Production model**: {model.get('model_name', 'N/A')} ({model.get('feature_set', 'N/A')})",
            f"- **Validation Log Loss**: {best_validation.get('log_loss', 'N/A')}",
            f"- **Validation Accuracy**: {best_validation.get('accuracy', 'N/A')}",
            "",
        ]

        lines.extend(
            [
                "## Model Comparison",
                "Latest validation results across candidate models.",
            ]
        )
        if charts.get("model_comparison"):
            rel = self._relative_chart_path(charts["model_comparison"])
            lines.append(f"![Model Comparison]({rel})")
        else:
            lines.append("_Model comparison data unavailable._")

        lines.append("")
        lines.append("## Backtest Performance")
        if charts.get("backtests"):
            rel = self._relative_chart_path(charts["backtests"])
            lines.append(f"![Backtest Log Loss]({rel})")
        else:
            lines.append("_Backtest chart unavailable._")

        lines.append("")
        lines.append("## Simulation Highlights")
        if sim.get("most_likely_champion"):
            lines.append(
                f"- **Most likely champion**: {sim['most_likely_champion'].get('team', sim['most_likely_champion'].get('team_key', 'N/A'))}"
                f" ({sim['most_likely_champion'].get('champion_probability', 'N/A')})"
            )
        lines.append("- **Top Contenders:**")
        if top_contenders:
            for contender in top_contenders:
                lines.append(
                    f"  - {contender.get('team', contender.get('team_key', 'Team'))}: "
                    f"{contender.get('champion_probability', contender.get('probability', 'N/A')):.3f}"
                    if isinstance(contender.get("champion_probability", contender.get("probability")), float)
                    else f"  - {contender.get('team', contender.get('team_key', 'Team'))}"
                )
        else:
            lines.append("  - _No simulation data available._")
        if charts.get("championship_probabilities"):
            rel = self._relative_chart_path(charts["championship_probabilities"])
            lines.append(f"![Championship Probabilities]({rel})")
        if charts.get("final_four_probabilities"):
            rel = self._relative_chart_path(charts["final_four_probabilities"])
            lines.append(f"![Final Four Probabilities]({rel})")

        lines.append("")
        lines.append("## Deterministic Pick Confidence")
        if confidence:
            counts = confidence.get("counts", {})
            lines.append(f"- Counts: {counts}")
            if charts.get("confidence_counts"):
                rel = self._relative_chart_path(charts["confidence_counts"])
                lines.append(f"![Confidence Tiers]({rel})")
            if low_conf:
                lines.append("- Lowest-confidence games:")
                for game in low_conf:
                    lines.append(
                        f"  - {game['round']} {game['region']} {game['slot']}: {game['deterministic_winner']} "
                        f"(p={game['winner_probability']:.2f})"
                    )
        else:
            lines.append("_Confidence data unavailable._")

        lines.append("")
        lines.append("## Upset Alerts")
        if upset_alerts:
            for alert in upset_alerts:
                lines.append(
                    f"- {alert.get('team', 'Team')} ({alert.get('seed', '?')} seed, {alert.get('region', 'Region')}): "
                    f"{alert.get('note', '')} — {alert.get('probability', 'N/A'):.2f}"
                    if isinstance(alert.get("probability"), float)
                    else f"- {alert}"
                )
        else:
            lines.append("_No upset signals captured._")

        if missing:
            lines.append("")
            lines.append(f"_Missing optional inputs: {', '.join(missing)}_")

        return "\n".join(lines)

    def _build_html(self, summary: Dict[str, object], charts: Dict[str, Optional[Path]]) -> str:
        md = self._build_markdown(summary, charts)
        md_escaped = md.replace("\n", "<br/>\n")
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>March Madness Project Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f8f9fb; color: #222; }}
    h1, h2 {{ color: #0b3d91; }}
    img {{ max-width: 720px; display: block; margin: 1rem 0; border: 1px solid #ddd; padding: 8px; background: #fff; }}
    .card {{ background: #fff; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  </style>
</head>
<body>
  <div class="card">
    {md_escaped}
  </div>
</body>
</html>
"""

    def _build_bracket_highlights(self, summary: Dict[str, object]) -> str:
        sim = summary["simulation"] or {}
        confidence = summary["confidence"] or {}
        upset_alerts = summary["upset_alerts"] or []

        lines = ["# 2026 Bracket Highlights", ""]
        if sim.get("most_likely_champion"):
            lines.append(
                f"- **Most likely champion**: {sim['most_likely_champion'].get('team', sim['most_likely_champion'].get('team_key', 'N/A'))}"
            )
        if sim.get("most_common_final_four"):
            ff = sim["most_common_final_four"].get("teams", [])
            if ff:
                lines.append(f"- **Most common Final Four**: {', '.join(ff)}")
        if sim.get("most_common_championship_matchup"):
            matchup = sim["most_common_championship_matchup"].get("teams", [])
            if matchup:
                lines.append(f"- **Most common title game**: {matchup[0]} vs {matchup[1]}")
        lines.append("")
        if confidence.get("lowest_confidence_games"):
            lines.append("## Low-Confidence Picks")
            for game in confidence["lowest_confidence_games"]:
                lines.append(
                    f"- {game['round']} {game['region']} {game['slot']}: {game['deterministic_winner']} "
                    f"(p={game['winner_probability']:.2f})"
                )
            lines.append("")
        if upset_alerts:
            lines.append("## Upset Opportunities")
            for alert in upset_alerts:
                prob = alert.get("probability")
                prob_str = f"{prob:.2f}" if isinstance(prob, float) else str(prob)
                lines.append(
                    f"- {alert.get('team', 'Team')} ({alert.get('seed', '?')} seed, {alert.get('region', 'Region')}): "
                    f"{alert.get('note', '')} — {prob_str}"
                )
        return "\n".join(lines)

    def _build_bracket_html(self, summary: Dict[str, object]) -> str:
        md = self._build_bracket_highlights(summary).replace("\n", "<br/>\n")
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Bracket Highlights</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 2rem; background: #fdfdfd; color: #222; }}
    h1, h2 {{ color: #0b3d91; }}
  </style>
</head>
<body>
{md}
</body>
</html>
"""

    def _relative_chart_path(self, path: Optional[Path]) -> str:
        if not path:
            return ""
        return Path("charts") / path.name

    def _print_console_summary(self, summary: Dict[str, object], files: Dict[str, Path]) -> None:
        model = summary["model"]["recommended_configuration"] or {}
        best_validation = summary["model"]["best_validation"] or {}
        sim = summary["simulation"] or {}
        top_contenders = sim.get("top_championship_contenders", [])[:5]
        print("Final Results Report")
        print("---------------------")
        print(f"Production model: {model.get('model_name', 'N/A')} ({model.get('feature_set', 'N/A')})")
        if best_validation:
            print(
                f"Validation log loss: {best_validation.get('log_loss', 'N/A')}, "
                f"accuracy: {best_validation.get('accuracy', 'N/A')}"
            )
        if top_contenders:
            print("Top championship probabilities:")
            for contender in top_contenders:
                prob = contender.get("champion_probability", contender.get("probability"))
                prob_str = f"{prob:.3f}" if isinstance(prob, float) else str(prob)
                print(f" - {contender.get('team', contender.get('team_key', 'Team'))}: {prob_str}")
        print("Generated files:")
        for label, path in files.items():
            print(f" - {label}: {path}")
