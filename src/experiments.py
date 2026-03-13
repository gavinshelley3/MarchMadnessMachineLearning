from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import get_config
from .data_pipeline import load_and_build_dataset, save_feature_summary_report
from .evaluate import compute_metrics
from .feature_metadata import (
    available_feature_sets,
    ablation_feature_sets,
    describe_feature_sets,
    final_feature_sets,
    select_diff_columns,
)
from .train import (
    prepare_features,
    run_gradient_boosting_baseline,
    run_logistic_regression_baseline,
    run_seed_baseline,
    time_based_split,
    train_with_arrays,
)
from .utils import ensure_parent_dir, set_random_seed

BACKTEST_SPLITS = [
    (2014, 2015),
    (2016, 2017),
    (2018, 2019),
    (2020, 2021),
]
ALL_FEATURE_SETS: Sequence[str] = tuple(available_feature_sets())
FEATURE_SET_OFFSETS = {name: idx * 10 for idx, name in enumerate(ALL_FEATURE_SETS)}
MODEL_SEED_OFFSETS = {
    "seed_baseline": 0,
    "logistic_regression": 1,
    "neural_net": 2,
    "gradient_boosting": 3,
}

MODE_MODEL_MAP: Dict[str, Sequence[str]] = {
    "comparison": ("seed_baseline", "logistic_regression", "neural_net"),
    "ablation": ("logistic_regression", "neural_net"),
    "final": ("logistic_regression", "neural_net", "gradient_boosting"),
}

def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


MODE_FEATURE_SET_DEFAULTS: Dict[str, List[str]] = {
    "comparison": ["core", "advanced"],
    "ablation": _dedupe_preserve_order(["core"] + ablation_feature_sets()),
    "final": _dedupe_preserve_order(final_feature_sets()),
}

MODE_OUTPUT_FILENAMES: Dict[str, tuple[str, str]] = {
    "comparison": ("model_comparison.csv", "model_comparison_summary.json"),
    "ablation": ("ablation_comparison.csv", "ablation_summary.json"),
    "final": ("final_model_comparison.csv", "final_model_summary.json"),
}


@dataclass
class SplitDefinition:
    name: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_end: int
    val_start: int


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_splits(
    dataset: pd.DataFrame,
    validation_start: int,
    include_backtests: bool = True,
) -> List[SplitDefinition]:
    splits: List[SplitDefinition] = []
    train_df, val_df = time_based_split(dataset, validation_start)
    splits.append(
        SplitDefinition(
            name="main_validation",
            train_df=train_df,
            val_df=val_df,
            train_end=int(train_df["Season"].max()),
            val_start=int(val_df["Season"].min()),
        )
    )
    if include_backtests:
        for train_end, val_start in BACKTEST_SPLITS:
            train_mask = dataset["Season"] <= train_end
            val_mask = dataset["Season"] >= val_start
            sub_train = dataset[train_mask].reset_index(drop=True)
            sub_val = dataset[val_mask].reset_index(drop=True)
            if sub_train.empty or sub_val.empty:
                continue
            splits.append(
                SplitDefinition(
                    name=f"backtest_train<= {train_end}_val>= {val_start}",
                    train_df=sub_train,
                    val_df=sub_val,
                    train_end=train_end,
                    val_start=val_start,
                )
            )
    return splits


def _seed_for_variant(base_seed: int, split_index: int, feature_set: str, model_name: str) -> int:
    feature_offset = FEATURE_SET_OFFSETS.get(feature_set, 0)
    model_offset = MODEL_SEED_OFFSETS.get(model_name, 0)
    return base_seed + split_index * 100 + feature_offset + model_offset


def _base_result_row(
    split: SplitDefinition, feature_set: str, model_name: str, feature_count: int
) -> Dict[str, object]:
    return {
        "split_name": split.name,
        "train_end_season": split.train_end,
        "validation_start_season": split.val_start,
        "feature_set": feature_set,
        "model_name": model_name,
        "feature_count": feature_count,
        "train_rows": len(split.train_df),
        "validation_rows": len(split.val_df),
    }


def _format_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "log_loss": float(metrics.get("log_loss", float("nan"))),
        "brier_score": float(metrics.get("brier_score", float("nan"))),
        "roc_auc": float(metrics.get("roc_auc", float("nan"))),
        "accuracy": float(metrics.get("accuracy", float("nan"))),
    }


def run_experiments(
    dataset: pd.DataFrame,
    diff_columns: Sequence[str],
    config,
    feature_sets: Sequence[str],
    include_backtests: bool,
    model_names: Sequence[str],
) -> pd.DataFrame:
    splits = build_splits(dataset, config.training.validation_start_season, include_backtests)
    results: List[Dict[str, object]] = []

    for split_index, split in enumerate(splits):
        for feature_set in feature_sets:
            selected_cols = select_diff_columns(diff_columns, feature_set)
            if not selected_cols:
                raise ValueError(f"No diff columns available for feature set '{feature_set}'.")
            X_train, y_train, X_val, y_val, feature_cols, _ = prepare_features(
                split.train_df, split.val_df, selected_cols
            )

            for model_name in model_names:
                feature_count = len(feature_cols)
                if model_name == "seed_baseline":
                    seed_probs = run_seed_baseline(split.val_df)
                    metrics, _ = compute_metrics(y_val, seed_probs)
                    feature_count = 1
                elif model_name == "logistic_regression":
                    seed_value = _seed_for_variant(
                        config.training.random_seed, split_index, feature_set, model_name
                    )
                    set_random_seed(seed_value)
                    probs = run_logistic_regression_baseline(X_train, y_train, X_val)
                    metrics, _ = compute_metrics(y_val, probs)
                elif model_name == "neural_net":
                    set_random_seed(
                        _seed_for_variant(
                            config.training.random_seed, split_index, feature_set, model_name
                        )
                    )
                    _, nn_metrics, _, _ = train_with_arrays(
                        X_train, y_train, X_val, y_val, config
                    )
                    metrics = nn_metrics
                elif model_name == "gradient_boosting":
                    seed_value = _seed_for_variant(
                        config.training.random_seed, split_index, feature_set, model_name
                    )
                    set_random_seed(seed_value)
                    probs = run_gradient_boosting_baseline(
                        X_train, y_train, X_val, random_state=seed_value
                    )
                    metrics, _ = compute_metrics(y_val, probs)
                else:
                    raise ValueError(f"Unsupported model '{model_name}'.")

                row = _base_result_row(split, feature_set, model_name, feature_count)
                row.update(_format_metrics(metrics))
                results.append(row)

    return pd.DataFrame(results)


def summarize_results(results: pd.DataFrame, mode: str) -> Dict[str, object]:
    if results.empty:
        return {"message": "No experiment results generated."}
    if mode == "ablation":
        return _summarize_ablation_results(results)
    if mode == "final":
        return _summarize_final_results(results)
    return _summarize_comparison_results(results)


def _base_summary_dict() -> Dict[str, object]:
    return {"feature_sets": describe_feature_sets()}


def _summarize_comparison_results(results: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = _base_summary_dict()
    main_df = results[results["split_name"] == "main_validation"]
    conclusions: List[str] = []
    if not main_df.empty:
        best = main_df.loc[main_df["log_loss"].idxmin()].to_dict()
        summary["best_validation_configuration"] = {
            "model": best["model_name"],
            "feature_set": best["feature_set"],
            "log_loss": best["log_loss"],
            "accuracy": best["accuracy"],
        }
        conclusions.append(
            f"Best validation config: {best['model_name']} with {best['feature_set']} features "
            f"(log_loss={best['log_loss']:.4f}, accuracy={best['accuracy']:.4f})."
        )

    backtest_df = results[results["split_name"].str.startswith("backtest")]
    if not backtest_df.empty:
        grouped = (
            backtest_df.groupby(["model_name", "feature_set"])["log_loss"]
            .mean()
            .reset_index()
        )
        if not grouped.empty:
            best_avg = grouped.loc[grouped["log_loss"].idxmin()]
            summary["best_average_backtest"] = {
                "model": best_avg["model_name"],
                "feature_set": best_avg["feature_set"],
                "avg_log_loss": best_avg["log_loss"],
            }

        improvements = {}
        for model_name in grouped["model_name"].unique():
            core_row = grouped[
                (grouped["model_name"] == model_name) & (grouped["feature_set"] == "core")
            ]
            adv_row = grouped[
                (grouped["model_name"] == model_name) & (grouped["feature_set"] == "advanced")
            ]
            if core_row.empty or adv_row.empty:
                continue
            core_value = float(core_row["log_loss"].iloc[0])
            adv_value = float(adv_row["log_loss"].iloc[0])
            improvements[model_name] = {
                "core_avg_log_loss": core_value,
                "advanced_avg_log_loss": adv_value,
                "log_loss_improvement": core_value - adv_value,
            }
        if improvements:
            summary["advanced_vs_core_backtests"] = improvements
            for model_name, stats in improvements.items():
                conclusions.append(
                    f"Advanced features changed average backtest log loss for {model_name} by "
                    f"{stats['log_loss_improvement']:.4f}."
                )

        advanced_bt = backtest_df[backtest_df["feature_set"] == "advanced"]
        if not advanced_bt.empty:
            wins = 0
            comparisons = 0
            for split_name in sorted(advanced_bt["split_name"].unique()):
                nn_row = advanced_bt[
                    (advanced_bt["split_name"] == split_name)
                    & (advanced_bt["model_name"] == "neural_net")
                ]
                log_row = advanced_bt[
                    (advanced_bt["split_name"] == split_name)
                    & (advanced_bt["model_name"] == "logistic_regression")
                ]
                if nn_row.empty or log_row.empty:
                    continue
                comparisons += 1
                if float(nn_row["log_loss"].iloc[0]) < float(log_row["log_loss"].iloc[0]):
                    wins += 1
            if comparisons:
                summary["neural_vs_logistic_backtests"] = {
                    "neural_wins": wins,
                    "total_comparisons": comparisons,
                }
                conclusions.append(
                    f"Neural net beat logistic regression on {wins} of {comparisons} advanced-feature backtest splits."
                )

    if conclusions:
        summary["conclusions"] = conclusions
    return summary


def _summarize_ablation_results(results: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = _base_summary_dict()
    main_df = results[results["split_name"] == "main_validation"]
    per_model_best: Dict[str, Dict[str, float]] = {}
    improvements: Dict[str, Dict[str, float]] = {}
    if not main_df.empty:
        for model_name, group in main_df.groupby("model_name"):
            best = group.loc[group["log_loss"].idxmin()]
            per_model_best[model_name] = {
                "feature_set": best["feature_set"],
                "log_loss": float(best["log_loss"]),
                "accuracy": float(best["accuracy"]),
            }
            core_row = group[group["feature_set"] == "core"]
            if core_row.empty:
                continue
            core_loss = float(core_row["log_loss"].iloc[0])
            improvements[model_name] = {}
            for _, row in group.iterrows():
                fs = row["feature_set"]
                if fs == "core":
                    continue
                improvements[model_name][fs] = core_loss - float(row["log_loss"])
    if per_model_best:
        summary["best_validation_per_model"] = per_model_best
    if improvements:
        summary["validation_improvements_vs_core"] = improvements

    backtest_df = results[results["split_name"].str.startswith("backtest")]
    if not backtest_df.empty:
        grouped = (
            backtest_df.groupby(["model_name", "feature_set"])["log_loss"]
            .mean()
            .reset_index()
        )
        comparisons: Dict[str, Dict[str, Dict[str, float]]] = {}
        for model_name in grouped["model_name"].unique():
            model_group = grouped[grouped["model_name"] == model_name]
            core_row = model_group[model_group["feature_set"] == "core"]
            if core_row.empty:
                continue
            core_loss = float(core_row["log_loss"].iloc[0])
            for _, row in model_group.iterrows():
                fs = row["feature_set"]
                if fs == "core":
                    continue
                comparisons.setdefault(model_name, {})[fs] = {
                    "avg_log_loss": float(row["log_loss"]),
                    "improvement_vs_core": core_loss - float(row["log_loss"]),
                }
        if comparisons:
            summary["backtest_improvements_vs_core"] = comparisons
    return summary


def _average_metrics(results: pd.DataFrame, group_col: str) -> List[Dict[str, object]]:
    grouped = (
        results.groupby(group_col)[["log_loss", "brier_score", "roc_auc", "accuracy"]]
        .mean()
        .reset_index()
    )
    return grouped.to_dict("records")


def _summarize_final_results(results: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = _base_summary_dict()
    main_df = results[results["split_name"] == "main_validation"]
    if not main_df.empty:
        best_row = main_df.loc[main_df["log_loss"].idxmin()]
        summary["best_model_main_split"] = {
            "model_name": best_row["model_name"],
            "feature_set": best_row["feature_set"],
            "log_loss": float(best_row["log_loss"]),
            "accuracy": float(best_row["accuracy"]),
        }
        summary["best_feature_set_main_split"] = best_row["feature_set"]

    backtest_df = results[results["split_name"].str.startswith("backtest")]
    if not backtest_df.empty:
        avg_model = (
            backtest_df.groupby("model_name")["log_loss"].mean().reset_index()
        )
        if not avg_model.empty:
            best_model = avg_model.loc[avg_model["log_loss"].idxmin()]
            summary["best_model_backtest_average"] = {
                "model_name": best_model["model_name"],
                "avg_log_loss": float(best_model["log_loss"]),
            }
        avg_feature = (
            backtest_df.groupby("feature_set")["log_loss"].mean().reset_index()
        )
        if not avg_feature.empty:
            best_feature = avg_feature.loc[avg_feature["log_loss"].idxmin()]
            summary["best_feature_set_backtest_average"] = {
                "feature_set": best_feature["feature_set"],
                "avg_log_loss": float(best_feature["log_loss"]),
            }

    summary["average_metrics_per_model"] = _average_metrics(results, "model_name")
    summary["average_metrics_per_feature_set"] = _average_metrics(results, "feature_set")

    ranking = (
        results.groupby(["model_name", "feature_set"])[["log_loss", "brier_score", "roc_auc", "accuracy"]]
        .mean()
        .reset_index()
        .sort_values("log_loss")
    )
    ranking_records = ranking.to_dict("records")
    summary["model_ranking"] = ranking_records
    if ranking_records:
        summary["recommended_production_configuration"] = ranking_records[0]
    return summary


def save_results(
    results: pd.DataFrame,
    summary: Dict[str, object],
    config,
    feature_sets: Sequence[str],
    include_backtests: bool,
    diff_columns: Sequence[str],
    mode: str,
    model_names: Sequence[str],
) -> None:
    reports_dir = config.paths.outputs_dir / "reports"
    comparison_file, summary_file = MODE_OUTPUT_FILENAMES.get(
        mode, MODE_OUTPUT_FILENAMES["comparison"]
    )
    ensure_parent_dir(reports_dir / comparison_file)

    ordered_cols = [
        "split_name",
        "train_end_season",
        "validation_start_season",
        "feature_set",
        "model_name",
        "feature_count",
        "train_rows",
        "validation_rows",
        "log_loss",
        "brier_score",
        "roc_auc",
        "accuracy",
    ]
    results = results.sort_values(
        ["split_name", "feature_set", "model_name"]
    ).reset_index(drop=True)
    results.to_csv(
        reports_dir / comparison_file,
        columns=ordered_cols,
        index=False,
    )

    summary_path = reports_dir / summary_file
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default))

    descriptions = describe_feature_sets()
    feature_set_summary = {}
    for name in feature_sets:
        try:
            count = len(select_diff_columns(list(diff_columns), name))
        except ValueError:
            count = 0
        feature_set_summary[name] = {
            "description": descriptions.get(name, ""),
            "diff_feature_count": count,
        }
    (reports_dir / "feature_set_summary.json").write_text(
        json.dumps(feature_set_summary, indent=2, default=_json_default)
    )

    experiment_config = {
        "mode": mode,
        "feature_sets": list(feature_sets),
        "models": list(model_names),
        "include_backtests": include_backtests,
        "validation_start_season": config.training.validation_start_season,
        "backtest_splits_attempted": BACKTEST_SPLITS if include_backtests else [],
    }
    (reports_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2, default=_json_default)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model comparison experiments.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing Kaggle CSVs (defaults to config.paths.data_dir).",
    )
    parser.add_argument(
        "--validation-start-season",
        type=int,
        default=None,
        help="First validation season (inclusive). Overrides config.",
    )
    parser.add_argument(
        "--skip-backtests",
        action="store_true",
        help="If provided, skip the rolling backtest comparisons.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=available_feature_sets(),
        help="Feature sets to compare. Defaults depend on --mode.",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_MODEL_MAP.keys()),
        default="comparison",
        help="Experiment mode: comparison, ablation, or final.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    if args.validation_start_season is not None:
        config.training.validation_start_season = args.validation_start_season

    dataset, diagnostics = load_and_build_dataset(args.data_dir)
    feature_summary_path = config.paths.outputs_dir / "reports" / "feature_summary.json"
    save_feature_summary_report(diagnostics, feature_summary_path)
    diff_columns = sorted([col for col in dataset.columns if col.startswith("Diff_")])

    mode = args.mode
    feature_sets = args.feature_sets or MODE_FEATURE_SET_DEFAULTS.get(
        mode, MODE_FEATURE_SET_DEFAULTS["comparison"]
    )
    model_names = MODE_MODEL_MAP.get(mode, MODE_MODEL_MAP["comparison"])
    include_backtests = not args.skip_backtests

    print(f"Experiment mode: {mode}")
    print(f"Running experiments for feature sets: {', '.join(feature_sets)}")
    if include_backtests:
        print("Rolling backtests: enabled")
    else:
        print("Rolling backtests: skipped")

    results = run_experiments(
        dataset, diff_columns, config, feature_sets, include_backtests, model_names
    )
    summary = summarize_results(results, mode)
    save_results(
        results,
        summary,
        config,
        feature_sets,
        include_backtests,
        diff_columns,
        mode,
        model_names,
    )
    print("Experiment summary:")
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
