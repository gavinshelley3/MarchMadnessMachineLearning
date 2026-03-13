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
    default_ablation_feature_sets,
    default_comparison_feature_sets,
    describe_feature_sets,
    select_diff_columns,
)
from .train import (
    prepare_features,
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
DEFAULT_MODEL_NAMES: Sequence[str] = ("seed_baseline", "logistic_regression", "neural_net")
ABLATION_MODEL_NAMES: Sequence[str] = ("logistic_regression", "neural_net")
MODEL_SEED_OFFSETS = {"logistic_regression": 1, "neural_net": 2}
FEATURE_SET_OFFSETS = {name: idx * 10 for idx, name in enumerate(sorted(available_feature_sets()))}
MODE_CHOICES = ("comparison", "ablation")
DEFAULT_OUTPUT_PREFIX = "model"
ABLATION_OUTPUT_PREFIX = "ablation"
SUPPLEMENTAL_FEATURE_SET_NAME = "core_plus_supplemental_ncaa"
CBBPY_FEATURE_SET_NAME = "core_plus_cbbpy_current"


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


def _feature_set_offset(feature_set: str) -> int:
    if feature_set not in FEATURE_SET_OFFSETS:
        FEATURE_SET_OFFSETS[feature_set] = (len(FEATURE_SET_OFFSETS) + 1) * 10
    return FEATURE_SET_OFFSETS[feature_set]


def _seed_for_variant(base_seed: int, split_index: int, feature_set: str, model_name: str) -> int:
    feature_offset = _feature_set_offset(feature_set)
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
    skip_empty_feature_sets: bool = False,
) -> pd.DataFrame:
    splits = build_splits(dataset, config.training.validation_start_season, include_backtests)
    results: List[Dict[str, object]] = []
    active_models = set(model_names)
    include_seed = "seed_baseline" in active_models
    include_logistic = "logistic_regression" in active_models
    include_nn = "neural_net" in active_models

    for split_index, split in enumerate(splits):
        for feature_set in feature_sets:
            selected_cols = select_diff_columns(diff_columns, feature_set)
            if not selected_cols:
                message = f"No diff columns available for feature set '{feature_set}'. Feature set skipped."
                print(message)
                if skip_empty_feature_sets:
                    continue
                raise ValueError(message)
            X_train, y_train, X_val, y_val, feature_cols, _ = prepare_features(
                split.train_df, split.val_df, selected_cols
            )

            if include_seed:
                seed_probs = run_seed_baseline(split.val_df)
                seed_metrics, _ = compute_metrics(y_val, seed_probs)
                seed_row = _base_result_row(split, feature_set, "seed_baseline", 1)
                seed_row.update(_format_metrics(seed_metrics))
                results.append(seed_row)

            if include_logistic:
                set_random_seed(
                    _seed_for_variant(config.training.random_seed, split_index, feature_set, "logistic_regression")
                )
                logit_probs = run_logistic_regression_baseline(X_train, y_train, X_val)
                logit_metrics, _ = compute_metrics(y_val, logit_probs)
                logistic_row = _base_result_row(split, feature_set, "logistic_regression", len(feature_cols))
                logistic_row.update(_format_metrics(logit_metrics))
                results.append(logistic_row)

            if include_nn:
                set_random_seed(
                    _seed_for_variant(config.training.random_seed, split_index, feature_set, "neural_net")
                )
                _, nn_metrics, _, _ = train_with_arrays(
                    X_train, y_train, X_val, y_val, config
                )
                nn_row = _base_result_row(split, feature_set, "neural_net", len(feature_cols))
                nn_row.update(_format_metrics(nn_metrics))
                results.append(nn_row)

    return pd.DataFrame(results)


def summarize_comparison_results(results: pd.DataFrame) -> Dict[str, object]:
    if results.empty:
        return {"message": "No experiment results generated."}

    summary: Dict[str, object] = {
        "feature_sets": describe_feature_sets(),
    }
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


def summarize_ablation_results(
    results: pd.DataFrame, baseline_feature_set: str = "core"
) -> Dict[str, object]:
    if results.empty:
        return {"message": "No ablation results generated.", "baseline_feature_set": baseline_feature_set}

    summary: Dict[str, object] = {"baseline_feature_set": baseline_feature_set}
    models = sorted(results["model_name"].unique())
    main_summary: Dict[str, Dict[str, float]] = {}
    backtest_summary: Dict[str, Dict[str, float]] = {}
    improvements_vs_core: Dict[str, Dict[str, Dict[str, float]]] = {}
    recommendations: Dict[str, Dict[str, str]] = {}
    harmful_feature_sets: Dict[str, List[str]] = {}
    notes: List[str] = []

    backtest_df = results[results["split_name"].str.startswith("backtest")]

    for model in models:
        model_rows = results[results["model_name"] == model]
        model_main = model_rows[model_rows["split_name"] == "main_validation"]
        model_backtests = backtest_df[backtest_df["model_name"] == model]
        recommendations[model] = {}

        baseline_main_loss = None
        if not model_main.empty:
            baseline_row = model_main[model_main["feature_set"] == baseline_feature_set]
            if not baseline_row.empty:
                baseline_main_loss = float(baseline_row["log_loss"].iloc[0])
            best = model_main.loc[model_main["log_loss"].idxmin()]
            main_summary[model] = {
                "feature_set": best["feature_set"],
                "log_loss": float(best["log_loss"]),
                "accuracy": float(best["accuracy"]),
            }
            recommendations[model]["main_validation"] = best["feature_set"]

            model_improvements = {}
            if baseline_main_loss is not None:
                for _, row in model_main.iterrows():
                    model_improvements[row["feature_set"]] = baseline_main_loss - float(row["log_loss"])
            improvements_vs_core.setdefault(model, {})["main_validation"] = model_improvements

            best_addition = max(
                ((fs, delta) for fs, delta in model_improvements.items() if fs != baseline_feature_set),
                key=lambda item: item[1],
                default=None,
            )
            if best_addition:
                notes.append(
                    f"{model}: best main-split addition vs core is {best_addition[0]} "
                    f"(Δlog_loss={best_addition[1]:+.4f})."
                )
                recommendations[model]["suggested_trimmed_set"] = best_addition[0]
        else:
            main_summary[model] = {}

        if not model_backtests.empty:
            grouped = (
                model_backtests.groupby("feature_set")["log_loss"]
                .mean()
                .reset_index()
            )
            best_bt = grouped.loc[grouped["log_loss"].idxmin()]
            backtest_summary[model] = {
                "feature_set": best_bt["feature_set"],
                "avg_log_loss": float(best_bt["log_loss"]),
            }
            recommendations[model]["backtests"] = best_bt["feature_set"]

            baseline_bt_row = grouped[grouped["feature_set"] == baseline_feature_set]
            baseline_bt = float(baseline_bt_row["log_loss"].iloc[0]) if not baseline_bt_row.empty else None
            bt_improvements = {}
            if baseline_bt is not None:
                for _, row in grouped.iterrows():
                    bt_improvements[row["feature_set"]] = baseline_bt - float(row["log_loss"])
            improvements_vs_core.setdefault(model, {})["backtests"] = bt_improvements

            harm_list: List[str] = []
            for feature_set, bt_delta in bt_improvements.items():
                main_delta = improvements_vs_core.get(model, {}).get("main_validation", {}).get(feature_set)
                if main_delta is not None and bt_delta < 0 and main_delta < 0:
                    harm_list.append(feature_set)
            if harm_list:
                harmful_feature_sets[model] = harm_list
        else:
            backtest_summary[model] = {}

        if "suggested_trimmed_set" not in recommendations[model]:
            recommendations[model]["suggested_trimmed_set"] = baseline_feature_set

    summary["main_validation"] = main_summary
    summary["backtests"] = backtest_summary
    summary["improvements_vs_core"] = improvements_vs_core
    summary["recommendations"] = recommendations
    if harmful_feature_sets:
        summary["consistently_harmful_feature_sets"] = harmful_feature_sets

    all_adv_rows = results[results["feature_set"] == "core_plus_all_advanced"]
    for model in models:
        adv_main = all_adv_rows[
            (all_adv_rows["model_name"] == model) & (all_adv_rows["split_name"] == "main_validation")
        ]
        core_main = results[
            (results["feature_set"] == baseline_feature_set)
            & (results["model_name"] == model)
            & (results["split_name"] == "main_validation")
        ]
        if not adv_main.empty and not core_main.empty:
            delta = float(core_main["log_loss"].iloc[0]) - float(adv_main["log_loss"].iloc[0])
            summary.setdefault("core_plus_all_advanced_effect", {})[model] = {"main_validation_delta": delta}
            notes.append(
                f"{model}: core_plus_all_advanced changes main log loss by {delta:+.4f} vs core."
            )

    if notes:
        summary["notes"] = notes

    return summary


def save_results(
    results: pd.DataFrame,
    summary: Dict[str, object],
    config,
    feature_sets: Sequence[str],
    include_backtests: bool,
    diff_columns: Sequence[str],
    model_names: Sequence[str],
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> None:
    reports_dir = config.paths.outputs_dir / "reports"
    base_path = reports_dir / f"{output_prefix}_comparison.csv"
    ensure_parent_dir(base_path)

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
    results.to_csv(base_path, columns=ordered_cols, index=False)

    summary_path = reports_dir / f"{output_prefix}_summary.json"
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
    (reports_dir / f"{output_prefix}_feature_set_summary.json").write_text(
        json.dumps(feature_set_summary, indent=2, default=_json_default)
    )

    experiment_config = {
        "feature_sets": list(feature_sets),
        "models": list(model_names),
        "include_backtests": include_backtests,
        "validation_start_season": config.training.validation_start_season,
        "backtest_splits_attempted": BACKTEST_SPLITS if include_backtests else [],
    }
    (reports_dir / f"{output_prefix}_config.json").write_text(
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
        help="Feature sets to compare. Defaults to all available.",
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="comparison",
        help="Select 'comparison' for the standard model comparison or 'ablation' for feature pruning studies.",
    )
    parser.add_argument(
        "--include-supplemental-kaggle",
        action="store_true",
        help="Include supplemental Kaggle NCAA Basketball features if available.",
    )
    parser.add_argument(
        "--supplemental-dir",
        type=Path,
        default=None,
        help="Override supplemental NCAA Basketball directory.",
    )
    parser.add_argument(
        "--include-cbbpy-current",
        action="store_true",
        help="Include cached current-season CBBpy features if available.",
    )
    parser.add_argument(
        "--cbbpy-season",
        type=int,
        default=None,
        help="Season to load from the cached CBBpy features (e.g., 2026).",
    )
    parser.add_argument(
        "--cbbpy-features",
        type=Path,
        default=None,
        help="Explicit path to a cached CBBpy team-features CSV.",
    )
    return parser.parse_args()


def _default_feature_sets_for_mode(mode: str) -> List[str]:
    if mode == "ablation":
        return default_ablation_feature_sets()
    return default_comparison_feature_sets()


def _ensure_feature_set(feature_sets: List[str], name: str) -> List[str]:
    if name in available_feature_sets() and name not in feature_sets:
        return feature_sets + [name]
    return feature_sets


def main() -> None:
    args = parse_args()
    config = get_config()
    if args.validation_start_season is not None:
        config.training.validation_start_season = args.validation_start_season

    data_dir = args.data_dir if args.data_dir else config.paths.raw_data_dir
    supplemental_dir = args.supplemental_dir if args.supplemental_dir else config.paths.supplemental_ncaa_dir
    dataset, diagnostics = load_and_build_dataset(
        data_dir,
        include_supplemental_kaggle=args.include_supplemental_kaggle,
        supplemental_dir=supplemental_dir,
        reports_dir=config.paths.outputs_dir / "reports",
        include_cbbpy_current=args.include_cbbpy_current,
        cbbpy_features_path=args.cbbpy_features,
        cbbpy_season=args.cbbpy_season,
    )
    feature_summary_path = config.paths.outputs_dir / "reports" / "feature_summary.json"
    save_feature_summary_report(diagnostics, feature_summary_path)
    diff_columns = sorted([col for col in dataset.columns if col.startswith("Diff_")])

    include_backtests = not args.skip_backtests
    mode = args.mode

    if args.feature_sets:
        feature_sets = list(args.feature_sets)
    else:
        feature_sets = _default_feature_sets_for_mode(mode)
        if args.include_supplemental_kaggle:
            feature_sets = _ensure_feature_set(feature_sets, SUPPLEMENTAL_FEATURE_SET_NAME)
        if args.include_cbbpy_current:
            feature_sets = _ensure_feature_set(feature_sets, CBBPY_FEATURE_SET_NAME)

    if mode == "ablation":
        model_names = ABLATION_MODEL_NAMES
        skip_empty = True
        output_prefix = ABLATION_OUTPUT_PREFIX
        summary_func = summarize_ablation_results
    else:
        model_names = DEFAULT_MODEL_NAMES
        skip_empty = False
        output_prefix = DEFAULT_OUTPUT_PREFIX
        summary_func = summarize_comparison_results

    print(f"[Mode: {mode}] Running experiments for feature sets: {', '.join(feature_sets)}")
    print(f"Models: {', '.join(model_names)}")
    if include_backtests:
        print("Rolling backtests: enabled")
    else:
        print("Rolling backtests: skipped")

    results = run_experiments(
        dataset,
        diff_columns,
        config,
        feature_sets,
        include_backtests,
        model_names,
        skip_empty_feature_sets=skip_empty,
    )
    summary = summary_func(results)
    save_results(
        results,
        summary,
        config,
        feature_sets,
        include_backtests,
        diff_columns,
        model_names,
        output_prefix,
    )
    print("Experiment summary:")
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
