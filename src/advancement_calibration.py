"""
Advancement Probability Calibration Module

Fits and applies post-training calibration (Isotonic Regression, Platt Scaling,
or mixed strategies) to advancement neural network outputs for each tournament
milestone.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .advancement_dataset import ADVANCEMENT_LABELS

LABEL_NAMES = [label for label, _ in ADVANCEMENT_LABELS]
MILESTONES = {
    "reached_round_of_32": "round_of_32",
    "reached_sweet_16": "sweet_16",
    "reached_elite_8": "elite_8",
    "reached_final_four": "final_four",
    "reached_championship_game": "championship_game",
    "won_championship": "champion",
}

CALIBRATOR_DIR = Path("outputs/models/advancement/calibrators/")
REPORT_PATH = Path("outputs/reports/advancement_calibration_report.json")
DIAGNOSTICS_PATH = Path("outputs/reports/advancement_calibration_diagnostics.json")
METADATA_PATH = CALIBRATOR_DIR / "calibration_metadata.json"


def fit_isotonic(y_true: np.ndarray, y_pred: np.ndarray):
    """Fit an isotonic regression calibrator."""

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred, y_true)
    return calibrator


def fit_platt(y_true: np.ndarray, y_pred: np.ndarray):
    """Fit a Platt-scaling (logistic) calibrator."""

    lr = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
    )
    lr.fit(y_pred.reshape(-1, 1), y_true)
    return lr


def apply_calibrator(calibrator, y_pred: np.ndarray, method: str) -> np.ndarray:
    """Apply a fitted calibrator."""

    if method == "isotonic":
        return calibrator.transform(y_pred)
    if method == "platt":
        return calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
    raise ValueError(f"Unknown calibration method: {method}")


def load_predictions(pred_path: Path) -> pd.DataFrame:
    return pd.read_csv(pred_path)


def save_calibrator(calibrator, milestone: str, method: str) -> str:
    CALIBRATOR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CALIBRATOR_DIR / f"{milestone}_{method}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(calibrator, f)
    return str(out_path)


def evaluate_calibration(y_true: np.ndarray, y_pred: np.ndarray, y_calib: np.ndarray) -> Dict[str, float]:
    return {
        "brier_before": float(brier_score_loss(y_true, y_pred)),
        "brier_after": float(brier_score_loss(y_true, y_calib)),
        "logloss_before": float(log_loss(y_true, y_pred, labels=[0, 1])),
        "logloss_after": float(log_loss(y_true, y_calib, labels=[0, 1])),
    }


def calibration_curve_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}


def _select_method(strategy: str, positives: int, isotonic_threshold: int) -> str:
    if strategy == "auto":
        return "isotonic" if positives >= isotonic_threshold else "platt"
    return strategy


def _value_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(values)) if len(values) else 0.0,
        "max": float(np.max(values)) if len(values) else 0.0,
        "mean": float(np.mean(values)) if len(values) else 0.0,
        "unique_values": int(len(np.unique(np.round(values, 6)))),
        "zero_fraction": float(np.mean(np.isclose(values, 0.0))) if len(values) else 0.0,
    }


def _detect_collapse(stats: Dict[str, float], min_unique: int, max_zero_fraction: float) -> Tuple[bool, str]:
    if stats["unique_values"] < min_unique:
        return True, f"unique_values<{min_unique}"
    if stats["zero_fraction"] > max_zero_fraction:
        return True, f"zero_fraction>{max_zero_fraction}"
    return False, ""


def _detect_metric_regression(
    metrics: Dict[str, float],
    max_brier_increase: float,
    max_logloss_increase: float,
) -> Tuple[bool, str]:
    brier_delta = metrics["brier_after"] - metrics["brier_before"]
    if brier_delta > max_brier_increase:
        return True, f"brier_increase>{max_brier_increase:.4f}"
    logloss_delta = metrics["logloss_after"] - metrics["logloss_before"]
    if logloss_delta > max_logloss_increase:
        return True, f"logloss_increase>{max_logloss_increase:.4f}"
    return False, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate advancement NN probabilities.")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV with validation predictions.")
    parser.add_argument(
        "--strategy",
        choices=["auto", "isotonic", "platt"],
        default="auto",
        help="Calibration strategy: auto picks per-milestone methods.",
    )
    parser.add_argument(
        "--auto-isotonic-threshold",
        type=int,
        default=120,
        help="Minimum positive labels required to prefer isotonic in auto strategy.",
    )
    parser.add_argument(
        "--min-calibrated-unique",
        type=int,
        default=4,
        help="Fallback to blending if calibrated values have fewer distinct values than this.",
    )
    parser.add_argument(
        "--max-zero-fraction",
        type=float,
        default=0.9,
        help="Fallback to blending if more than this fraction of calibrated values are ≈0.",
    )
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=0.4,
        help="Portion of raw probabilities kept when blending with calibrated outputs after a collapse.",
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--diagnostics-report", type=Path, default=DIAGNOSTICS_PATH)
    parser.add_argument("--metadata-json", type=Path, default=METADATA_PATH)
    parser.add_argument(
        "--max-brier-increase",
        type=float,
        default=0.001,
        help="Maximum allowed absolute Brier score increase before falling back to blending/raw outputs.",
    )
    parser.add_argument(
        "--max-logloss-increase",
        type=float,
        default=0.01,
        help="Maximum allowed absolute log-loss increase before falling back to blending/raw outputs.",
    )
    parser.add_argument(
        "--degrade-blend-weight",
        type=float,
        default=1.0,
        help="Raw probability weight when falling back due to metric regression (use 1.0 to skip calibration).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 <= args.blend_weight <= 1.0):
        raise ValueError("--blend-weight must be in [0, 1].")

    df = load_predictions(args.predictions)
    calibration_report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": args.strategy,
        "milestones": {},
    }
    diagnostics: Dict[str, Any] = {
        "generated_at": calibration_report["generated_at"],
        "strategy": args.strategy,
        "auto_isotonic_threshold": args.auto_isotonic_threshold,
        "min_calibrated_unique": args.min_calibrated_unique,
        "max_zero_fraction": args.max_zero_fraction,
        "blend_weight": args.blend_weight,
        "milestones": {},
    }
    metadata: Dict[str, Any] = {
        "generated_at": calibration_report["generated_at"],
        "strategy": args.strategy,
        "milestones": {},
    }

    for label in LABEL_NAMES:
        stage = MILESTONES[label]
        true_col = f"true_{label}"
        pred_col = f"pred_{label}"
        if true_col not in df.columns or pred_col not in df.columns:
            raise KeyError(f"Missing columns {true_col} / {pred_col} in {args.predictions}")
        y_true = df[true_col].to_numpy(dtype=float)
        y_pred = df[pred_col].to_numpy(dtype=float)
        positives = int(y_true.sum())
        method = _select_method(args.strategy, positives, args.auto_isotonic_threshold)
        calibrator = fit_isotonic(y_true, y_pred) if method == "isotonic" else fit_platt(y_true, y_pred)
        y_calib = apply_calibrator(calibrator, y_pred, method)

        raw_stats = _value_stats(y_pred)
        calib_stats = _value_stats(y_calib)
        collapsed, reason = _detect_collapse(
            calib_stats,
            args.min_calibrated_unique,
            args.max_zero_fraction,
        )
        metrics = evaluate_calibration(y_true, y_pred, y_calib)
        curve = calibration_curve_summary(y_true, y_pred)
        save_path = save_calibrator(calibrator, stage, method)
        metric_regressed, metric_reason = _detect_metric_regression(
            metrics,
            args.max_brier_increase,
            args.max_logloss_increase,
        )

        fallback_applied = False
        fallback_reason = ""
        fallback_weight = 0.0
        if collapsed or metric_regressed:
            fallback_applied = True
            fallback_reason = reason or metric_reason
            weight = args.blend_weight if collapsed else args.degrade_blend_weight
            weight = float(np.clip(weight, 0.0, 1.0))
            fallback_weight = weight
            if weight >= 0.999:
                y_calib = y_pred.copy()
                calib_stats = _value_stats(y_calib)
                print(f"[calibration] Raw fallback applied for {stage}: {fallback_reason}")
            else:
                y_calib = weight * y_pred + (1.0 - weight) * y_calib
                calib_stats = _value_stats(y_calib)
                print(f"[calibration] Blend fallback applied for {stage}: {fallback_reason} (weight={weight:.2f})")
            metrics = evaluate_calibration(y_true, y_pred, y_calib)

        calibration_report["milestones"][stage] = {
            "calibrator_path": save_path,
            "method": method,
            **metrics,
            "calibration_curve": curve,
            "blend_applied": fallback_applied,
            "blend_weight": fallback_weight if fallback_applied else 0.0,
            "fallback_reason": fallback_reason,
        }
        diagnostics["milestones"][stage] = {
            "positives": positives,
            "raw": raw_stats,
            "calibrated": calib_stats,
            "collapse_detected": collapsed,
            "collapse_reason": reason,
            "metric_regressed": metric_regressed,
            "metric_reason": metric_reason,
            **metrics,
        }
        metadata["milestones"][stage] = {
            "calibrator_path": save_path,
            "method": method,
            "positives": positives,
            "blend_applied": fallback_applied,
            "blend_weight": fallback_weight if fallback_applied else None,
            "fallback_reason": fallback_reason,
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_report.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata_json
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2)
    with open(args.diagnostics_report, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Calibration complete. Report saved to {args.report}")
    print(f"Diagnostics written to {args.diagnostics_report}")
    print(f"Calibration metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
