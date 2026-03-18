"""
Advancement Probability Calibration Module

Fits and applies post-training calibration (Isotonic Regression, Platt Scaling)
to advancement neural network outputs for each tournament milestone.
"""

import argparse
import json
import os
from pathlib import Path
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

MILESTONES = [
    "round_of_32",
    "sweet_16",
    "elite_8",
    "final_four",
    "championship_game",
    "champion",
]

CALIBRATOR_DIR = Path("outputs/models/advancement/calibrators/")
REPORT_PATH = Path("outputs/reports/advancement_calibration_report.json")


def fit_isotonic(y_true, y_pred):
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred, y_true)
    return calibrator

def fit_platt(y_true, y_pred):
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(y_pred.reshape(-1, 1), y_true)
    return lr

def apply_calibrator(calibrator, y_pred, method):
    if method == "isotonic":
        return calibrator.transform(y_pred)
    elif method == "platt":
        return calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")

def load_predictions(pred_path: Path):
    df = pd.read_csv(pred_path)
    return df

def save_calibrator(calibrator, milestone, method):
    CALIBRATOR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CALIBRATOR_DIR / f"{milestone}_{method}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(calibrator, f)
    return str(out_path)

def evaluate_calibration(y_true, y_pred, y_calib):
    return {
        "brier_before": float(brier_score_loss(y_true, y_pred)),
        "brier_after": float(brier_score_loss(y_true, y_calib)),
        "logloss_before": float(log_loss(y_true, y_pred, labels=[0, 1])),
        "logloss_after": float(log_loss(y_true, y_calib, labels=[0, 1])),
    }

def calibration_curve_summary(y_true, y_pred):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }

def main():
    parser = argparse.ArgumentParser(description="Calibrate advancement NN probabilities.")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV with validation predictions.")
    parser.add_argument("--method", choices=["isotonic", "platt"], default="isotonic")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    df = load_predictions(args.predictions)
    report = {"milestones": {}, "method": args.method}

    for milestone in MILESTONES:
        y_true = df[f"label_{milestone}"].values
        y_pred = df[f"prob_{milestone}"].values
        if args.method == "isotonic":
            calibrator = fit_isotonic(y_true, y_pred)
        else:
            calibrator = fit_platt(y_true, y_pred)
        y_calib = apply_calibrator(calibrator, y_pred, args.method)
        save_path = save_calibrator(calibrator, milestone, args.method)
        metrics = evaluate_calibration(y_true, y_pred, y_calib)
        curve = calibration_curve_summary(y_true, y_pred)
        report["milestones"][milestone] = {
            "calibrator_path": save_path,
            **metrics,
            "calibration_curve": curve,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Calibration complete. Report saved to {args.report}")

if __name__ == "__main__":
    main()
