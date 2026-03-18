import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from src.advancement_calibration import fit_isotonic, fit_platt, apply_calibrator

@pytest.fixture
def sample_data():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.4, 0.6])
    return y_true, y_pred


def test_isotonic_fit_and_transform(sample_data):
    y_true, y_pred = sample_data
    calibrator = fit_isotonic(y_true, y_pred)
    y_calib = calibrator.transform(y_pred)
    assert y_calib.shape == y_pred.shape
    assert np.all((y_calib >= 0) & (y_calib <= 1))


def test_platt_fit_and_transform(sample_data):
    y_true, y_pred = sample_data
    calibrator = fit_platt(y_true, y_pred)
    y_calib = calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
    assert y_calib.shape == y_pred.shape
    assert np.all((y_calib >= 0) & (y_calib <= 1))


def test_apply_calibrator_isotonic(sample_data):
    y_true, y_pred = sample_data
    calibrator = fit_isotonic(y_true, y_pred)
    y_calib = apply_calibrator(calibrator, y_pred, method="isotonic")
    assert np.allclose(y_calib, calibrator.transform(y_pred))


def test_apply_calibrator_platt(sample_data):
    y_true, y_pred = sample_data
    calibrator = fit_platt(y_true, y_pred)
    y_calib = apply_calibrator(calibrator, y_pred, method="platt")
    assert np.allclose(y_calib, calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1])


def test_calibrator_pickle(tmp_path, sample_data):
    y_true, y_pred = sample_data
    calibrator = fit_isotonic(y_true, y_pred)
    pkl_path = tmp_path / "calibrator.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(calibrator, f)
    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)
    y_calib = loaded.transform(y_pred)
    assert np.allclose(y_calib, calibrator.transform(y_pred))
