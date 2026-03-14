from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.train_advancement import (
    LABEL_NAMES,
    prepare_features,
    prepare_targets,
    select_feature_columns,
    train_with_optimizer,
)


def _build_dummy_dataset() -> pd.DataFrame:
    rows = []
    for idx, team_id in enumerate(range(1, 6), start=0):
        rows.append(
            {
                "Season": 2015 + idx,
                "TeamID": team_id,
                "TeamName": f"Team {team_id}",
                "Seed": "W0{}".format((idx % 5) + 1),
                "SeedNum": (idx % 5) + 1,
                "Wins": idx % 3,
                "PlayInOffset": 0,
                "MetricA": float(team_id),
                "MetricB": float(team_id * 2),
                "reached_round_of_32": int(idx >= 1),
                "reached_sweet_16": int(idx >= 2),
                "reached_elite_8": int(idx >= 3),
                "reached_final_four": int(idx >= 4),
                "reached_championship_game": int(idx >= 4),
                "won_championship": int(idx == 4),
            }
        )
    return pd.DataFrame(rows)


def test_select_feature_columns_excludes_labels() -> None:
    df = _build_dummy_dataset()
    features = select_feature_columns(df)
    for label in LABEL_NAMES:
        assert label not in features
    assert "SeedNum" in features


def test_train_with_optimizer_smoke(tmp_path) -> None:
    df = _build_dummy_dataset()
    train_df = df.iloc[:4].reset_index(drop=True)
    val_df = df.iloc[4:].reset_index(drop=True)
    feature_cols = select_feature_columns(df)
    X_train, X_val, scaler = prepare_features(train_df, val_df, feature_cols)
    y_train, y_val = prepare_targets(train_df, val_df)
    args = SimpleNamespace(
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        sgd_momentum=0.9,
        max_epochs=2,
        patience=2,
        min_delta=0.0,
        seed=123,
    )
    result = train_with_optimizer(
        "sgd",
        X_train,
        y_train,
        X_val,
        y_val,
        feature_cols,
        scaler,
        train_df,
        val_df,
        args,
        tmp_path / "models",
        tmp_path / "predictions",
    )
    assert "metrics" in result
    assert set(LABEL_NAMES).issubset(result["metrics"]["per_label"].keys())
