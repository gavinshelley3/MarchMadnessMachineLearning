# March Madness Machine Learning

A deep learning class project that builds an end-to-end pipeline for predicting NCAA Division I men's basketball tournament matchups using the Kaggle March Machine Learning Mania dataset. The initial goal is to aggregate season stats, create matchup features, train a feedforward neural net, and evaluate performance, with a path toward generating a 2026 bracket.

## Project Structure

`
+-- data/
¦   +-- raw/                  # Place Kaggle CSVs here (ignored by git)
¦   +-- processed/
+-- notebooks/
¦   +-- 01_data_inspection.ipynb
+-- outputs/
¦   +-- models/               # Saved weights, scalers
+-- src/
¦   +-- config.py             # Paths + hyperparameters
¦   +-- data_loading.py       # CSV loaders
¦   +-- inspect_data.py       # CLI data summary
¦   +-- feature_engineering.py
¦   +-- dataset_builder.py
¦   +-- model.py              # PyTorch tabular NN
¦   +-- train.py              # Training entrypoint
¦   +-- evaluate.py           # Metrics helpers
¦   +-- utils.py
+-- tests/
    +-- test_feature_engineering.py
`

## Setup

1. **Python**: Use Python 3.11+.
2. **Install dependencies** (prefer a virtual environment):
   `ash
   pip install -r requirements.txt
   `
3. **Obtain Kaggle data**: Download the March Machine Learning Mania dataset from Kaggle and copy these files into data/raw/:
   - MTeams.csv
   - MRegularSeasonDetailedResults.csv
   - MNCAATourneyCompactResults.csv
   - MNCAATourneySeeds.csv
   - MMasseyOrdinals.csv

VS Code is the primary local editor. You can also sync this repo to Google Colab later for GPU training since everything lives in the src/ package and uses relative paths.

## Usage

### Inspect data

`ash
python -m src.inspect_data --data-dir data/raw
`
Use --tables to inspect a subset (e.g., --tables teams tourney_seeds).

### Train the baseline model

`ash
python -m src.train --data-dir data/raw --validation-start-season 2015
`
This command will:
- build season-level team aggregates
- create symmetric tournament matchup features
- split by season (training before 2015, validation 2015+ by default)
- standardize features, train the PyTorch model, run early stopping, and
- save artifacts (model weights, scaler, feature list) inside outputs/models/

### Notebooks

Open 
otebooks/01_data_inspection.ipynb in VS Code or Colab to explore the raw data interactively.

## Testing & CI

Lightweight smoke tests live under 	ests/. Run them locally with:

`ash
pytest
`

GitHub Actions (.github/workflows/ci.yml) runs these checks on each push and pull request. An accompanying workflow (.github/workflows/auto-merge.yml) can enable auto-merge for PRs once you apply the uto-merge label on GitHub and all checks succeed.

## Automation

- Pushing to any non-default branch triggers the Auto PR workflow (`.github/workflows/auto-pr.yml`) which opens/updates a PR targeting the default branch, applies the `auto-merge` label, and enables GitHub auto-merge.
- Remove the `auto-merge` label (or disable auto-merge in the PR UI) if you want to pause the automatic merge.
- Branch protection now enforces the `CI / tests` check, so the PR will merge itself after that workflow passes.

## Next Steps

- Experiment with richer feature engineering (tempo, opponent-adjusted stats, recent-form windows)
- Explore alternative models (gradient boosting, calibrated ensembles)
- Integrate bracket generation logic once the matchup predictor is stable
- Add Colab notebooks for GPU training runs
