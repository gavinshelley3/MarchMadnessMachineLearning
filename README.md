# March Madness Machine Learning

A deep learning class project that builds an end-to-end pipeline for predicting NCAA Division I men's basketball tournament matchups using the Kaggle March Machine Learning Mania dataset. The initial goal is to aggregate season stats, create matchup features, train a feedforward neural net, and evaluate performance, with a path toward generating a 2026 bracket.

## Project Structure

`
.
+-- data/
¦   +-- raw/        # Place Kaggle CSVs here (ignored by git)
¦   +-- processed/
+-- notebooks/
¦   +-- 01_data_inspection.ipynb
+-- outputs/
¦   +-- models/     # Saved weights, scalers
¦   +-- reports/    # Metrics, predictions, diagnostics
+-- src/
¦   +-- config.py
¦   +-- data_loading.py
¦   +-- inspect_data.py
¦   +-- feature_engineering.py
¦   +-- dataset_builder.py
¦   +-- data_pipeline.py
¦   +-- dataset_diagnostics.py
¦   +-- model.py
¦   +-- train.py
¦   +-- evaluate.py
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

### Dataset diagnostics
`ash
python -m src.dataset_diagnostics --data-dir data/raw
`
This builds the modeling dataset, saves it to data/processed/matchup_dataset.csv, and writes diagnostics to outputs/reports/dataset_diagnostics.json.

### Train the baseline model
`ash
python -m src.train --data-dir data/raw --validation-start-season 2015
`
This command will:
- build season-level team aggregates
- create symmetric tournament matchup features
- split by season (training before 2015, validation 2015+ by default)
- standardize features, train the PyTorch model, run early stopping, and
- save artifacts (model weights, scaler, feature lists) inside outputs/models/
- emit validation predictions (outputs/reports/val_predictions.csv) and metrics (outputs/reports/val_metrics.json)

### Notebooks
Open 
otebooks/01_data_inspection.ipynb in VS Code or Colab to explore the raw data interactively.

## Testing & CI

Lightweight smoke tests live under 	ests/. Run them locally with:
`ash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
`

GitHub Actions (.github/workflows/ci.yml) runs these checks on each push and pull request. An accompanying workflow (.github/workflows/auto-merge.yml) can enable auto-merge for PRs once you apply the uto-merge label on GitHub and all checks succeed.

## Automation

- Pushing to any non-default branch triggers the Auto PR workflow (.github/workflows/auto-pr.yml) which (when configured) opens/updates a PR targeting the default branch, applies the uto-merge label, and enables GitHub auto-merge.
- Remove the uto-merge label (or disable auto-merge in the PR UI) if you want to pause the automatic merge.
- Branch protection should enforce the CI / tests check so the PR merges itself after the workflow passes.
- Add a epo-scoped personal access token as the AUTO_PR_TOKEN repository secret if you want the workflow to open/label PRs for you; without it, the workflow simply logs that auto PRs are skipped.

## Next Steps

- Experiment with richer feature engineering (tempo, opponent-adjusted stats, recent-form windows)
- Explore alternative models (gradient boosting, calibrated ensembles)
- Integrate bracket generation logic once the matchup predictor is stable
- Add Colab notebooks for GPU training runs
