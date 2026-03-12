# March Madness Machine Learning

A deep learning class project that predicts NCAA Division I men's basketball tournament matchups using the Kaggle March Machine Learning Mania dataset. The baseline pipeline aggregates regular-season stats, engineers matchup features, trains a feedforward neural net, and evaluates performance so that future bracket work has a solid foundation.

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
    +-- test_evaluation_metrics.py
`

## Setup

1. **Python**: Use Python 3.11+.
2. **Install dependencies** (prefer a virtual environment):
   `ash
   pip install -r requirements.txt
   `
3. **Obtain Kaggle data**: Download the March Machine Learning Mania dataset and copy the core CSVs into data/raw/:
   - MTeams.csv
   - MRegularSeasonDetailedResults.csv
   - MNCAATourneyCompactResults.csv
   - MNCAATourneySeeds.csv
   - MMasseyOrdinals.csv

## Usage

### Inspect raw tables
`ash
python -m src.inspect_data --data-dir data/raw
`
Lists required files, row counts, season coverage, and seed parsing coverage.

### Dataset diagnostics
`ash
python -m src.dataset_diagnostics --data-dir data/raw
`
Builds the modeling dataset, saves it to data/processed/matchup_dataset.csv, and writes a JSON diagnostic report to outputs/reports/dataset_diagnostics.json.

### Train (with evaluation + reporting)
`ash
python -m src.train --data-dir data/raw --validation-start-season 2015
`
This command:
- Aggregates season-level team stats and matchup features.
- Splits by season (train < 2015, validation = 2015 by default).
- Trains the PyTorch model with early stopping and saves artifacts to outputs/models/.
- Generates an evaluation suite under outputs/reports/:
  - metrics.json – log loss, Brier score, ROC AUC, accuracy, and confusion matrix.
  - al_predictions.csv – matchup IDs with probabilities and predicted classes.
  - calibration_table.csv – 10-bin calibration analysis.
  - seed_gap_metrics.json – accuracy/log loss by seed-difference bucket.
  - upset_metrics.json – underdog accuracy/log loss.
  - aseline_comparison.json – neural net vs. logistic regression vs. seed heuristic.
  - acktest_summary.csv – rolling seasonal backtests for multiple cutoff years.

### Notebooks
Open 
otebooks/01_data_inspection.ipynb in VS Code or Colab for exploratory analysis.

## Testing & CI

Run tests locally:
`ash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
`
GitHub Actions (.github/workflows/ci.yml) runs the same check on pushes and pull requests.

## Automation

- .github/workflows/auto-pr.yml (optional) can auto-open PRs and label them uto-merge when provided with an AUTO_PR_TOKEN secret.
- Branch protection should require the CI / tests workflow before auto-merge completes.

## Model Evaluation and Backtesting

With diagnostics and reporting enabled, you can iterate confidently:
1. python -m src.dataset_diagnostics --data-dir data/raw
2. python -m src.train --data-dir data/raw --validation-start-season 2015

Review the artifacts in outputs/reports/ to understand calibration, upset handling, seed-gap performance, baseline comparisons, and rolling backtests.

## Next Steps

- Expand feature engineering (tempo, opponent-adjusted stats, recency weighting).
- Experiment with alternative models (gradient boosting, calibrated ensembles).
- Layer in bracket simulation logic once matchup predictions are reliable.
- Add GPU-enabled Colab notebooks for faster experimentation.
