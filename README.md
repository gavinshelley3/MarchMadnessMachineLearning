# March Madness Machine Learning

A deep learning class project built around the Kaggle March Machine Learning Mania dataset. The repo aggregates regular-season statistics, engineers matchup features, trains a neural network, and reports reproducible diagnostics so future bracket research has a trustworthy baseline.

## Project Structure

```
.
+-- data/
|   +-- raw/        # Place Kaggle CSVs here (ignored by git)
|   +-- processed/
+-- notebooks/
|   +-- 01_data_inspection.ipynb
+-- outputs/
|   +-- models/     # Saved models, scalers
|   +-- reports/    # Metrics, predictions, diagnostics
+-- src/
|   +-- config.py
|   +-- data_loading.py
|   +-- feature_engineering.py
|   +-- dataset_builder.py
|   +-- data_pipeline.py
|   +-- dataset_diagnostics.py
|   +-- experiments.py
|   +-- train.py
|   +-- evaluate.py
|   +-- utils.py
+-- tests/
    +-- test_feature_engineering.py
    +-- test_evaluation_metrics.py
    +-- test_advanced_features.py
```

## Setup

1. **Python**: Use Python 3.11+.
2. **Install dependencies** (ideally in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Kaggle data** and copy these five CSVs into `data/raw/`:
   - `MTeams.csv`
   - `MRegularSeasonDetailedResults.csv`
   - `MNCAATourneyCompactResults.csv`
   - `MNCAATourneySeeds.csv`
   - `MMasseyOrdinals.csv`

## Usage

### Inspect raw tables
```bash
python -m src.inspect_data --data-dir data/raw
```
Outputs row counts, season coverage, and seed parsing coverage for each required table.

### Build dataset diagnostics
```bash
python -m src.dataset_diagnostics --data-dir data/raw
```
Writes the modeling dataset to `data/processed/matchup_dataset.csv`, a summary JSON to `outputs/reports/dataset_diagnostics.json`, and a per-feature summary to `outputs/reports/feature_summary.json`.

### Train (with evaluation + reporting)
```bash
python -m src.train --data-dir data/raw --validation-start-season 2015
```
This command:
- Builds season-level team features and symmetric matchup rows.
- Splits by season (train < 2015, validation = 2015 by default).
- Trains the PyTorch model with early stopping and saves artifacts in `outputs/models/`.
- Generates a research-grade report suite in `outputs/reports/`:
  - `metrics.json` – log loss, Brier score, ROC AUC, accuracy, and confusion matrix.
  - `val_predictions.csv` – matchup IDs with predicted probabilities and classes.
  - `calibration_table.csv` – 10-bin calibration diagnostics.
  - `seed_gap_metrics.json` & `upset_metrics.json` – performance on favorites vs. underdogs.
  - `baseline_comparison.json` – neural net vs. logistic regression vs. seed heuristic.
  - `backtest_summary.csv` – rolling seasonal backtests for multiple cutoffs.

### Notebooks
Open `notebooks/01_data_inspection.ipynb` in VS Code or Colab for exploratory data work.

## Advanced Feature Engineering

The feature layer now captures richer basketball signals:
- **Shooting efficiency**: effective FG%, true shooting %, 3PA rate, FT rate.
- **Possession-based ratings**: estimated possessions, offensive/defensive ratings, net rating.
- **Rebounding & turnover margins**: rebound margin, offensive rebounding rate, turnover margin, assist rate.
- **Recent form windows**: 5- and 10-game win %, scoring margin, offensive/defensive ratings, and eFG%.
- **Opponent adjustments**: adjusted scoring margin subtracts the average opponent margin faced.
- **Neutral-site awareness**: win % and scoring margin in neutral games to better match tournament conditions.

Every new feature feeds directly into the matchup dataset as Team1 - Team2 differences, so the existing training and evaluation stack automatically benefits.

## Model Evaluation and Backtesting

For a full diagnostics loop:
1. `python -m src.dataset_diagnostics --data-dir data/raw`
2. `python -m src.train --data-dir data/raw --validation-start-season 2015`

Review `outputs/reports/` for calibration, upset handling, seed-gap performance, baseline comparisons, and rolling backtests.

## Model Comparison Experiments

Use the experiment runner to compare feature sets and model families:

```bash
python -m src.experiments --data-dir data/raw --validation-start-season 2015
```

This command rebuilds the dataset, evaluates the seed baseline, logistic regression, and the neural net across the core-only and advanced feature sets, and replays the rolling backtests. It writes the following artifacts to `outputs/reports/`:

- `model_comparison.csv` – per-split metrics (log loss, Brier score, ROC AUC, accuracy) for every feature-set/model pair.
- `model_comparison_summary.json` – key conclusions (best validation config, backtest improvements, neural vs. logistic wins).
- `feature_set_summary.json` – descriptions and column counts per feature set.
- `experiment_config.json` – reproducibility details (feature sets, models, splits, validation season).

## Testing & CI

Run the test suite locally with:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```
GitHub Actions (`.github/workflows/ci.yml`) enforces the same tests on pushes and pull requests.

## Automation

- `.github/workflows/auto-pr.yml` (optional) can auto-open PRs and label them `auto-merge` when an `AUTO_PR_TOKEN` secret is supplied.
- Branch protection should require the `CI / tests` workflow before enabling auto-merge.

## Next Steps

- Prototype richer basketball features (tempo, opponent-adjusted shooting) and compare via the new reports.
- Explore alternative models (gradient boosting, calibrated ensembles), then wire them into the existing evaluation harness.
- Layer in bracket simulation once matchup predictions are reliable.
- Add GPU-enabled Colab notebooks for faster experimentation.
