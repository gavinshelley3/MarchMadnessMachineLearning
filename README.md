# March Madness Machine Learning

A deep learning class project built around the Kaggle March Machine Learning Mania dataset. The repo aggregates regular-season statistics, engineers matchup features, trains a neural network, and reports reproducible diagnostics so future bracket research has a trustworthy baseline.

## Project Structure

```
.
+-- data/
|   +-- raw/        # Kaggle CSVs checked into git for reproducibility
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
  - `metrics.json` â€“ log loss, Brier score, ROC AUC, accuracy, and confusion matrix.
  - `val_predictions.csv` â€“ matchup IDs with predicted probabilities and classes.
  - `calibration_table.csv` â€“ 10-bin calibration diagnostics.
  - `seed_gap_metrics.json` & `upset_metrics.json` â€“ performance on favorites vs. underdogs.
  - `baseline_comparison.json` â€“ neural net vs. logistic regression vs. seed heuristic.
  - `backtest_summary.csv` â€“ rolling seasonal backtests for multiple cutoffs.

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

Use the experiment runner to compare feature sets and model families (--mode comparison is the default):

`ash
python -m src.experiments --data-dir data/raw --validation-start-season 2015 --mode comparison
`

This command rebuilds the dataset, evaluates the seed baseline, logistic regression, and the neural net across the core-only and advanced feature sets, and replays the rolling backtests. It writes the following artifacts to outputs/reports/:

- model_comparison.csv – per-split metrics (log loss, Brier score, ROC AUC, accuracy) for every feature-set/model pair.
- model_comparison_summary.json – key conclusions (best validation config, backtest improvements, neural vs. logistic wins).
- eature_set_summary.json – descriptions and column counts per feature set.
- experiment_config.json – reproducibility details (feature sets, models, splits, validation season, chosen mode).

## Feature Ablation and Pruning Experiments

Identify which advanced feature families add signal on top of the core slate:

`ash
python -m src.experiments --data-dir data/raw --validation-start-season 2015 --mode ablation
`

Artifacts (blation_comparison.csv, blation_summary.json) quantify improvements vs. the core baseline for each model and feature group, and summarize rolling backtest performance.

## Final Model Selection Experiments

Promote the strongest feature sets and models before bracket simulation:

`ash
python -m src.experiments --data-dir data/raw --validation-start-season 2015 --mode final
`

This mode restricts comparisons to the finalist configurations (logistic regression with core_plus_opponent_adjustment, neural net with core, optional gradient boosting) and produces:

- inal_model_comparison.csv – split-by-split metrics for every finalist pair.
- inal_model_summary.json – best validation/backtest performers, per-model averages, ranked configurations, and the recommended production choice for future brackets.

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

## 2026 Inference Pipeline

The locked production configuration is **logistic regression** trained on the core_plus_opponent_adjustment feature set. The CLI below ensures the production artifacts exist (training and saving them under outputs/models/production/ if necessary) and then scores every 2026 matchup listed in Kaggle’s SampleSubmissionStage2.csv.

`ash
python -m src.predict_2026 --data-dir data/raw
` 

Key behaviors:

- Looks up (or retrains, with --force-retrain) the production logistic model + scaler.
- Builds 2026 regular-season team features, joins the 2026 tournament teams from the sample submission, and creates pairwise diff features.
- Saves predictions under outputs/predictions/:
  - 2026_matchup_predictions.csv – probabilities, seeds, team names, and the predicted winner for every Kaggle matchup ID.
  - 2026_prediction_metadata.json – provenance, artifact metadata, coverage stats, and file paths for reproducibility.

Need a different bracket file? Pass --sample-submission path/to/file.csv. Use --force-retrain any time you want to regenerate the production model artifacts from scratch.

