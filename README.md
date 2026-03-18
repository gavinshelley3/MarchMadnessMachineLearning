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
   > Tip: you can leave the official Kaggle export inside `data/raw/march_machine_learning_mania_2026/` without shuffling files.  
   > The loader now searches that nested folder (and prefers the largest menâ€™s CSV when duplicates exist) whenever you point commands at `data/raw/`, so the pipeline automatically ignores lightweight samples and the parallel womenâ€™s tables.

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

## Feature Ablation and Pruning Experiments

To understand which advanced feature groups actually help, run the ablation workflow:

```bash
python -m src.experiments --data-dir data/raw --validation-start-season 2015 --mode ablation
```

The ablation mode compares `core` against targeted additions (`core_plus_efficiency`, `core_plus_recent_form`, etc.) using logistic regression and the neural net on the main validation split and rolling backtests. It produces:

- `ablation_comparison.csv` – per-split metrics for each model/feature-set pair in the ablation study.
- `ablation_summary.json` – derived conclusions (best trimmed set per model, improvement vs `core`, which groups to prune, recommended next configuration).
- `ablation_feature_set_summary.json` & `ablation_config.json` – descriptions and reproducibility metadata for the ablation run.

Inspect `ablation_summary.json` to see whether a feature group improves log loss relative to `core`, and to identify consistently harmful groups.

## 2026 Bracket Generation

Once matchup probabilities are saved (see `outputs/predictions/2026_matchup_predictions.csv`), you can turn them into a deterministic projected bracket. The bracket structure itself is stored as editable JSON so the projected field can be replaced with the official bracket when it is announced.

1. Edit or replace `data/brackets/projected_2026_bracket.json` with the desired bracket structure.
2. Generate the bracket:
   ```bash
   python -m src.generate_bracket --bracket-file data/brackets/projected_2026_bracket.json \
       --predictions-file outputs/predictions/2026_matchup_predictions.csv
   ```
3. Outputs land in `outputs/brackets/`:
   - `*_bracket_results.json` – structured per-round breakdown plus metadata about the model and unresolved play-in slots.
   - `*_bracket_results.csv` – flat table of every game with probabilities and winners.
   - `*_bracket_summary.txt` – quick human-readable recap (Final Four, champion, notes).
   - `*_top_upsets.csv` – optional list of higher-seed wins with the highest model confidence.

Use the `--bracket-file` flag to point to the official bracket JSON once it is released.

## 2026 Tournament Simulation

After matchup probabilities exist, you can run Monte Carlo simulations to translate single-game odds into whole-bracket advancement probabilities:

```bash
python -m src.simulate_bracket \
  --bracket-file data/brackets/projected_2026_bracket.json \
  --predictions-file outputs/predictions/2026_matchup_predictions.csv \
  --n-sims 5000 \
  --seed 123
```

Outputs (written to `outputs/brackets/`) include:

- `*_simulation_team_probabilities.csv` – advancement probabilities for every round (Round of 32 through champion).
- `*_simulation_summary.json` – metadata (projected vs official), simulation settings, top champions, most common Final Four/championship matchups, and deterministic confidence counts.
- `*_pick_confidence.csv` – deterministic bracket picks with matchup win probabilities and labeled confidence tiers (high ≥ 0.75, medium 0.60–0.75, low < 0.60).
- `*_upset_risk_report.csv` – optional callouts for lower seeds with meaningful upset chances or favorites with early risk.

Swap `--bracket-file` once the official bracket is available, and adjust `--n-sims`/`--seed` to trade runtime for smoother probability estimates.

## Bracket Visualization

To create a presentation-friendly HTML page that shows every round, regional path, deterministic winners, and the probabilities powering each game, run:

```bash
python -m src.render_bracket_view \
  --bracket-results outputs/brackets/2026_projected_bracket_results.json \
  --output outputs/final_report/bracket_view.html
```

Both flags are optional; by default the script will use the latest `_bracket_results.json` in `outputs/brackets/` and will write to `outputs/final_report/bracket_view.html`. Open the resulting HTML file locally to review the entire bracket without rerunning the pipeline.

The renderer arranges East/West and South/Midwest regions in true bracket form, flows rounds inward toward the Final Four, highlights winners, and calls out the champion prominently for presentation-ready screenshots.

## Final Bracket Selection

After both baseline and CBBpy-enriched predictions/simulations exist, run the selector to build a balanced final bracket without retraining:

```bash
# Optional replays of the deterministic baselines
python -m src.bracket_selection --strategy baseline_deterministic --label baseline
python -m src.bracket_selection --strategy enriched_deterministic --label cbbpy

# Simulation-informed final bracket (default label = strategy name)
python -m src.bracket_selection \
  --strategy simulation_informed \
  --baseline-weight 0.65 \
  --close-margin 0.12 \
  --simulation-weight 0.65 \
  --label final
```

Strategies:

- `baseline_deterministic` – greedy bracket from the production baseline probabilities.
- `enriched_deterministic` – greedy bracket from the CBBpy-enriched probabilities.
- `blended_probabilities` – convex combination of baseline/enriched win probabilities (controlled by `--baseline-weight`).
- `simulation_informed` – blends win probabilities and uses Monte Carlo advancement odds for coin-flip situations while keeping high-confidence favorites intact.

Each run writes labeled artifacts under `outputs/brackets/` (`*_bracket_results_<label>.json/.csv/.txt`) and a presentation HTML view at `outputs/final_report/bracket_view_final.html` unless `--no-render-html` is supplied. The selector also emits `outputs/reports/final_bracket_selection_comparison.json` (optionally `.md`), highlighting champion/final-four changes and round-by-round pick deltas versus both deterministic baselines. Swapping to the official bracket later only requires pointing `--bracket-file` at the new JSON.

## Supplemental Kaggle NCAA Basketball Integration

Download the public [NCAA Basketball](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset) tables locally and extract the **men’s** CSVs into `data/raw/ncaa_basketball/`. Files such as `cbb.csv`, `cbb25.csv`, etc., are automatically detected as men’s data, while any `wbb`/women’s tables are logged and skipped. The dataset inventory report explicitly lists which files were ignored so you can confirm that women’s data never enters the modeling pipeline.

Verify that both datasets are reachable by running:

```bash
python -m src.dataset_diagnostics \
  --data-dir data/raw \
  --include-supplemental-kaggle
```

This command writes:

- `outputs/reports/local_dataset_inventory.json` – recursive file inventory with men’s/women’s classification (women’s files are marked `selected_for_use: false`)
- `outputs/reports/supplemental_team_mapping_report.json` – mapping coverage and unmatched team names
- `outputs/reports/supplemental_feature_summary.json` – supplemental feature counts and join coverage

To compare the current baseline against “core + supplemental NCAA” features, run:

```bash
python -m src.experiments \
  --data-dir data/raw \
  --include-supplemental-kaggle
```

Set `--skip-backtests` if runtime becomes excessive. All supplemental signals remain optional—if the directory is empty, the pipeline logs a warning and falls back to the base March Madness features.

## Current-Season Enrichment with CBBpy

[CBBpy](https://pypi.org/project/cbbpy/) provides an ESPN scraper for men’s college basketball. It is already listed in `requirements.txt`, so `pip install -r requirements.txt` installs it automatically. This repo only uses the **men’s** scraper and never touches `cbbpy.womens_scraper`.

1. Fetch and cache boxscores for the desired portion of the current season (defaults to November 1 → today when omitted):
   ```bash
   python -m src.cbbpy_enrichment \
     --season 2026 \
     --start-date 2026-03-01 \
     --end-date 2026-03-01
   ```
   - Raw boxscores are saved under `data/current_season/cbbpy/` so repeated runs reuse the cache.
   - Team-level feature tables land in `data/current_season/cbbpy/cbbpy_team_features_<season>.csv`.
   - Diagnostics (games scraped, teams matched, unmatched aliases, cache hits) are written to `outputs/reports/cbbpy_fetch_summary.json`.
2. Use the cached features inside the existing pipeline when desired:
   - Dataset diagnostics / experiments / training:
     ```bash
     python -m src.dataset_diagnostics \
       --data-dir data/raw \
       --include-cbbpy-current \
       --cbbpy-season 2026 \
       --cbbpy-features data/current_season/cbbpy/cbbpy_team_features_2026.csv
     ```
   - Experiments: add `--include-cbbpy-current` (and optionally `--feature-sets core core_plus_cbbpy_current`) to compare the production configuration against the CBBpy-augmented set.
   - Training: pass `--include-cbbpy-current` to inject the cached features into the modeling dataset before fitting.

The enrichment remains optional—the model still runs if the CBBpy cache is absent, and the optional feature group can be toggled via `--include-cbbpy-current`. Women’s data is intentionally excluded and any ambiguous team names are logged under `outputs/reports/cbbpy_fetch_summary.json` for review.

## Second Neural Network: Tournament Advancement Model

In addition to the matchup classifier, the repo now contains a team-level advancement model. It builds one row per tournament team/season (using the same pre-tournament features) and predicts multi-output probabilities for reaching the Round of 32, Sweet 16, Elite 8, Final Four, championship game, and winning the title.

1. Build the dataset and train with Adam:
   ```bash
   python -m src.train_advancement \
     --validation-start-season 2015 \
     --optimizer adam \
     --include-supplemental-kaggle
   ```
2. Re-run with SGD + momentum to compare optimizers on the exact same split:
   ```bash
   python -m src.train_advancement \
     --validation-start-season 2015 \
     --optimizer sgd \
     --learning-rate 0.01 \
     --sgd-momentum 0.9 \
     --include-supplemental-kaggle
   ```
3. Calibrate the validation predictions with the stage-specific strategy (auto-picks isotonic for early rounds and Platt for sparse rounds, with safety fallbacks):
   ```bash
   python -m src.advancement_calibration \
     --predictions outputs/predictions/advancement_validation_predictions.csv \
     --strategy auto
   ```
   This writes per-milestone calibrators, `outputs/reports/advancement_calibration_report.json`, a detailed diagnostics file at `outputs/reports/advancement_calibration_diagnostics.json`, and metadata under `outputs/models/advancement/calibrators/calibration_metadata.json` so inference can see which method/blend was used.

Artifacts:
- Processed dataset: `data/processed/advancement_dataset.csv`
- Models/scalers per optimizer: `outputs/models/advancement/<optimizer>/`
- Validation predictions: `outputs/predictions/advancement_validation_<optimizer>.csv`
- Optimizer comparison: `outputs/reports/advancement_optimizer_comparison.{json,csv}`

The advancement NN is optional (nothing in the bracket flow changes automatically), but it provides a bracket-aware plausibility signal that can later augment the deterministic picks or simulation weighting.

## Advancement-Informed Final Bracket Selection

The advancement NN now plugs into the bracket workflow as a transparent tie-breaker:

1. **Infer advancement probabilities for the projected/official field**
   ```bash
   python -m src.advancement_inference \
     --season 2026 \
     --bracket-file data/brackets/projected_2026_bracket.json \
     --baseline-predictions outputs/predictions/2026_matchup_predictions_projected_field_baseline.csv
   ```
   This resolves every projected team (plus play-in options) to TeamIDs, rebuilds their pre-tournament features, loads the saved SGD advancement model, and writes `outputs/predictions/2026_advancement_probabilities.csv` alongside `outputs/reports/advancement_inference_summary.json`.

2. **Blend matchup, simulation, and advancement signals when selecting the final bracket**
   ```bash
   python -m src.bracket_selection \
     --strategy advancement_informed \
     --label final_adv \
     --baseline-predictions outputs/predictions/2026_matchup_predictions_projected_field_baseline.csv \
     --enriched-predictions outputs/predictions/2026_matchup_predictions_projected_field_cbbpy.csv \
     --simulation-probabilities outputs/brackets/2026_projected_simulation_team_probabilities_cbbpy.csv \
     --advancement-probabilities outputs/predictions/2026_advancement_probabilities.csv \
     --baseline-weight 0.65 \
     --simulation-weight 0.35 \
     --advancement-weight 0.3 \
     --close-margin 0.12
   ```
   The `advancement_informed` strategy blends baseline/enriched matchup odds, consults Monte Carlo advancement rates when available, and tilts close calls using the NN’s team-level probabilities. Outputs still land under `outputs/brackets/` (JSON/CSV/txt), the rendered HTML view (default `outputs/final_report/bracket_view_final.html`), and a comparison report (for example `outputs/reports/advancement_bracket_comparison.json`), making it easy to see how the new signal changed Final Four and champion picks.

Both commands remain bracket-file driven, so swapping in the official bracket JSON automatically reuses the same inference + selection flow without touching code.

> **Calibration transparency:** the inference summary (`outputs/reports/advancement_inference_summary.json`) records which calibration metadata file and per-milestone methods were used, ensuring the advancement-informed bracket always references a reproducible calibration strategy.

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
