# Upgrade Notes

This documents the move from "Tkinter desktop tool" to "API microservice
plus drift monitoring." It's split into three parts: bugs that had to be
fixed before that upgrade was even possible, what's new, and the design
decisions worth knowing about if you extend this further.

## Before touching any of the new features: the existing code didn't run

None of this was mentioned in the task, but it's the reason the upgrade
took more than just adding two new files. `src/` was a package mid-refactor
that had never actually been executed end-to-end - every cross-module
import in it was broken:

1. **`src/_init_.py`** - single underscores, not `__init__.py`. `src` was
   never actually a package via this file.
2. **`src/config.py` didn't exist.** A near-duplicate lived at
   `src/tuning/config.py`, which also had its own bugs: `ROOT_DIR` was
   computed as `Path(__file__).resolve().parent.parent`, which from
   `src/tuning/config.py` resolves to `src/`, not the repo root, so
   `DATA_DIR`/`MODEL_DIR` pointed at nonexistent folders one level too
   deep. It also pointed `DATA_PATH` at `data/Sheet2.csv`, which has never
   existed in this repo - the real file is `data/raw/raw_data.csv`. Moved
   to `src/config.py` (where `train.py` and everything else already
   expected it) and fixed both issues.
3. **Every subpackage's `__init__.py` was empty** (`models/`, `metrics/`,
   `preprocessing/`, `tuning/`, `utils/` - all 0 bytes). Nothing was
   re-exported, so `from src.models import build_svr_pipeline` and
   similar imports in `optuna_tuning.py` and `train.py` failed.
   `tests/*.py` additionally expected flat imports like `from
   src.data_loader import load_dataset` that pointed at modules which
   didn't exist at all (the real code was nested under
   `src/preprocessing/data_loader.py`). Added the missing re-exports in
   each `__init__.py`, plus flat compatibility shims (`src/data_loader.py`,
   `src/features.py`, `src/evaluation.py`, `src/visualization.py`,
   `src/optuna_tuning.py`) so both import styles resolve to the same code.
4. **`src/train/` had no `__init__.py` at all**, so `from src.train import
   main` (which `app/gui.py` relied on) couldn't work even as a namespace
   package.
5. **`tests/README.md` recommended running `pytest` directly**, but that
   never actually worked - only `python -m pytest` did, because `-m`
   invocation adds the current directory to `sys.path` and plain `pytest`
   doesn't. Added `pytest.ini` (`pythonpath = .`) so the documented command
   actually runs the tests.
6. **No model was ever saved.** Neither `Untitled-2.py` nor the
   would-be `src/train/train.py` had a single `joblib.dump`/`pickle` call
   anywhere - every run retrained SVR + GPR + both ensembles from scratch
   and only persisted metrics CSVs and PNG plots. This is the main reason
   "wrap the trained model in an API" wasn't a small task: there was no
   trained model artifact in the repo to wrap. `src/models/persistence.py`
   is the fix; `scripts/train_model.py` now produces
   `models/saved_models/model_bundle.joblib` for the API to load.

With all of that fixed, `python -m pytest` (7 original tests) passes for
the first time, and the pipeline can actually run outside of the GUI.

## What's new

**Model persistence** (`src/models/persistence.py`, `src/models/inference.py`)
Saves the fitted SVR pipeline, GPR pipeline, RidgeCV stacking meta-learner,
ensemble weight, and metrics as one joblib bundle, plus a JSON copy of the
metadata and a CSV copy of the cleaned training data (for the drift
monitor). `inference.py` runs a raw `(Vc, Fz, ap)` point through all four
models via `build_feature_row()`, which reuses the exact same feature-
engineering function used at training time - no separate "inference
feature logic" that training and serving could quietly drift apart from.

**REST API** (`app/main.py`) - FastAPI, replacing `app/gui.py`
- `POST /predict` - one machining-parameter point in, all four model
  outputs back, plus a per-request "is this inside the training envelope"
  check. Every call is logged to `logs/prediction_log.jsonl`.
- `POST /predict/batch` - up to 500 points in one call.
- `GET /health`, `GET /model/info` - for load balancers / MES
  integration monitoring.
- `GET /drift/report` - runs the drift analysis below against everything
  logged so far.
- Interactive docs at `/docs` (Swagger) and `/redoc` for free, via FastAPI.
- Input validation is deliberately two-tier: Pydantic rejects physically
  impossible input (negative/zero/absurd values) with a 422; a request
  with, say, `Vc=500` (positive, just outside the ~7.5-17.5 range the
  model was trained on) is still served, just flagged
  `within_training_envelope: false`. Rejecting out-of-envelope requests
  outright would mean never seeing them, which defeats the point of
  monitoring for drift.

**Drift monitoring** (`monitoring/drift_monitor.py`)
Compares a batch of current `(Vc, Fz, ap)` readings against the training
baseline using PSI (Population Stability Index) and a Kolmogorov-Smirnov
test per feature. Runs standalone (`python -m monitoring.drift_monitor
--from-log`) or via `GET /drift/report`. Exit codes (0/1/2 = stable/
warning/drift) make it usable as a scheduled job that gates or triggers a
retraining pipeline. `--simulate-drift` generates a synthetic "tool wear"
batch (cutting speed creeping up, continuous spread around the training
grid instead of landing exactly on it) to demo the tool without waiting
for real production traffic. See "PSI needs more samples than you'd
think" below for why this isn't just a threshold check.

**Training script** (`scripts/train_model.py`, `src/train/train.py`)
Thin CLI over a `main()` that: loads + validates data, tunes SVR/GPR with
Optuna (now with median pruning and a configurable trial budget/search
space - see below), collects out-of-fold predictions for honest ensemble
evaluation, refits the base models on all available data, and persists
the result. Accepts pre-computed `svr_params`/`gpr_params` to skip
tuning entirely - useful for redeploying a known-good configuration
without paying for a fresh search.

**Docker** - `Dockerfile` + `docker-compose.yml`. Trains offline, serves
an immutable artifact (the image ships whatever's in
`models/saved_models/` at build time rather than training on container
start); mount a volume over `/app/models` to swap in a retrained bundle
without rebuilding. Not build-tested in the sandbox this was developed in
(no Docker daemon available there) - straightforward pip-only
dependencies, but worth a first build/run check on your end.

**Tests** - `test_persistence.py`, `test_inference.py`, `test_api.py`,
`test_drift_monitor.py` added alongside the 7 original (now-passing) tests.

## Design decisions worth knowing about

**Feature set: 12 features, not 15.** `add_engineered_features()` computes
squares, pairwise interactions, pairwise ratios, *and* log-transforms of
Vc/Fz/ap. But `Untitled-2.py` - the only version of this pipeline that
was ever actually run and evaluated - built its final training matrix
from just the first 12 (no logs). `prepare_feature_matrix()` now defaults
to that same 12-feature set, so the model being served matches the one
whose metrics are believable, rather than silently training on a
different feature set nobody evaluated. The log columns are still
computed and available (`MODEL_FEATURE_COLUMNS` in `src/preprocessing/
features.py` is one list edit away from including them) if you want to
experiment.

**Why training defaults to a "quick" preset, and what "full" costs.**
On this dataset, `PolynomialFeatures` inside the SVR pipeline expands the
12 base features to 90 at `poly_degree=2`, 454 at `degree=3`, and 1819 at
`degree=4` - measured fit times of ~1.2s / ~6s / ~25s respectively, on
only 119 training rows. `degree=4` is also solidly into
more-features-than-samples overfitting territory. The default `quick`
preset (25/25/40 trials, `poly_degree` capped at 3) trains in a couple of
minutes; `--preset full` restores the original 60/60/80-trial, degree-4
search space from `configs/training_config.yaml`, which took over two
minutes for just 20 trials in testing - expect it to run considerably
longer. Optuna median pruning (on by default) cuts short trials that are
clearly uncompetitive after 1-2 CV folds rather than always paying for
all 5.

**Why `DRIFT_DETECTED` requires the KS test to agree, not PSI alone.**
This dataset's raw features are a narrow DOE grid (Vc/Fz/ap each take
only ~4-5 discrete values across 119 rows). PSI is a standard,
well-established drift metric, but it turned out to be badly miscalibrated
on a grid this narrow at realistic sample sizes: a batch of 11 samples
resampled from the *exact same* grid (zero real drift, verified by
construction) produced a PSI of 1.4 on Vc - dramatically over the 0.25
"significant shift" threshold - purely because a couple of bins randomly
landed at zero count. The KS test on that same batch correctly reported
p=0.57 (not significant). So the severe `DRIFT_DETECTED` verdict now
requires KS significance *and* a corroborating PSI/out-of-range signal;
the milder `WARNING` tier stays PSI-only on purpose, as a cheap, sensitive
early signal that's allowed to occasionally be noise. `MIN_SAMPLES` (30)
and `LOW_SAMPLE_CAVEAT_THRESHOLD` (50) reflect the same finding - below
~30-50 samples, treat a PSI-only signal skeptically. This was found by
testing the monitor against known-stable data (a plain resample of the
baseline) and checking it correctly stayed quiet, not just against the
"drift" case - it's worth doing the same check before trusting a metric
like this on any new dataset.

**Metrics are reported honestly, not polished up.** Out-of-fold on 119
samples: SVR R²=0.37, GPR R²=0.38, weighted ensemble R²=0.39, stacking
ensemble R²=0.40 (exact numbers will shift slightly if you retrain -
`models/saved_models/model_metadata.json` has the current run's values).
That's a modest fit for a genuinely small dataset with a narrow
experimental design, evaluated properly out-of-sample. Nothing here is
tuned to make that number look better than it is.

## Explicitly not touched

- **`configs/*.yaml`** - never wired to any code (no file in `src/` reads
  them; `src/utils/config_loader.py`'s YAML loader is itself an
  unimplemented stub) and in places describe columns that don't match the
  actual data (`feed_rate`/`spindle_speed` vs. the real `Vc`/`Fz`). Left
  alone rather than guessing at intent for something outside what was asked.
- **`src/preprocessing/{clean,engineer,utils}.py`, `src/models/{ensemble,
  catboost,randomforest,xgboost}.py`, `src/utils/{config_loader,logger}.py`**
  - pre-existing `NotImplementedError` stubs, not on any import path the
  API, training script, or monitor uses. Left as future-extension points.
- **Permutation feature importance** - present in `Untitled-2.py` but
  fragile (string-matching polynomial feature names back to their
  original variable) and not part of what was asked for. A clean
  `/model/feature-importance` endpoint built on `sklearn.inspection.
  permutation_importance` against the final fitted models would be a
  reasonable follow-up.
- **`src/utils/pokayoke.py`** (`validate_input`) - existed but was unused
  anywhere. Wired into `src/train/train.py` as a sanity check (no nulls,
  no negative machining parameters) on the raw loaded data before feature
  engineering.
