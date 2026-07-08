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

**Automatic retraining** (`monitoring/retrain_trigger.py`) - closes the
loop between the two pieces above: checks the drift verdict, and if it's
`DRIFT_DETECTED`, retrains and decides whether to actually deploy the
result. Before overwriting anything, the current bundle is backed up to
`models/saved_models/archive/<timestamp>/`; after retraining, the new
model's out-of-fold RMSE is compared against the old one's recorded RMSE,
and if it's worse by more than `--max-regression-pct` (default 15%), the
backup is restored instead of promoting a regression. An automated
pipeline silently deploying a worse quality-prediction model is a worse
outcome than it doing nothing. Every attempt - skipped, promoted, or
rolled back - is appended to `logs/retrain_log.jsonl`, readable via
`GET /retrain/history`. Deliberately a CLI/scheduled-job tool rather than
an API endpoint: retraining takes minutes and overwrites the production
model, neither of which belongs behind a synchronous HTTP call.
`--dry-run` reports what would happen without doing it;
`--force-retrain` skips the drift check entirely, for a periodic
(e.g. weekly) retrain regardless of detected drift.

**Training script** (`scripts/train_model.py`, `src/train/train.py`)
Thin CLI over a `main()` that: loads + validates data, tunes SVR/GPR with
Optuna (now with median pruning and a configurable trial budget/search
space - see below), collects out-of-fold predictions for honest ensemble
evaluation, refits the base models on all available data, computes
permutation feature importance, and persists all of it. Accepts
pre-computed `svr_params`/`gpr_params` to skip tuning entirely - useful
for redeploying a known-good configuration without paying for a fresh
search.

**Feature importance** (`src/metrics/feature_importance.py`) - the
original prototype computed permutation importance too, but did it
*inside* the polynomial-expanded feature space and then tried to map
auto-generated names like `x3 x7` back to `Vc`/`Fz`/`ap` by string
matching - fragile, and the reason this was initially left as a follow-up
rather than ported over directly. It turned out not to need porting:
`sklearn.inspection.permutation_importance` takes the whole fitted
*pipeline* (poly expansion included) and permutes whatever columns of X
you hand it, so calling it with our 12 named engineered features means
the result is already indexed by clean names, with the internal
polynomial expansion handled transparently. `by_variable` further rolls
those 12 up to the raw `Vc`/`Fz`/`ap` parameter each one derives from (an
interaction term like `Vc_Fz` counts toward both, not split between them
- see the docstring). Computed once at training time, served statically
via `GET /model/feature-importance`. On the current model, `Vc` (cutting
speed) comes out as the most important parameter for both SVR and GPR.

**Docker** - `Dockerfile` + `docker-compose.yml`. Trains offline, serves
an immutable artifact (the image ships whatever's in
`models/saved_models/` at build time rather than training on container
start); mount a volume over `/app/models` to swap in a retrained bundle
without rebuilding. Not build-tested in the sandbox this was developed in
(no Docker daemon available there) - straightforward pip-only
dependencies, but worth a first build/run check on your end.

**Authentication** (`app/auth.py`) - opt-in API key check, off by
default. Set `CNC_API_KEY` and every data endpoint requires a matching
`X-API-Key` header (constant-time comparison via `secrets.compare_digest`,
not a plain `==`); `/health` never requires one, so liveness checks keep
working either way. Off-by-default was a deliberate choice: every test and
example written before this feature existed keeps working unmodified,
rather than needing a key threaded through everywhere. This is a
shared-secret scheme appropriate for a handful of trusted internal
systems (an MES, a QC script) - not a real identity provider for many
distinct external users.

**Tests** - `test_persistence.py`, `test_inference.py`, `test_api.py`,
`test_auth.py`, `test_drift_monitor.py`, `test_feature_importance.py`,
`test_retrain_trigger.py` added alongside the 7 original (now-passing) tests.

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
preset (25 trials each for SVR/GPR/RandomForest/GradientBoosting, 40 for
the ensemble weight, SVR's `poly_degree` capped at 3) trains in a couple
of minutes; `--preset full` restores the original 60/60-trial, degree-4
search space from `configs/training_config.yaml` for SVR/GPR specifically
(RF/GBM trial counts don't change much between presets - tree ensembles
are cheap here, seconds not tens-of-seconds per trial, regardless of
poly_degree concerns that don't apply to them). `full` took over two
minutes for just 20 SVR trials in testing - expect it to run considerably
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

**Metrics are reported honestly, not polished up.** As of the changes in
this section, out-of-fold on 119 samples: R² ranges from 0.19 (the
classical power-law baseline) up to 0.63 (Gradient Boosting / the 4-model
stack) - see "Round 2" immediately below for the full comparison and how
that range was arrived at. `models/saved_models/model_metadata.json` has
the current run's exact values; they'll shift slightly on retraining.
Nothing here is tuned to make any number look better than it is - see the
CV-seed-variance finding below for why any single point estimate should
be read with that in mind.

## Round 2: investigating "how do we increase accuracy" empirically

The changes above got the API and monitoring working against a real,
if modest, model (R² around 0.37-0.40). Asked directly how to improve
accuracy given the dataset/method/algorithm limitations, the useful
answer turned out to require actually interrogating the dataset rather
than reaching for generic advice. In order:

**The dataset has 19 exact-duplicate design points.** It's a complete
5x5x4 factorial DOE (100 unique `(Vc,Fz,ap)` combinations across 119
rows), and 19 of those 100 were measured twice. The within-pair standard
deviation of `Ra` for those 19 replicated points - pure measurement/
process noise, since `Vc`/`Fz`/`ap` are identical within a pair and there's
no input information to distinguish them - averages 0.257, against an
overall `Ra` std of 0.401. That suggested a substantial fraction of total
variance (roughly 70%, by a rough calculation) might be irreducible,
implying a low ceiling on achievable R² no matter the model. **This
estimate turned out to be too pessimistic** - see below - but the
underlying technique (using replicated design points to estimate "pure
error", standard in DOE/RSM analysis) is worth knowing, and the fact that
it's based on only 19 pairs (each just n=2) means it has high sampling
uncertainty of its own and shouldn't be taken as precise.

**Checked whether the CV setup was leaking information through those
replicates.** Plain `StratifiedKFold` doesn't know about the replicate
structure and, at this project's default seed, splits 17 of the 19
replicate pairs across train/test - meaning some "held-out" test points
had an identical-input twin in the training fold. Tested this directly:
re-evaluated SVR/GPR under `StratifiedGroupKFold` (grouping by exact
`(Vc,Fz,ap)` identity, so replicates always land in the same fold)
against the original `StratifiedKFold`, across 5 different CV seeds. The
result was **not what a leakage-inflation hypothesis would predict**: SVR's
R² was consistently *higher* under the leakage-safe grouped CV in every
seed tested, not lower. The clearer and more robust finding from this
experiment was that **R² on this dataset varies a lot** - from 0.24 to
0.50 for the identical model and hyperparameters, changing only the CV
random seed - which on its own says a single point-estimate R² (any of
the ones quoted anywhere in this project, before or after this round of
changes) carries real uncertainty just from having 119 samples.
`StratifiedGroupKFold` is used throughout regardless of that inconclusive
directional result, since it's the methodologically correct choice for a
dataset with known exact replicates independent of which way it happens
to move any particular number - see `create_replicate_groups()` in
`src/tuning/optuna_tuning.py`.

**Found and fixed a real syntax error while looking for other model
options.** `src/models/{randomforest,xgboost,catboost,ensemble}.py`
weren't just unimplemented (`NotImplementedError`) - they had an actual
Python syntax error (an unterminated triple-quoted docstring: `"""Train a
random forest model."` is missing two closing quote characters), so none
of the four could even be imported, let alone run. Fixed the syntax in
all four; implemented `randomforest.py` for real, left `xgboost.py`/
`catboost.py` as clearly-documented stubs (adding those specific
libraries as dependencies wasn't obviously worth it - see below), and
pointed `ensemble.py` at where the actual ensembling logic lives.

**The real finding: model family matters far more than anything else
tried.** Built RandomForest, GradientBoosting (both via scikit-learn - no
new dependencies), and a classical machining power-law model
(`Ra = C * Vc^a * Fz^b * ap^c`, fit by OLS in log-space - the textbook
Taguchi/RSM approach to exactly this problem, notably never tried in this
project before). Compared all five model families on **identical**
`StratifiedGroupKFold` folds for a fair side-by-side:

| Model | R² (out-of-fold) | RMSE |
|---|---|---|
| PowerLaw | 0.19 | 0.360 |
| SVR | 0.39 | 0.311 |
| GPR | 0.41 | 0.308 |
| RandomForest | 0.60 | 0.253 |
| **GradientBoosting** | **0.63** | **0.242** |

Random Forest and especially Gradient Boosting substantially outperform
the SVR/GPR approach the original prototype exclusively used - a jump
from R²~0.40 to R²~0.63, on the exact same 12 engineered features. The
likely reason: this dataset's raw parameters each take only ~4-5 discrete
values (the DOE grid again), which suits trees - which split on
thresholds - more naturally than SVR/GPR's smooth global kernel/polynomial
fit, and trees don't inherit the multicollinearity that comes from
running `PolynomialFeatures` over an already-hand-engineered 12-feature
set. **This is the actual answer to "how do I increase accuracy given
these limitations": add the right model family, not more feature
engineering or a fancier kernel for the models already in use.**

A 4-model stacking ensemble (SVR + GPR + RandomForest + GradientBoosting,
RidgeCV meta-learner) reaches R²=0.63, RMSE=0.242 - matching
GradientBoosting alone almost exactly, meaning the stack isn't adding
much beyond what GBM already captures on its own, but doesn't hurt either.
A 5-model version that also included PowerLaw was tested and rejected: it
added a small, hard-to-interpret *negative* meta-learner weight without
improving stacked RMSE, so PowerLaw is trained, evaluated, and served
standalone (`power_law_prediction` in the API, plus its fitted formula
logged at training time) but excluded from the stacking input - see
`STACKED_MODEL_KEYS` in `src/train/train.py`.

`recommended_model` (used for `/predict`'s `recommended_prediction`) now
picks the best out-of-fold RMSE across all seven candidates - the five
individual models plus both ensembles - rather than assuming it's always
one of the two ensembles. On the currently-shipped model, that's
`GradientBoosting`.

**Why RF/GBM's hyperparameters were tuned with the corrected CV from the
start, while SVR/GPR's weren't re-tuned.** `tune_svr`/`tune_gpr` gained an
optional `groups` parameter (defaults to `None`, preserving old behavior)
so future full retrains use `StratifiedGroupKFold` throughout; the
already-tuned SVR/GPR hyperparameters from before this round weren't
recomputed, since re-tuning is expensive (tens of minutes at the `full`
preset) and the CV-seed-variance finding above suggests the hyperparameters
themselves are unlikely to be highly sensitive to this specific choice.
The final reported metrics, the stacking meta-learner, and all RF/GBM
tuning are all computed under the corrected grouped CV regardless.

**Why XGBoost/CatBoost weren't added despite existing as stub files.**
scikit-learn's `GradientBoostingRegressor` already captures the "boosted
trees" approach and delivered the result above with zero new
dependencies. XGBoost's/CatBoost's usual advantages (large-scale data,
GPU training, native categorical handling) don't apply to 119 rows of
all-continuous features. The stub files now explain this and point at
`build_gbm_pipeline` if someone wants to add one of those libraries later
anyway (e.g. for a much larger future dataset).

**One important caveat that applies to every model here, not just the
new ones.** All hyperparameter tuning (RF/GBM included) selects
hyperparameters using the same CV split it then reports performance on.
This is a mild, well-known source of optimism (the model has been
selected, in part, to do well on exactly the folds being scored) and was
already true of the original SVR/GPR tuning before this round of
changes - a fully nested CV (an outer loop the hyperparameter search
never sees) would remove it but wasn't implemented here, consistent with
the original design. Worth knowing if reporting these numbers externally.

## Round 3: SQLite, and integrating a real CAM tool + a real measurement device

Asked to integrate three concrete things: FreeCAD (CAM), a TIME3233
physical roughness tester, and to swap the JSONL logs mentioned as a
"future improvement" in Round 2. All three turned out to connect: the
storage swap is what makes the other two actually useful together.

**Why SQLite, specifically, and not a "real" time-series database.** The
JSONL files worked fine for "append a record, read them all back later" -
which is all drift monitoring and retrain history ever needed. Linking a
prediction to the physical measurement of the same part is a genuinely
different kind of access (a join, by `job_id`), and a flat append-only
file is the wrong tool for that regardless of format. The standard answer
at real production scale would be a dedicated time-series database
(InfluxDB, TimescaleDB, ...), but that's a second service to run, network
configuration, and a new heavy dependency - real cost for a single-node
service at this project's actual traffic. SQLite is a Python standard
library module (no new dependency at all), is still just one file (as
easy to back up or mount as a Docker volume as the JSONL files were), and
WAL mode (enabled in `monitoring/storage.py`) handles concurrent
reads-while-writing fine at this scale. `monitoring/request_log.py`
becomes a thin re-export shim over the new `monitoring/storage.py` so
existing import sites didn't need to change;
`scripts/migrate_jsonl_to_sqlite.py` is a one-time import for anyone
upgrading from a version of this project that had accumulated real JSONL
history.

**The `job_id` / measurements / accuracy-report design.** `/predict` now
accepts an optional `job_id` (auto-generated if omitted, always echoed
back in the response) and a new `POST /measurements` accepts a physical
reading tagged with the same id. `GET /accuracy/report` joins the two and
reports real RMSE/MAE/MAPE/bias between *predicted* and *measured* Ra -
answering a different and stronger question than
`monitoring/drift_monitor.py` does. Drift monitoring watches whether
*inputs* have shifted from training; the accuracy report checks whether
*predictions* are still actually correct, against ground truth. Nothing
about this required either integration below to exist first - it's
useful the moment anything (a script, a curl command, a human) submits a
measurement - but the two integrations are what make it easy to actually
populate in a real shop.

**TIME3233: verified facts vs. an undocumented protocol, and what was
built around that gap.** Confirmed via product listings: it's a portable
stylus roughness tester (Beijing TIME High Technology), 50mm traversing
length, 800um range, ~55 parameters including Ra, and it transfers to a
PC over RS232 - normally via the vendor's "TIMESurf" software, which can
also export a session to Excel. Not confirmed anywhere: the actual
byte-level serial protocol TIMESurf speaks to the device. Rather than
guess at a specification and present it as working,
`integrations/time3233/reader.py` ships two paths: watching the folder
TIMESurf exports to (recommended - depends only on a documented,
vendor-controlled export format, via a header-detection heuristic robust
enough to find "Ra" under several plausible column-naming schemes without
false-matching things like "Range" or "Parameter" - see
`find_ra_column()` and its tests) and a best-effort direct-serial mode
(common RS232 defaults for this instrument class, a configurable regex,
and a `raw-capture` mode built specifically to let you discover your
actual device's real output format rather than trust an assumed one).
Both were tested for real: the export-watching path end-to-end against a
live API instance (synthetic export file -> ingested -> matched to a real
prediction via job_id -> correct accuracy-report numbers); the
serial-specific code paths only via `pyserial` being importable and the
regex/parsing logic in isolation, since no physical device was available.

**FreeCAD: what's tested vs. what's read defensively.** No FreeCAD
installation was available to test against, so the integration is split
by how confident each half is. `roughness_predictor_core.py` - unit
conversions (`Vc = pi*D*N/1000`, `Fz = feed_rate/(N*flutes)`, checked
against hand calculations and round-tripped through their inverses) and
the FreeCAD-object attribute-extraction logic - has zero FreeCAD
dependency (it's duck-typed attribute traversal, not an import of
`FreeCAD`/`FreeCADGui`) and so is fully unit-tested, including against
mock objects standing in for FreeCAD's Path/ToolController/Tool model.
`FreeCAD_RoughnessPredictor.FCMacro` - the actual Qt panel that runs
inside FreeCAD - could not be run in the environment this was built in.
It's written to degrade gracefully rather than assume it's right: every
property is read via several plausible attribute paths with `hasattr`
guards, and every derived value is shown in an editable field for the
user to review before anything is sent to the API, rather than trusting
extraction blindly. See `integrations/freecad/README.md`.

**New tests.** `test_storage.py` (SQLite predictions/measurements/
retrain-events round-trip, accuracy-report join logic including the
latest-measurement-wins-on-remeasure case), `test_time3233_integration.py`
(Ra-column detection including deliberate false-positive traps like
"Range"/"Parameter", export parsing, the serial regex, mocked HTTP
submission), `test_freecad_integration.py` (unit conversions, the HTTP
client, and the mock-object extraction tests described above) - 70 new
tests, 138 total.

## Explicitly not touched

- **`configs/*.yaml`** - never wired to any code (no file in `src/` reads
  them; `src/utils/config_loader.py`'s YAML loader is itself an
  unimplemented stub) and in places describe columns that don't match the
  actual data (`feed_rate`/`spindle_speed` vs. the real `Vc`/`Fz`). Left
  alone rather than guessing at intent for something outside what was asked.
- **`src/preprocessing/{clean,engineer,utils}.py`, `src/models/{ensemble,
  catboost,xgboost}.py`, `src/utils/{config_loader,logger}.py`** -
  pre-existing `NotImplementedError` stubs (`randomforest.py` used to be
  one too, but is now a real implementation - see "Round 2" above, which
  also covers a syntax error found and fixed across all four `src/models/`
  files listed there). Not on any import path the API, training script, or
  monitor uses. Left as future-extension points.
- **Permutation feature importance was initially left as a follow-up, then
  implemented anyway** - see "What's new" above for why it turned out to
  be straightforward once approached at the pipeline level instead of the
  polynomial-expanded feature level.
- **`src/utils/pokayoke.py`** (`validate_input`) - existed but was unused
  anywhere. Wired into `src/train/train.py` as a sanity check (no nulls,
  no negative machining parameters) on the raw loaded data before feature
  engineering.
