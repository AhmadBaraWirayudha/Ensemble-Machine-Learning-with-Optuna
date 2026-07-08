# Ensemble Machine Learning with Optuna for Surface Roughness Prediction

An industrial machine learning project for predicting surface roughness (`Ra`) in CNC milling of S45C steel - compares SVR, Gaussian Process Regression, Random Forest, Gradient Boosting, and a classical power-law model, tuned with Optuna, and served as a REST API with data drift monitoring rather than a desktop GUI.

> **New here and wondering why this doesn't look like a small tweak?**
> The original project's `src/` package had never actually run end-to-end
> (broken imports throughout) and never saved a trained model to disk. See
> [`UPGRADE_NOTES.md`](UPGRADE_NOTES.md) for the full list of what was
> broken, what's new, and the reasoning behind the non-obvious choices -
> including a model comparison that took out-of-fold R² from ~0.40 to
> ~0.63 by adding tree-based models the original approach never tried.

---

# Overview

Surface roughness is one of the most important quality indicators in machining processes. Traditional trial-and-error parameter tuning is expensive, time-consuming, and inconsistent.

This project provides a machine learning pipeline that predicts surface roughness from machining parameters:

- Cutting speed (`Vc`)
- Feed per tooth (`Fz`)
- Axial depth of cut (`ap`)

...and serves it as a REST API so any factory/MES system can query it programmatically, with a companion tool that watches for when incoming machining parameters drift away from what the model was trained on (as happens naturally as tool wear progresses).

The modeling approach compares five model families and two ensembling strategies, and serves whichever comes out on top on out-of-fold performance:

- Support Vector Regression (SVR)
- Gaussian Process Regression (GPR)
- Random Forest
- Gradient Boosting
- A classical machining power-law model (`Ra = C * Vc^a * Fz^b * ap^c`, fit in log-space)
- Weighted ensemble learning (Optuna-tuned blend of SVR + GPR)
- Stacking ensemble learning (RidgeCV meta-learner over SVR + GPR + Random Forest + Gradient Boosting)
- Automated hyperparameter tuning using Optuna, with pruning

On the currently-shipped model, **Gradient Boosting alone is the best performer** (out-of-fold R²=0.63) - see `UPGRADE_NOTES.md` for the full comparison and why tree-based models turned out to suit this dataset so much better than the SVR/GPR approach the original project relied on exclusively.

---

# Features

- Industrial machining dataset processing
- Feature engineering (squares, interactions, ratios)
- Five compared model families: SVR, GPR, Random Forest, Gradient Boosting, classical power-law regression
- Optuna hyperparameter optimization with median pruning (SVR, GPR, Random Forest, Gradient Boosting)
- Weighted ensemble (SVR+GPR) + stacking meta-learner (SVR+GPR+RF+GBM) - trained model picks whichever candidate genuinely performs best, individual or ensemble
- Replicate-aware cross-validation (`StratifiedGroupKFold`) - this dataset has 19 exact-duplicate design points that a naive CV split can leak across train/test
- **Model persistence** - training produces a loadable artifact, not just metrics
- **REST API** (FastAPI) with interactive docs, single + batch prediction, health/info endpoints
- **SQLite storage** for predictions, physical measurements, and retrain history - queryable and joinable, not flat log files
- **Data drift monitoring** (PSI + Kolmogorov-Smirnov) - standalone script or `/drift/report` endpoint
- **Automatic retraining** with rollback - detects drift, retrains, and only promotes the result if it's actually better
- **Physical measurement + accuracy tracking** - compare predictions against real measured roughness (e.g. from a stylus tester), not just watch for input drift
- **FreeCAD integration** - predict Ra directly from a Path (CAM) job's tool/feed/speed parameters
- **TIME3233 roughness tester integration** - feed real measurements in via TIMESurf's export or direct serial
- Opt-in API key authentication
- Automated metrics evaluation and plot generation
- Docker + docker-compose for deployment

---

# Project Structure

```text
.
├── README.md
├── UPGRADE_NOTES.md        # what changed and why
├── requirements.txt         # active service deps (training + API + monitoring)
├── Requirements.txt         # original deps list, kept as-is (see legacy/)
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
│
├── src/
│   ├── config.py             # paths, constants (was missing/misplaced before)
│   ├── data_loader.py        # flat-import shim -> preprocessing/data_loader.py
│   ├── features.py           # flat-import shim -> preprocessing/features.py
│   ├── evaluation.py         # flat-import shim -> metrics/evaluation.py
│   ├── visualization.py      # flat-import shim -> metrics/visualization.py
│   ├── optuna_tuning.py      # flat-import shim -> tuning/optuna_tuning.py
│   ├── preprocessing/        # data_loader.py, features.py
│   ├── models/                # models.py (SVR/GPR/GBM pipelines), randomforest.py, power_law.py, persistence.py, inference.py
│   ├── tuning/                 # optuna_tuning.py (SVR/GPR/RF/GBM/ensemble-weight tuning, replicate-aware CV)
│   ├── metrics/                # evaluation.py, visualization.py, feature_importance.py
│   ├── utils/                  # pokayoke.py (input validation)
│   └── train/                   # train.py (main training orchestration)
│
├── app/
│   ├── main.py               # FastAPI service
│   ├── auth.py                # opt-in API key authentication
│   └── schemas.py            # Pydantic request/response models
│
├── monitoring/
│   ├── drift_monitor.py      # PSI/KS drift analysis, CLI + library
│   ├── retrain_trigger.py    # checks drift, retrains + promotes/rolls back
│   ├── storage.py             # SQLite backend: predictions, measurements, retrain events
│   └── request_log.py        # thin re-export shim over storage.py (kept for import compatibility)
│
├── scripts/
│   ├── train_model.py        # CLI: train + persist a model bundle
│   ├── example_client.py     # example of a third-party system calling the API
│   └── migrate_jsonl_to_sqlite.py  # one-time import of any pre-upgrade .jsonl logs
│
├── integrations/
│   ├── time3233/               # TIME3233 roughness tester -> POST /measurements
│   │   ├── reader.py            # TIMESurf-export watcher + best-effort direct serial
│   │   └── README.md            # what's verified vs. not (undocumented serial protocol)
│   └── freecad/                 # FreeCAD Path (CAM) job -> POST /predict
│       ├── roughness_predictor_core.py   # unit conversions + HTTP client (FreeCAD-independent, tested)
│       ├── FreeCAD_RoughnessPredictor.FCMacro  # the actual FreeCAD macro (Qt panel)
│       └── README.md
│
├── data/raw/raw_data.csv
├── models/saved_models/       # model_bundle.joblib, model_metadata.json, training_baseline.csv, feature_importance.json
│   └── archive/                # timestamped backups from monitoring/retrain_trigger.py
├── logs/production.db          # predictions, measurements, retrain events (SQLite; created on first use)
├── reports/{metrics,figures,drift}/
├── configs/                    # not currently wired to any code - see UPGRADE_NOTES.md
├── notebooks/
├── legacy/                     # original Tkinter GUI + monolithic script, kept for reference
└── tests/
```

---

# Quick Start

```bash
pip install -r requirements.txt

# 1. Train and persist a model (a couple of minutes; see UPGRADE_NOTES.md
#    for what the "quick" vs "full" preset trades off)
python scripts/train_model.py

# 2. Serve it
uvicorn app.main:app --reload

# 3. Query it (from another terminal) - or open http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Vc": 12.5, "Fz": 0.1, "ap": 1.0}'

# or, using the example client:
python scripts/example_client.py
```

```bash
# Or with Docker, once a model has been trained (step 1 above):
docker compose up --build
```

---

# Machine Learning Pipeline

## 1. Data Loading

`src/preprocessing/data_loader.py` loads `data/raw/raw_data.csv` and cleans it: strips whitespace from column names, renames the source file's `Ax` column to `ap`, coerces to numeric, and drops invalid rows. 119 clean samples survive.

| Input | Description |
|---|---|
| `Vc` | Cutting speed |
| `Fz` | Feed per tooth |
| `ap` | Axial depth of cut |

| Target | Description |
|---|---|
| `Ra` | Surface roughness |

## 2. Feature Engineering

`src/preprocessing/features.py` generates squared terms, pairwise interactions, pairwise ratios, and log transforms (`Vc²`, `Vc × Fz`, `Vc / ap`, `log(Vc)`, ...). The model is trained on the first 12 of these (everything except the three log features) - see `UPGRADE_NOTES.md` for why.

## 3. Model Development

**SVR pipeline:** PolynomialFeatures -> RobustScaler -> PowerTransformer -> RBF-kernel SVR.
**GPR pipeline:** PolynomialFeatures -> RobustScaler -> PowerTransformer -> RBF + RationalQuadratic + WhiteKernel, combined.
**Random Forest / Gradient Boosting:** the 12 engineered features directly, no polynomial expansion - trees split on raw feature values and capture nonlinearity/interactions natively, so an explicit polynomial expansion in front of them would only add thousands of collinear columns without giving them any new capability (unlike SVR/GPR, which need it to represent nonlinearity at all).
**Power-law model:** `Ra = C * Vc^a * Fz^b * ap^c`, fit by ordinary least squares on `log(Ra)` vs. `log(Vc), log(Fz), log(ap)` - the classical Taguchi/RSM approach to this exact problem. Only 4 parameters, so essentially no overfitting risk, and directly interpretable: the fitted exponents say how sensitive `Ra` is to each parameter.

## 4. Hyperparameter Optimization

Optuna (TPE sampler + median pruning) tunes:

- **SVR:** `C`, `epsilon`, `gamma`, `poly_degree`
- **GPR:** `amplitude`, `length_scale`, `alpha`, `noise`
- **Random Forest:** `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- **Gradient Boosting:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `min_samples_leaf`

The power-law model has no hyperparameters (closed-form OLS).

Cross-validation uses `StratifiedGroupKFold`, grouping by exact `(Vc,Fz,ap)` identity: this dataset is a complete 5x5x4 factorial design where 19 of the 100 unique parameter combinations were measured twice, and a plain stratified split doesn't know that (see `UPGRADE_NOTES.md` for what happens if it isn't accounted for).

## 5. Ensemble Learning

**Weighted ensemble:** `prediction = alpha * GPR + (1 - alpha) * SVR`, with `alpha` tuned by Optuna against out-of-fold predictions. Kept for continuity/comparison with the original two-model design.
**Stacking ensemble:** a RidgeCV meta-learner over `[SVR_pred, GPR_pred, RandomForest_pred, GradientBoosting_pred]`, fit on out-of-fold predictions to avoid leakage. The power-law model is evaluated and served standalone but deliberately excluded from this stack - see `UPGRADE_NOTES.md` for why (it added a small, hard-to-interpret negative weight without improving stacked RMSE).

Training picks whichever of all seven candidates - five individual models plus both ensembles - has the best out-of-fold RMSE, and records it as `recommended_model` in the saved bundle. The API surfaces this as `recommended_prediction` so a caller who just wants one number doesn't have to choose. On the currently-shipped model, that's Gradient Boosting.

## 6. Model Evaluation

Metrics: MSE, RMSE, MAE, MAPE, MBE, R². All computed from **out-of-fold** predictions (`StratifiedGroupKFold`, 5 folds over target quantile bins, replicates never split across train/test), not in-sample fit:

| Model | R² (out-of-fold) | RMSE |
|---|---|---|
| Power-law | 0.19 | 0.360 |
| SVR | 0.39 | 0.311 |
| GPR | 0.41 | 0.308 |
| Weighted Ensemble | 0.44 | 0.300 |
| Random Forest | 0.60 | 0.253 |
| Stacking Ensemble | 0.63 | 0.242 |
| **Gradient Boosting** | **0.63** | **0.242** |

Exact numbers shift slightly on retraining - see `models/saved_models/model_metadata.json` for the current run's values, and `UPGRADE_NOTES.md` for the full story behind this comparison (including why R² on a 119-sample dataset shouldn't be read as a precise number - it swung from 0.24 to 0.50 for one model just from changing the CV random seed).

## 7. Feature Importance

Permutation importance (`sklearn.inspection.permutation_importance`) run on the final fitted pipelines - since it operates on the whole pipeline, the internal `PolynomialFeatures` expansion (where present) is handled transparently and the result comes back indexed by the 12 named engineered features directly, no reverse-mapping of auto-generated names required. `by_variable` further rolls those up to whichever of `Vc`/`Fz`/`ap` each one derives from. Served via `GET /model/feature-importance`; on the current model, `Vc` (cutting speed) is the most important parameter across all four models it's computed for (SVR, GPR, Random Forest, Gradient Boosting).

---

# REST API

```bash
uvicorn app.main:app --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/predict` | POST | required if set | One `{Vc, Fz, ap}` in, every model's prediction + range check back |
| `/predict/batch` | POST | required if set | Up to 500 points in one call |
| `/measurements` | POST | required if set | Record a physical roughness measurement (e.g. from a stylus tester), optionally tagged with a `job_id` |
| `/accuracy/report` | GET | required if set | Predicted-vs-actual accuracy, joining predictions to measurements by `job_id` |
| `/health` | GET | never | Liveness + whether a model is loaded |
| `/model/info` | GET | required if set | Metrics, hyperparameters, training timestamp |
| `/model/feature-importance` | GET | required if set | Permutation importance, per engineered feature and rolled up to Vc/Fz/ap |
| `/drift/report` | GET | required if set | Drift analysis over everything logged so far |
| `/retrain/history` | GET | required if set | Log of past automatic retrain attempts (skipped/promoted/rolled-back) |

"required if set" = required only if `CNC_API_KEY` is set on the deployment (see Authentication below); by default it isn't, and every endpoint is open.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Vc": 12.5, "Fz": 0.1, "ap": 1.0}'
```

```json
{
  "input": {"Vc": 12.5, "Fz": 0.1, "ap": 1.0},
  "job_id": "f8c65a3c-a0cf-44e5-9026-7d91595f337e",
  "svr_prediction": 1.17,
  "gpr_prediction": 1.25,
  "gpr_uncertainty_std": 0.31,
  "rf_prediction": 1.31,
  "gbm_prediction": 1.52,
  "power_law_prediction": 1.09,
  "weighted_ensemble_prediction": 1.21,
  "stacking_ensemble_prediction": 1.42,
  "recommended_model": "GradientBoosting",
  "recommended_prediction": 1.52,
  "range_check": {"within_training_envelope": true, "out_of_range_features": {}},
  "model_trained_at": "2026-07-04T15:04:51+00:00"
}
```

Requests outside the training envelope (e.g. `Vc=500`) are still served, not rejected - they're flagged (`within_training_envelope: false`) instead, since rejecting them would mean the drift monitor never sees them either. Every served prediction is recorded in `logs/production.db` for the drift monitor and accuracy report to read; pass `job_id` explicitly (or use the auto-generated one, echoed back in the response) to link it to a later physical measurement - see Physical Measurement & Accuracy Tracking below.

---

# Authentication

Off by default - every endpoint above works with no credential, same as a fresh clone of this repo always has. Set `CNC_API_KEY` to turn on API-key authentication for every data endpoint (`/predict`, `/predict/batch`, `/measurements`, `/accuracy/report`, `/model/info`, `/model/feature-importance`, `/drift/report`, `/retrain/history`). `/health` never requires a key, so load balancers and container orchestrators can check liveness without one.

```bash
export CNC_API_KEY="some-long-random-string"
uvicorn app.main:app

curl -H "X-API-Key: some-long-random-string" -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d '{"Vc": 12.5, "Fz": 0.1, "ap": 1.0}'
```

```bash
python scripts/example_client.py --api-key some-long-random-string
```

This is a shared-secret scheme (one key, checked with a constant-time comparison), appropriate for a service reached by a small number of trusted internal systems - not a substitute for a real identity provider if this is ever exposed to many distinct external users who each need their own revocable credential. See `app/auth.py`.

---

# Data Drift Monitoring

Training data here is a narrow experimental grid (`Vc`/`Fz`/`ap` each take only ~4-5 discrete values). Production readings won't land on that grid, and as tool wear progresses, operators compensate by adjusting speeds/feeds - so the real operating envelope gradually walks away from what the model saw during training. This is what `monitoring/drift_monitor.py` watches for.

```bash
# Compare against everything the API has logged so far
python -m monitoring.drift_monitor --from-log

# Compare against a CSV of new production readings
python -m monitoring.drift_monitor --input production_batch.csv

# No production traffic yet? Generate a synthetic "tool wear" batch:
python -m monitoring.drift_monitor --simulate-drift
```

Each feature gets a PSI (Population Stability Index) and a two-sample Kolmogorov-Smirnov test against the training baseline. Exit codes (`0`/`1`/`2` = stable/warning/drift) make this usable directly as a scheduled job that gates or triggers retraining. The same analysis is available live via `GET /drift/report`.

See `UPGRADE_NOTES.md` for why the severe "drift detected" verdict specifically requires the KS test to agree rather than trusting PSI alone - on this dataset's narrow grid, PSI by itself produced a dramatic false positive during testing.

---

# Automatic Retraining

`monitoring/retrain_trigger.py` closes the loop: it checks the drift verdict above, and if it's `DRIFT_DETECTED`, retrains and decides whether to actually deploy the result.

```bash
# Check drift; retrain only if needed
python -m monitoring.retrain_trigger

# See what it would do without doing it
python -m monitoring.retrain_trigger --dry-run

# Retrain regardless of drift (e.g. a periodic scheduled job)
python -m monitoring.retrain_trigger --force-retrain
```

Before overwriting anything, the current model is backed up to `models/saved_models/archive/<timestamp>/`. After retraining, the new model's out-of-fold RMSE is compared against the old one's; if it's worse by more than `--max-regression-pct` (default 15%), the backup is restored instead of promoting a regression - an automated pipeline silently deploying a worse quality-prediction model is a worse outcome than it doing nothing. Every attempt is recorded in `logs/production.db`, readable via `GET /retrain/history`.

This is a CLI tool meant for a scheduled job (cron, CI pipeline, etc.), not an API endpoint - retraining takes minutes and overwrites the production model, neither of which belongs behind a synchronous HTTP call.

---

# Physical Measurement & Accuracy Tracking

Everything above (drift monitoring, auto-retrain) watches whether *inputs* have shifted from training. It's a genuinely different, stronger question whether the model's *predictions* are still actually correct - and the only way to answer that is against ground truth: a real measurement of the part that was actually machined.

Tag a prediction with a `job_id` (supplied or auto-generated - see the `/predict` example above), then after the part is machined and measured, submit the reading tagged with the same `job_id`:

```bash
curl -X POST http://127.0.0.1:8000/predict -d '{"Vc": 12.5, "Fz": 0.1, "ap": 1.0, "job_id": "PART-42"}'
# ... machine the part, measure it (see integrations/time3233/ below) ...
curl -X POST http://127.0.0.1:8000/measurements \
  -d '{"Ra_measured": 0.65, "job_id": "PART-42", "device": "TIME3233"}'
curl http://127.0.0.1:8000/accuracy/report
```

`GET /accuracy/report` joins every prediction to its matching measurement (by `job_id`; the latest measurement wins if a part was re-measured) and reports RMSE/MAE/MAPE and bias between what was predicted and what was actually measured. Both are stored in `logs/production.db` (see `monitoring/storage.py`) - a real SQLite database rather than the flat JSONL files this project used before, specifically so this join is a normal indexed query instead of something a flat append-only log can't really do. A dedicated time-series database would be the standard answer at real production scale; SQLite (a Python stdlib module, still one file, WAL mode for concurrent reads/writes) is the right-sized choice at this project's actual scale - see `UPGRADE_NOTES.md`.

---

# External Integrations

**`integrations/freecad/`** - predicts Ra directly from a FreeCAD Path (CAM) job: pick an operation, the macro derives `Vc`/`Fz`/`ap` from the tool diameter, spindle RPM, feed rate, and flute count you've already set up, shows them in editable fields for review, and calls the API. Log a measurement back in the same panel after machining. See `integrations/freecad/README.md` - importantly, for what's actually been verified given this was built without a FreeCAD installation to test against (the unit-conversion math is fully unit-tested; the exact FreeCAD property names are read defensively with editable fallbacks, not assumed).

**`integrations/time3233/`** - feeds real measurements from a TIME3233 portable stylus roughness tester into `/measurements`. Two paths: watching the folder the vendor's own "TIMESurf" software exports to (recommended - doesn't depend on reverse-engineering anything), or best-effort direct serial (the exact protocol isn't publicly documented, so this needs verification against your actual device - see `integrations/time3233/README.md`, including a `raw-capture` mode built specifically to help with that verification).

---

# Installation

```bash
git clone https://github.com/AhmadBaraWirayudha/Ensemble-Machine-Learning-with-Optuna.git
cd Ensemble-Machine-Learning-with-Optuna

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`pyserial` is commented out in `requirements.txt` - it's only needed for `integrations/time3233/reader.py`'s direct-serial mode (not its recommended TIMESurf-export-watching mode, and not needed anywhere else). Install it separately if you need that: `pip install pyserial`.

# Running Tests

```bash
pytest
```

138 tests covering data loading, feature engineering, model construction (including the power-law model and replicate-grouping logic), metrics, persistence, inference, the API, authentication, drift detection, automatic retraining, SQLite storage, and the FreeCAD/TIME3233 integrations. Most run against real artifacts (the actual training data, an actually-trained model) rather than mocks - train a model first (`python scripts/train_model.py`) if `models/saved_models/` is empty.

---

# Example Outputs

`python scripts/train_model.py` writes:

```text
reports/metrics/metrics_report.csv
reports/figures/*.png              # target distribution, actual-vs-predicted, model comparison, feature importance
reports/feature_importance/*.csv   # detailed (12 features) and by-variable (Vc/Fz/ap) breakdowns
models/saved_models/model_bundle.joblib
models/saved_models/model_metadata.json
models/saved_models/training_baseline.csv
models/saved_models/feature_importance.json
```

`python -m monitoring.drift_monitor` writes a timestamped JSON report to `reports/drift/`.

---

# Industrial Relevance

- CNC machining optimization
- Manufacturing quality prediction
- Smart manufacturing / MES integration
- Predictive quality control
- MLOps for small-data industrial models (drift detection, model versioning via metadata)

---

# Future Improvements

**The highest-leverage one, from here:** more data. Adding Random Forest/
Gradient Boosting took out-of-fold R² from ~0.40 to ~0.63 (see
`UPGRADE_NOTES.md`), but 119 samples across only 3 input variables is
still a hard ceiling on what any model can do. The two concrete versions
of "more data": (1) more experimental runs, especially more replicate
measurements at repeated `(Vc,Fz,ap)` points - this dataset has only 19,
and a larger set would pin down the true noise floor with much less
uncertainty than the current estimate; (2) capturing input variables this
dataset doesn't have at all - tool wear state, coolant flow, tool
geometry, machine vibration - which plausibly explain a real share of
what's currently unexplained variance, no matter how good the model
fitting `Vc`/`Fz`/`ap` alone gets. The infrastructure to actually collect
option (1) at production scale now exists - `/measurements` and
`/accuracy/report` (see Physical Measurement & Accuracy Tracking above)
turn every predict-then-measure cycle into a labeled data point - but
using it to retrain on *measured* Ra rather than the original static
dataset isn't wired up yet; `scripts/train_model.py` still trains only
from `data/raw/raw_data.csv`.

Smaller/infrastructure-level items:

- Verify `integrations/time3233/`'s serial protocol and `integrations/freecad/`'s exact property names against real hardware/FreeCAD versions (both were built without either available - see their READMEs for exactly what's unverified)
- Feed accumulated `/measurements` data back into training, not just the accuracy report
- A real identity provider (OAuth2/similar) if this ever needs many distinct external users rather than a handful of trusted internal systems (current auth is a single shared API key - see Authentication)
- A proper time-series database if request volume ever outgrows SQLite (see Physical Measurement & Accuracy Tracking above for why SQLite was the right-sized choice for now)

---

# Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML | Scikit-learn |
| Optimization | Optuna |
| API | FastAPI, Uvicorn, Pydantic |
| Storage | SQLite (predictions, measurements, retrain events) |
| Monitoring | SciPy (KS test), custom PSI implementation |
| Visualization | Matplotlib |
| Data | Pandas, NumPy |
| Integrations | FreeCAD Path/Qt (macro), pyserial (optional, TIME3233 direct serial), openpyxl (TIMESurf Excel export) |
| Deployment | Docker, docker-compose |
| Testing | Pytest, httpx, responses |

---

# License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

---

# Author

Original ML pipeline: Ahmad Bara Wirayudha (Mechanical Engineering x Machine Learning x Industrial AI)

API microservice + drift monitoring upgrade: see `UPGRADE_NOTES.md`
