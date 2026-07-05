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
- **Data drift monitoring** (PSI + Kolmogorov-Smirnov) - standalone script or `/drift/report` endpoint
- **Automatic retraining** with rollback - detects drift, retrains, and only promotes the result if it's actually better
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
│   └── request_log.py        # shared JSONL prediction log (API writes, monitor reads)
│
├── scripts/
│   ├── train_model.py        # CLI: train + persist a model bundle
│   └── example_client.py     # example of a third-party system calling the API
│
├── data/raw/raw_data.csv
├── models/saved_models/       # model_bundle.joblib, model_metadata.json, training_baseline.csv, feature_importance.json
│   └── archive/                # timestamped backups from monitoring/retrain_trigger.py
├── logs/                       # prediction_log.jsonl, retrain_log.jsonl (created on first use)
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

Requests outside the training envelope (e.g. `Vc=500`) are still served, not rejected - they're flagged (`within_training_envelope: false`) instead, since rejecting them would mean the drift monitor never sees them either. Every served prediction is appended to `logs/prediction_log.jsonl` for the drift monitor to read.

---

# Authentication

Off by default - every endpoint above works with no credential, same as a fresh clone of this repo always has. Set `CNC_API_KEY` to turn on API-key authentication for every data endpoint (`/predict`, `/predict/batch`, `/model/info`, `/model/feature-importance`, `/drift/report`, `/retrain/history`). `/health` never requires a key, so load balancers and container orchestrators can check liveness without one.

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

Before overwriting anything, the current model is backed up to `models/saved_models/archive/<timestamp>/`. After retraining, the new model's out-of-fold RMSE is compared against the old one's; if it's worse by more than `--max-regression-pct` (default 15%), the backup is restored instead of promoting a regression - an automated pipeline silently deploying a worse quality-prediction model is a worse outcome than it doing nothing. Every attempt is logged to `logs/retrain_log.jsonl`, readable via `GET /retrain/history`.

This is a CLI tool meant for a scheduled job (cron, CI pipeline, etc.), not an API endpoint - retraining takes minutes and overwrites the production model, neither of which belongs behind a synchronous HTTP call.

---

# Installation

```bash
git clone https://github.com/AhmadBaraWirayudha/Ensemble-Machine-Learning-with-Optuna.git
cd Ensemble-Machine-Learning-with-Optuna

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

# Running Tests

```bash
pytest
```

68 tests covering data loading, feature engineering, model construction (including the power-law model and replicate-grouping logic), metrics, persistence, inference, the API, authentication, drift detection, and automatic retraining. Most run against real artifacts (the actual training data, an actually-trained model) rather than mocks - train a model first (`python scripts/train_model.py`) if `models/saved_models/` is empty.

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
fitting `Vc`/`Fz`/`ap` alone gets.

Smaller/infrastructure-level items:

- Swap the JSONL prediction log for a real time-series store at higher request volume
- Real-time CNC/IoT integration feeding `monitoring/request_log.py` directly from machine controllers
- CAD/CAM integration

---

# Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML | Scikit-learn |
| Optimization | Optuna |
| API | FastAPI, Uvicorn, Pydantic |
| Monitoring | SciPy (KS test), custom PSI implementation |
| Visualization | Matplotlib |
| Data | Pandas, NumPy |
| Deployment | Docker, docker-compose |
| Testing | Pytest, httpx |

---

# License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

---

# Author

Original ML pipeline: Ahmad Bara Wirayudha (Mechanical Engineering x Machine Learning x Industrial AI)

API microservice + drift monitoring upgrade: see `UPGRADE_NOTES.md`
