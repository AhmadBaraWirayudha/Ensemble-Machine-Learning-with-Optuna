# Ensemble Machine Learning with Optuna for Surface Roughness Prediction

An industrial machine learning project for predicting surface roughness (`Ra`) in CNC milling of S45C steel using ensemble learning, Gaussian Process Regression (GPR), Support Vector Regression (SVR), and Optuna-based hyperparameter optimization - served as a REST API with data drift monitoring, rather than a desktop GUI.

> **New here and wondering why this doesn't look like a small tweak?**
> The original project's `src/` package had never actually run end-to-end
> (broken imports throughout) and never saved a trained model to disk. See
> [`UPGRADE_NOTES.md`](UPGRADE_NOTES.md) for the full list of what was
> broken, what's new, and the reasoning behind the non-obvious choices.

---

# Overview

Surface roughness is one of the most important quality indicators in machining processes. Traditional trial-and-error parameter tuning is expensive, time-consuming, and inconsistent.

This project provides a machine learning pipeline that predicts surface roughness from machining parameters:

- Cutting speed (`Vc`)
- Feed per tooth (`Fz`)
- Axial depth of cut (`ap`)

...and serves it as a REST API so any factory/MES system can query it programmatically, with a companion tool that watches for when incoming machining parameters drift away from what the model was trained on (as happens naturally as tool wear progresses).

The modeling approach combines:

- Support Vector Regression (SVR)
- Gaussian Process Regression (GPR)
- Weighted ensemble learning (Optuna-tuned blend weight)
- Stacking ensemble learning (RidgeCV meta-learner)
- Automated hyperparameter tuning using Optuna, with pruning

---

# Features

- Industrial machining dataset processing
- Feature engineering (squares, interactions, ratios)
- SVR with polynomial feature expansion
- Gaussian Process Regression with custom kernels
- Optuna hyperparameter optimization with median pruning
- Weighted ensemble optimization + stacking meta-learner
- **Model persistence** - training produces a loadable artifact, not just metrics
- **REST API** (FastAPI) with interactive docs, single + batch prediction, health/info endpoints
- **Data drift monitoring** (PSI + Kolmogorov-Smirnov) - standalone script or `/drift/report` endpoint
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
│   ├── models/                # models.py (SVR/GPR pipelines), persistence.py, inference.py
│   ├── tuning/                 # optuna_tuning.py (SVR/GPR/ensemble-weight tuning)
│   ├── metrics/                # evaluation.py, visualization.py
│   ├── utils/                  # pokayoke.py (input validation)
│   └── train/                   # train.py (main training orchestration)
│
├── app/
│   ├── main.py               # FastAPI service
│   └── schemas.py            # Pydantic request/response models
│
├── monitoring/
│   ├── drift_monitor.py      # PSI/KS drift analysis, CLI + library
│   └── request_log.py        # shared JSONL prediction log (API writes, monitor reads)
│
├── scripts/
│   ├── train_model.py        # CLI: train + persist a model bundle
│   └── example_client.py     # example of a third-party system calling the API
│
├── data/raw/raw_data.csv
├── models/saved_models/       # model_bundle.joblib, model_metadata.json, training_baseline.csv
├── logs/                       # prediction_log.jsonl (created on first API request)
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
**GPR pipeline:** RBF + RationalQuadratic + WhiteKernel, combined.

## 4. Hyperparameter Optimization

Optuna (TPE sampler + median pruning) tunes:

- **SVR:** `C`, `epsilon`, `gamma`, `poly_degree`
- **GPR:** `amplitude`, `length_scale`, `alpha`, `noise`

## 5. Ensemble Learning

**Weighted ensemble:** `prediction = alpha * GPR + (1 - alpha) * SVR`, with `alpha` tuned by Optuna against out-of-fold predictions.
**Stacking ensemble:** a RidgeCV meta-learner over `[SVR_pred, GPR_pred]`, also fit on out-of-fold predictions to avoid leakage.

Training picks whichever of the two has the better out-of-fold RMSE and records it as `recommended_model` in the saved bundle - the API surfaces this as `recommended_prediction` so a caller who just wants one number doesn't have to choose.

## 6. Model Evaluation

Metrics: MSE, RMSE, MAE, MAPE, MBE, R². All computed from **out-of-fold** predictions (stratified 5-fold CV over target quantile bins), not in-sample fit - see `models/saved_models/model_metadata.json` after training for the current run's numbers.

---

# REST API

```bash
uvicorn app.main:app --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | One `{Vc, Fz, ap}` in, all four model outputs + range check back |
| `/predict/batch` | POST | Up to 500 points in one call |
| `/health` | GET | Liveness + whether a model is loaded |
| `/model/info` | GET | Metrics, hyperparameters, training timestamp |
| `/drift/report` | GET | Drift analysis over everything logged so far |

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
  "weighted_ensemble_prediction": 1.21,
  "stacking_ensemble_prediction": 1.23,
  "recommended_model": "Stacking_Ensemble",
  "recommended_prediction": 1.23,
  "range_check": {"within_training_envelope": true, "out_of_range_features": {}},
  "model_trained_at": "2026-07-02T06:47:08+00:00"
}
```

Requests outside the training envelope (e.g. `Vc=500`) are still served, not rejected - they're flagged (`within_training_envelope: false`) instead, since rejecting them would mean the drift monitor never sees them either. Every served prediction is appended to `logs/prediction_log.jsonl` for the drift monitor to read.

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

36 tests covering data loading, feature engineering, model construction, metrics, persistence, inference, the API, and drift detection. Most run against real artifacts (the actual training data, an actually-trained model) rather than mocks - train a model first (`python scripts/train_model.py`) if `models/saved_models/` is empty.

---

# Example Outputs

`python scripts/train_model.py` writes:

```text
reports/metrics/metrics_report.csv
reports/figures/*.png              # target distribution, actual-vs-predicted, model comparison
models/saved_models/model_bundle.joblib
models/saved_models/model_metadata.json
models/saved_models/training_baseline.csv
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

- Authentication on the API (currently open, suitable for a trusted internal network)
- Swap the JSONL prediction log for a real time-series store at higher request volume
- A `/model/feature-importance` endpoint (permutation importance existed in the original prototype but wasn't reimplemented here - see `UPGRADE_NOTES.md`)
- Automatic retraining triggered from `monitoring/drift_monitor.py`'s exit code
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
