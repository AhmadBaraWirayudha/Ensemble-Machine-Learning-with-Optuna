"""
REST API for the CNC surface-roughness prediction models.

This is the "strip the Tkinter GUI" half of the upgrade: the same trained
SVR + GPR + Random Forest + Gradient Boosting + power-law comparison
(plus weighted/stacking ensembles) that used to only be reachable by
clicking a button in a desktop window is now queryable by any MES/factory
system that can make an HTTP request. See monitoring/ for the other half
(data drift detection and automatic retraining).

Run with:
    uvicorn app.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.models.persistence import load_model_bundle, load_baseline_data, load_metadata, load_feature_importance
from src.models.inference import predict_all, check_input_range
from app.schemas import (
    MachiningParams,
    PredictionResponse,
    RangeCheck,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from app.auth import require_api_key, auth_enabled
from monitoring.request_log import log_prediction
from monitoring.drift_monitor import analyze_drift, read_prediction_log, MIN_SAMPLES
from monitoring.retrain_trigger import read_retrain_log

# Populated at startup by the lifespan handler below. A plain module-level
# dict (rather than app.state) keeps the "is anything loaded" check in
# health_response() trivially simple.
state = {"bundle": None, "baseline": None, "metadata": None, "feature_importance": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["bundle"] = load_model_bundle()
        state["baseline"] = load_baseline_data()
        state["metadata"] = load_metadata()
        state["load_error"] = None
        try:
            state["feature_importance"] = load_feature_importance()
        except FileNotFoundError:
            # Trained with compute_importance=False, or a bundle from
            # before this feature existed - not fatal, /model/
            # feature-importance will just report it's unavailable.
            state["feature_importance"] = None
    except FileNotFoundError as e:
        # Don't crash the process on boot - let it come up, report unhealthy
        # on /health, and give a clear error on endpoints that need a model.
        # A container that crash-loops because training hasn't run yet is a
        # worse failure mode than one that starts and says so plainly.
        state["load_error"] = str(e)
    yield


app = FastAPI(
    title="CNC Surface Roughness Prediction API",
    description=(
        "Predicts surface roughness (Ra) for polycrystalline CNC milling "
        "from cutting speed (Vc), feed per tooth (Fz), and axial depth of "
        "cut (ap). Compares SVR, Gaussian Process, Random Forest, Gradient "
        "Boosting, and a classical power-law model (all Optuna-tuned where "
        "applicable), plus weighted and stacking ensembles, and serves "
        "whichever performed best out-of-fold. Replaces the original "
        "Tkinter desktop tool so other factory/MES systems can query it "
        "directly."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_model():
    if state["bundle"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model is loaded "
                f"({state['load_error'] or 'unknown reason'}). "
                "Run `python scripts/train_model.py` and restart the service."
            ),
        )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    if state["bundle"] is None:
        return HealthResponse(status="unhealthy", model_loaded=False, auth_enabled=auth_enabled())

    return HealthResponse(
        status="ok",
        model_loaded=True,
        recommended_model=state["bundle"]["recommended_model"],
        model_trained_at=state["bundle"]["trained_at"],
        n_train_samples=state["bundle"]["n_train_samples"],
        auth_enabled=auth_enabled(),
    )


@app.get("/model/info", dependencies=[Depends(require_api_key)])
def model_info():
    _require_model()
    return state["metadata"]


@app.get("/model/feature-importance", dependencies=[Depends(require_api_key)])
def feature_importance():
    """
    Permutation importance, computed once at training time (see
    src/metrics/feature_importance.py) and served as-is here rather than
    recomputed per request. `by_variable` rolls the engineered features up
    to the raw Vc/Fz/ap parameter they derive from - start there; `detailed`
    has the per-engineered-feature breakdown.
    """
    _require_model()

    if state["feature_importance"] is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "This model was trained with compute_importance=False (or "
                "predates this endpoint). Retrain without --no-importance "
                "to populate it."
            ),
        )

    return state["feature_importance"]


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict(params: MachiningParams):
    _require_model()

    result = predict_all(state["bundle"], params.Vc, params.Fz, params.ap)
    range_result = check_input_range(state["baseline"], params.Vc, params.Fz, params.ap)

    log_prediction(params.Vc, params.Fz, params.ap, result)

    return PredictionResponse(
        input=params,
        **result,
        range_check=RangeCheck(**range_result),
        model_trained_at=state["bundle"]["trained_at"],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, dependencies=[Depends(require_api_key)])
def predict_batch(request: BatchPredictionRequest):
    _require_model()

    predictions = []
    for params in request.items:
        result = predict_all(state["bundle"], params.Vc, params.Fz, params.ap)
        range_result = check_input_range(state["baseline"], params.Vc, params.Fz, params.ap)
        log_prediction(params.Vc, params.Fz, params.ap, result)

        predictions.append(
            PredictionResponse(
                input=params,
                **result,
                range_check=RangeCheck(**range_result),
                model_trained_at=state["bundle"]["trained_at"],
            )
        )

    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


@app.get("/drift/report", dependencies=[Depends(require_api_key)])
def drift_report(min_samples: int = Query(default=MIN_SAMPLES, ge=1)):
    """
    Runs the same PSI/KS drift analysis as `python -m monitoring.drift_monitor`,
    but against everything this running service has logged via /predict and
    /predict/batch so far, rather than a file you pass in.
    """
    _require_model()

    current_df = read_prediction_log()

    if len(current_df) < min_samples:
        return {
            "status": "insufficient_data",
            "message": (
                f"Only {len(current_df)} predictions logged so far; "
                f"need at least {min_samples} to report on drift. This is "
                "normal for a freshly-deployed service."
            ),
            "n_logged": len(current_df),
        }

    return analyze_drift(state["baseline"], current_df)


@app.get("/retrain/history", dependencies=[Depends(require_api_key)])
def retrain_history(limit: int = Query(default=20, ge=1, le=500)):
    """
    Read-only view of what `python -m monitoring.retrain_trigger` has done
    over time (skipped/promoted/rolled-back, with RMSE before/after).
    Triggering a retrain itself isn't exposed here on purpose - it can take
    minutes and overwrites the production model, which doesn't belong
    behind a synchronous HTTP call. Run it as a scheduled job instead.
    """
    events = read_retrain_log()
    return {"count": len(events), "events": events[-limit:]}
