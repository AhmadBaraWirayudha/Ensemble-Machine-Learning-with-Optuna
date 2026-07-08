"""
SQLite storage backend for production data: predictions, physical
measurements (e.g. from a roughness tester), and retrain events.

Why SQLite over the JSONL files this replaces, and why not a "real"
time-series database: JSONL got slow and awkward to query as soon as
there was any real cross-referencing to do - specifically, joining a
prediction to the physical measurement taken after the part was machined
(see `measurements` below and `compute_accuracy_report`), which is exactly
the kind of thing a flat append-only file is bad at and an indexed table
is good at. A dedicated time-series database (InfluxDB, TimescaleDB, ...)
would be the standard answer at real production scale, but that means a
second service to run, network config, and a new heavy dependency - all
real operational cost for a single-node service at this project's actual
traffic level. SQLite is a Python standard-library module, is still just
one file (as easy to back up/mount as the JSONL files were), and WAL mode
(enabled below) handles the "one writer, several readers" pattern this
project actually has just fine.

Three tables:
  - predictions:     every prediction the API has served
  - measurements:     physical roughness measurements (from a tester, or
                       any other source), optionally tagged with a job_id
                       to link back to the prediction for that same part
  - retrain_events:   every attempt monitoring/retrain_trigger.py has made
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    job_id TEXT,
    Vc REAL NOT NULL,
    Fz REAL NOT NULL,
    ap REAL NOT NULL,
    svr_prediction REAL,
    gpr_prediction REAL,
    rf_prediction REAL,
    gbm_prediction REAL,
    power_law_prediction REAL,
    weighted_ensemble_prediction REAL,
    stacking_ensemble_prediction REAL,
    recommended_model TEXT,
    recommended_prediction REAL,
    source TEXT DEFAULT 'api'
);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_job_id ON predictions(job_id);

CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    job_id TEXT,
    Ra_measured REAL NOT NULL,
    device TEXT,
    raw_payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_measurements_timestamp ON measurements(timestamp);
CREATE INDEX IF NOT EXISTS idx_measurements_job_id ON measurements(job_id);

CREATE TABLE IF NOT EXISTS retrain_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    drift_verdict TEXT,
    old_rmse REAL,
    new_rmse REAL,
    pct_change REAL,
    n_samples INTEGER
);
CREATE INDEX IF NOT EXISTS idx_retrain_events_timestamp ON retrain_events(timestamp);
"""


@contextmanager
def get_connection(db_path=None):
    """
    Context manager yielding a sqlite3.Connection with the schema already
    applied. Opens and closes a fresh connection per call rather than
    sharing one module-level connection - simpler and safer than manual
    connection pooling given FastAPI can run sync route handlers across a
    small thread pool, and SQLite connections aren't safe to share across
    threads without extra care. WAL mode lets reads (the drift monitor,
    the accuracy report) proceed without blocking on writes (the API
    logging a new prediction).
    """
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------

def log_prediction(vc: float, fz: float, ap: float, prediction: dict, job_id: str = None, source: str = "api", db_path=None) -> None:
    """Record one served prediction. `job_id`, if supplied, is what later
    links this prediction to a physical measurement of the same part -
    see log_measurement()."""

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                timestamp, job_id, Vc, Fz, ap,
                svr_prediction, gpr_prediction, rf_prediction, gbm_prediction, power_law_prediction,
                weighted_ensemble_prediction, stacking_ensemble_prediction,
                recommended_model, recommended_prediction, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(), job_id, vc, fz, ap,
                prediction.get("svr_prediction"), prediction.get("gpr_prediction"),
                prediction.get("rf_prediction"), prediction.get("gbm_prediction"),
                prediction.get("power_law_prediction"),
                prediction.get("weighted_ensemble_prediction"), prediction.get("stacking_ensemble_prediction"),
                prediction.get("recommended_model"), prediction.get("recommended_prediction"),
                source,
            ),
        )


def read_prediction_log(db_path=None) -> pd.DataFrame:
    """Read every logged prediction as a DataFrame. Returns an empty
    DataFrame (not an error) if nothing's been logged yet."""

    with get_connection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id", conn)

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "job_id", "Vc", "Fz", "ap"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# --------------------------------------------------------------------
# Physical measurements
# --------------------------------------------------------------------

def log_measurement(ra_measured: float, job_id: str = None, device: str = None, raw_payload: str = None, db_path=None) -> None:
    """Record one physical roughness measurement (e.g. from a stylus
    tester). Tag it with the same job_id used at prediction time to be
    able to compare the two later - see compute_accuracy_report()."""

    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO measurements (timestamp, job_id, Ra_measured, device, raw_payload) VALUES (?, ?, ?, ?, ?)",
            (_now(), job_id, ra_measured, device, raw_payload),
        )


def read_measurements(db_path=None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM measurements ORDER BY id", conn)

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "job_id", "Ra_measured", "device"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def compute_accuracy_report(db_path=None) -> dict:
    """
    Join predictions to measurements by job_id and compute how accurate
    the model's predictions actually were against ground truth - the real
    test of production accuracy, as opposed to monitoring/drift_monitor.py
    (which only watches whether *inputs* have shifted, not whether
    predictions are actually still correct). Only predictions that have a
    job_id AND a matching measurement contribute; everything else is
    reported as "unmatched" so it's clear how much of the traffic this
    covers.
    """

    predictions = read_prediction_log(db_path)
    measurements = read_measurements(db_path)

    matchable_preds = predictions[predictions["job_id"].notna()]
    matchable_meas = measurements[measurements["job_id"].notna()]

    if matchable_preds.empty or matchable_meas.empty:
        return {
            "status": "insufficient_data",
            "message": "No predictions and measurements share a job_id yet. Tag both with the same job_id to compare.",
            "n_predictions_with_job_id": int(len(matchable_preds)),
            "n_measurements_with_job_id": int(len(matchable_meas)),
        }

    # last measurement per job_id (in case of re-measurement/retries)
    latest_meas = matchable_meas.sort_values("timestamp").groupby("job_id").tail(1)
    merged = matchable_preds.merge(latest_meas, on="job_id", suffixes=("_pred", "_meas"))

    if merged.empty:
        return {
            "status": "insufficient_data",
            "message": "Predictions and measurements exist but no job_id appears in both.",
            "n_predictions_with_job_id": int(len(matchable_preds)),
            "n_measurements_with_job_id": int(len(matchable_meas)),
        }

    merged["residual"] = merged["Ra_measured"] - merged["recommended_prediction"]
    merged["abs_pct_error"] = (merged["residual"].abs() / merged["Ra_measured"].replace(0, pd.NA)) * 100

    import numpy as np
    rmse = float(np.sqrt((merged["residual"] ** 2).mean()))
    mae = float(merged["residual"].abs().mean())
    mape = float(merged["abs_pct_error"].dropna().mean()) if merged["abs_pct_error"].notna().any() else None
    bias = float(merged["residual"].mean())

    return {
        "status": "ok",
        "n_matched_pairs": int(len(merged)),
        "n_predictions_with_job_id": int(len(matchable_preds)),
        "n_measurements_with_job_id": int(len(matchable_meas)),
        "rmse_actual_vs_predicted": round(rmse, 4),
        "mae_actual_vs_predicted": round(mae, 4),
        "mape_pct": round(mape, 2) if mape is not None else None,
        "bias_measured_minus_predicted": round(bias, 4),
        "interpretation": (
            "Positive bias means the model is under-predicting actual "
            "roughness (parts are rougher than expected); negative means "
            "it's over-predicting."
        ),
    }


# --------------------------------------------------------------------
# Retrain events
# --------------------------------------------------------------------

def log_retrain_event(record: dict, db_path=None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO retrain_events (timestamp, action, reason, drift_verdict, old_rmse, new_rmse, pct_change, n_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(), record.get("action"), record.get("reason"), record.get("drift_verdict"),
                record.get("old_rmse"), record.get("new_rmse"), record.get("pct_change"),
                record.get("n_samples"),
            ),
        )


def read_retrain_log(db_path=None) -> list:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM retrain_events ORDER BY id").fetchall()
    return [dict(row) for row in rows]
