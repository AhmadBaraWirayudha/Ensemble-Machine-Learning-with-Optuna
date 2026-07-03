"""
Append-only JSONL log of every prediction the API has served. One JSON
object per line: raw inputs, all four model outputs, and whether the
request fell inside the training envelope.

This is deliberately a flat file rather than a database - it's the
simplest thing that (a) the API can append to on every request without
adding an operational dependency, and (b) the drift monitor can load into
a DataFrame in one line. For a higher-throughput deployment this would be
swapped for a real time-series store, but the read/write functions here
are the only two places that would need to change.
"""

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import PREDICTION_LOG_PATH


def log_prediction(vc: float, fz: float, ap: float, prediction: dict, log_path=None) -> None:
    """Append one served prediction to the JSONL log."""

    log_path = log_path or PREDICTION_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Vc": vc,
        "Fz": fz,
        "ap": ap,
        "svr_prediction": prediction.get("svr_prediction"),
        "gpr_prediction": prediction.get("gpr_prediction"),
        "weighted_ensemble_prediction": prediction.get("weighted_ensemble_prediction"),
        "stacking_ensemble_prediction": prediction.get("stacking_ensemble_prediction"),
        "recommended_model": prediction.get("recommended_model"),
        "recommended_prediction": prediction.get("recommended_prediction"),
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_prediction_log(log_path=None) -> pd.DataFrame:
    """
    Read the JSONL prediction log into a DataFrame. Returns an empty
    DataFrame (not an error) if the log doesn't exist yet or has no
    entries - a service that just started has no traffic yet, and that's
    a normal state, not a failure.
    """

    log_path = log_path or PREDICTION_LOG_PATH

    if not log_path.exists():
        return pd.DataFrame(columns=["timestamp", "Vc", "Fz", "ap"])

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a corrupted line rather than failing the whole read

    if not records:
        return pd.DataFrame(columns=["timestamp", "Vc", "Fz", "ap"])

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df
