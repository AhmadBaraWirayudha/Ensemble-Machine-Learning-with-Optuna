"""
Model persistence.

The original pipeline (Untitled-2.py) trained SVR + GPR + a stacking
meta-learner every single run and never saved them anywhere - only metrics
CSVs and PNG plots survived a run. That made "wrap the trained model in an
API" impossible, since there was no trained model artifact to load. This
module is the missing piece: it saves everything the API needs to serve
predictions without retraining, plus a copy of the cleaned training data
for the drift monitor to use as its reference distribution.
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.config import (
    MODEL_BUNDLE_PATH,
    MODEL_METADATA_PATH,
    BASELINE_DATA_PATH,
    FEATURE_IMPORTANCE_PATH,
)

BUNDLE_FORMAT_VERSION = 2

# Keys in the bundle dict that are fitted model objects rather than
# JSON-serializable metadata - excluded when writing model_metadata.json.
_MODEL_OBJECT_KEYS = ("svr_model", "gpr_model", "rf_model", "gbm_model", "power_law_model", "meta_model")


def save_model_bundle(
    svr_model,
    gpr_model,
    rf_model,
    gbm_model,
    power_law_model,
    meta_model,
    best_alpha,
    feature_columns,
    svr_params,
    gpr_params,
    rf_params,
    gbm_params,
    metrics,
    recommended_model,
    training_df,
    n_train_samples,
    random_state,
    bundle_path=None,
    metadata_path=None,
    baseline_path=None,
):
    """
    Persist a fully-trained ensemble plus its metadata and training
    baseline. Writes three files:

      - model_bundle.joblib   fitted sklearn objects the API loads
      - model_metadata.json   human-readable copy of the metrics/params
      - training_baseline.csv cleaned raw training data, for drift checks

    Format version 2: adds rf_model/gbm_model/power_law_model alongside
    the original svr_model/gpr_model, following the finding (see
    UPGRADE_NOTES.md) that tree-based models substantially outperform
    SVR/GPR on this dataset. meta_model (the stacking ensemble) is fit
    over svr_model + gpr_model + rf_model + gbm_model - power_law_model is
    still saved and served standalone (it's a genuinely useful
    interpretable baseline) but isn't a stacking input; see
    src/train/train.py's STACKED_MODEL_KEYS docstring for why.
    """

    bundle_path = bundle_path or MODEL_BUNDLE_PATH
    metadata_path = metadata_path or MODEL_METADATA_PATH
    baseline_path = baseline_path or BASELINE_DATA_PATH

    trained_at = datetime.now(timezone.utc).isoformat()

    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "svr_model": svr_model,
        "gpr_model": gpr_model,
        "rf_model": rf_model,
        "gbm_model": gbm_model,
        "power_law_model": power_law_model,
        "meta_model": meta_model,
        "best_alpha": float(best_alpha),
        "feature_columns": list(feature_columns),
        "svr_params": svr_params,
        "gpr_params": gpr_params,
        "rf_params": rf_params,
        "gbm_params": gbm_params,
        "metrics": metrics,
        "recommended_model": recommended_model,
        "trained_at": trained_at,
        "n_train_samples": int(n_train_samples),
        "random_state": random_state,
    }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)

    metadata = {k: v for k, v in bundle.items() if k not in _MODEL_OBJECT_KEYS}
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(baseline_path, index=False)

    return bundle_path, metadata_path, baseline_path


def load_model_bundle(bundle_path=None):
    """Load a previously saved model bundle. Raises FileNotFoundError with
    a helpful message if training hasn't been run yet."""

    bundle_path = bundle_path or MODEL_BUNDLE_PATH

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {bundle_path}. "
            "Run `python scripts/train_model.py` first."
        )

    return joblib.load(bundle_path)


def load_metadata(metadata_path=None):
    metadata_path = metadata_path or MODEL_METADATA_PATH

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No model metadata found at {metadata_path}. "
            "Run `python scripts/train_model.py` first."
        )

    with open(metadata_path) as f:
        return json.load(f)


def load_baseline_data(baseline_path=None) -> pd.DataFrame:
    baseline_path = baseline_path or BASELINE_DATA_PATH

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"No training baseline found at {baseline_path}. "
            "Run `python scripts/train_model.py` first."
        )

    return pd.read_csv(baseline_path)


def save_feature_importance(detailed, by_variable, path=None):
    """
    Persist permutation importance results (see
    src/metrics/feature_importance.py) as JSON: {"detailed": {feature:
    {model: value}}, "by_variable": {variable: {model: value}}}.
    """

    path = path or FEATURE_IMPORTANCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "detailed": detailed.to_dict(orient="index"),
        "by_variable": by_variable.to_dict(orient="index"),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def load_feature_importance(path=None) -> dict:
    path = path or FEATURE_IMPORTANCE_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"No feature importance results found at {path}. "
            "Run `python scripts/train_model.py` first."
        )

    with open(path) as f:
        return json.load(f)
