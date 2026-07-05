"""
Shared prediction logic used by the API (and anything else that wants a
prediction from a loaded model bundle). Keeping this out of app/main.py
means the FastAPI layer stays a thin HTTP wrapper and this logic is
independently testable/reusable.
"""

from src.features import build_feature_row

# Must match src.train.train.STACKED_MODEL_KEYS's *order* - the stacking
# meta-learner was fit on a column stack in this exact order, so serving
# has to reproduce it exactly or the meta-learner's weights get applied to
# the wrong model's prediction.
_STACKING_INPUT_ORDER = ["svr_prediction", "gpr_prediction", "rf_prediction", "gbm_prediction"]


def predict_all(bundle, vc: float, fz: float, ap: float) -> dict:
    """
    Run every model in the bundle (SVR, GPR, RandomForest,
    GradientBoosting, PowerLaw, the SVR+GPR weighted ensemble, and the
    SVR+GPR+RF+GBM stacking ensemble) on one raw (Vc, Fz, ap) point and
    return every prediction, plus whichever the training run recommends
    as "the" answer (recommended_prediction).
    """

    row = build_feature_row(vc, fz, ap, bundle["feature_columns"])

    svr_pred = float(bundle["svr_model"].predict(row)[0])

    gpr_pred_arr, gpr_std_arr = bundle["gpr_model"].predict(row, return_std=True)
    gpr_pred = float(gpr_pred_arr[0])
    gpr_std = float(gpr_std_arr[0])

    rf_pred = float(bundle["rf_model"].predict(row)[0])
    gbm_pred = float(bundle["gbm_model"].predict(row)[0])
    power_law_pred = float(bundle["power_law_model"].predict(row)[0])

    alpha = bundle["best_alpha"]
    weighted_pred = alpha * gpr_pred + (1 - alpha) * svr_pred

    stacking_inputs = {"svr_prediction": svr_pred, "gpr_prediction": gpr_pred, "rf_prediction": rf_pred, "gbm_prediction": gbm_pred}
    stacking_row = [[stacking_inputs[name] for name in _STACKING_INPUT_ORDER]]
    stacking_pred = float(bundle["meta_model"].predict(stacking_row)[0])

    predictions_by_name = {
        "SVR": svr_pred,
        "GPR": gpr_pred,
        "RandomForest": rf_pred,
        "GradientBoosting": gbm_pred,
        "PowerLaw": power_law_pred,
        "Weighted_Ensemble": weighted_pred,
        "Stacking_Ensemble": stacking_pred,
    }

    recommended_model = bundle["recommended_model"]
    recommended_value = predictions_by_name[recommended_model]

    return {
        "svr_prediction": svr_pred,
        "gpr_prediction": gpr_pred,
        "gpr_uncertainty_std": gpr_std,
        "rf_prediction": rf_pred,
        "gbm_prediction": gbm_pred,
        "power_law_prediction": power_law_pred,
        "weighted_ensemble_prediction": weighted_pred,
        "ensemble_alpha": float(alpha),
        "stacking_ensemble_prediction": stacking_pred,
        "recommended_model": recommended_model,
        "recommended_prediction": recommended_value,
    }


def check_input_range(baseline_df, vc: float, fz: float, ap: float) -> dict:
    """
    Cheap, immediate, single-request sanity check: is this specific query
    inside the (Vc, Fz, ap) envelope the model was actually trained on?

    This is deliberately simple (min/max bounds) and answers a different
    question than the drift monitor: this flags "the model is being asked
    to extrapolate right now", the drift monitor flags "the incoming
    traffic's distribution has shifted over time". Both matter; neither
    replaces the other.
    """

    bounds = {
        "Vc": (float(baseline_df["Vc"].min()), float(baseline_df["Vc"].max())),
        "Fz": (float(baseline_df["Fz"].min()), float(baseline_df["Fz"].max())),
        "ap": (float(baseline_df["ap"].min()), float(baseline_df["ap"].max())),
    }

    values = {"Vc": vc, "Fz": fz, "ap": ap}

    out_of_range = {}
    for name, value in values.items():
        lo, hi = bounds[name]
        if value < lo or value > hi:
            out_of_range[name] = {"value": value, "trained_min": lo, "trained_max": hi}

    return {
        "within_training_envelope": len(out_of_range) == 0,
        "out_of_range_features": out_of_range,
    }
