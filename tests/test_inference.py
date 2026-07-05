import pandas as pd
import pytest

from src.models.persistence import load_model_bundle, load_baseline_data
from src.models.inference import predict_all, check_input_range


@pytest.fixture(scope="module")
def bundle():
    return load_model_bundle()


@pytest.fixture(scope="module")
def baseline():
    return load_baseline_data()


def test_predict_all_returns_all_models(bundle):
    result = predict_all(bundle, vc=10.0, fz=0.1, ap=1.0)

    for key in [
        "svr_prediction", "gpr_prediction", "gpr_uncertainty_std",
        "rf_prediction", "gbm_prediction", "power_law_prediction",
        "weighted_ensemble_prediction", "stacking_ensemble_prediction",
        "recommended_model", "recommended_prediction",
    ]:
        assert key in result

    # Ra (surface roughness) is physically non-negative; predictions from a
    # model trained on data in [0.1, 1.89] should land in a plausible band,
    # not blow up or go negative for an in-envelope input.
    assert -0.5 < result["recommended_prediction"] < 3.0


def test_predict_all_is_deterministic(bundle):
    r1 = predict_all(bundle, vc=12.5, fz=0.1, ap=1.25)
    r2 = predict_all(bundle, vc=12.5, fz=0.1, ap=1.25)
    assert r1 == r2


def test_recommended_prediction_matches_recommended_model(bundle):
    result = predict_all(bundle, vc=10.0, fz=0.1, ap=1.0)

    key_by_name = {
        "SVR": "svr_prediction",
        "GPR": "gpr_prediction",
        "RandomForest": "rf_prediction",
        "GradientBoosting": "gbm_prediction",
        "PowerLaw": "power_law_prediction",
        "Weighted_Ensemble": "weighted_ensemble_prediction",
        "Stacking_Ensemble": "stacking_ensemble_prediction",
    }
    expected_key = key_by_name[result["recommended_model"]]
    assert result["recommended_prediction"] == result[expected_key]


def test_check_input_range_in_envelope(baseline):
    result = check_input_range(baseline, vc=10.0, fz=0.1, ap=1.0)
    assert result["within_training_envelope"] is True
    assert result["out_of_range_features"] == {}


def test_check_input_range_flags_out_of_range(baseline):
    result = check_input_range(baseline, vc=999.0, fz=0.1, ap=1.0)
    assert result["within_training_envelope"] is False
    assert "Vc" in result["out_of_range_features"]
    assert "Fz" not in result["out_of_range_features"]
