"""
These tests assume `python scripts/train_model.py` has already been run at
least once (same assumption test_data_loader.py etc. make about the raw
CSV being present - this suite tests against real artifacts rather than
mocks, consistent with the rest of this test directory).
"""

import numpy as np
import pandas as pd

from src.models.persistence import (
    load_model_bundle,
    load_metadata,
    load_baseline_data,
    save_model_bundle,
    save_feature_importance,
    load_feature_importance,
)


def test_load_model_bundle_has_expected_keys():
    bundle = load_model_bundle()

    for key in [
        "svr_model", "gpr_model", "rf_model", "gbm_model", "power_law_model",
        "meta_model", "best_alpha", "feature_columns", "recommended_model",
        "trained_at", "n_train_samples",
    ]:
        assert key in bundle


def test_load_metadata_excludes_model_objects():
    metadata = load_metadata()

    # the JSON copy should be readable without unpickling sklearn objects
    for key in ["svr_model", "gpr_model", "rf_model", "gbm_model", "power_law_model", "meta_model"]:
        assert key not in metadata

    assert "metrics" in metadata
    assert "recommended_model" in metadata


def test_load_baseline_data_matches_training_columns():
    baseline = load_baseline_data()

    assert list(baseline.columns) == ["Vc", "Fz", "ap", "Ra"]
    assert len(baseline) > 0
    assert baseline.isnull().sum().sum() == 0


def test_save_and_load_round_trip(tmp_path):
    from sklearn.svm import SVR
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from src.models.power_law import PowerLawRegressor

    bundle_path = tmp_path / "bundle.joblib"
    metadata_path = tmp_path / "metadata.json"
    baseline_path = tmp_path / "baseline.csv"

    dummy_df = pd.DataFrame({"Vc": [10.0, 12.5], "Fz": [0.1, 0.1], "ap": [1.0, 1.0], "Ra": [0.5, 0.6]})
    X_dummy = [[10.0, 0.1, 1.0], [12.5, 0.1, 1.0]]
    y_dummy = [0.5, 0.6]

    save_model_bundle(
        svr_model=SVR().fit(X_dummy, y_dummy),
        gpr_model=GaussianProcessRegressor().fit(X_dummy, y_dummy),
        rf_model=RandomForestRegressor(n_estimators=5).fit(X_dummy, y_dummy),
        gbm_model=GradientBoostingRegressor(n_estimators=5).fit(X_dummy, y_dummy),
        power_law_model=PowerLawRegressor().fit(X_dummy, y_dummy),
        meta_model=RidgeCV().fit([[0, 0, 0, 0], [1, 1, 1, 1], [0.5, 0.4, 0.6, 0.5]], [0, 1, 0.5]),
        best_alpha=0.5,
        feature_columns=["Vc", "Fz", "ap"],
        svr_params={"C": 1.0},
        gpr_params={"alpha": 1.0},
        rf_params={"n_estimators": 5},
        gbm_params={"n_estimators": 5},
        metrics={"SVR": {"RMSE": 0.1}},
        recommended_model="SVR",
        training_df=dummy_df,
        n_train_samples=2,
        random_state=42,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        baseline_path=baseline_path,
    )

    assert bundle_path.exists()
    assert metadata_path.exists()
    assert baseline_path.exists()

    reloaded = load_model_bundle(bundle_path)
    assert reloaded["best_alpha"] == 0.5
    assert reloaded["feature_columns"] == ["Vc", "Fz", "ap"]
    assert "rf_model" in reloaded
    assert "power_law_model" in reloaded

    reloaded_baseline = pd.read_csv(baseline_path)
    assert len(reloaded_baseline) == 2


def test_feature_importance_round_trip(tmp_path):
    path = tmp_path / "feature_importance.json"

    detailed = pd.DataFrame({"SVR": [0.6, 0.4], "GPR": [0.5, 0.5]}, index=["Vc", "Fz"])
    by_variable = pd.DataFrame({"SVR": [0.6, 0.4], "GPR": [0.5, 0.5]}, index=["Vc", "Fz"])

    save_feature_importance(detailed, by_variable, path=path)
    assert path.exists()

    reloaded = load_feature_importance(path=path)
    assert reloaded["detailed"]["Vc"]["SVR"] == 0.6
    assert reloaded["by_variable"]["Fz"]["GPR"] == 0.5


def test_load_feature_importance_from_real_training_run():
    # Assumes scripts/train_model.py was run with compute_importance=True
    # (the default), same prerequisite as the rest of this file.
    result = load_feature_importance()
    assert "detailed" in result
    assert "by_variable" in result
    assert set(result["by_variable"].keys()) == {"Vc", "Fz", "ap"}
