import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_importance import (
    compute_permutation_importance,
    aggregate_by_base_variable,
    compute_all_feature_importance,
)
from src.models.persistence import load_model_bundle
from src.data_loader import load_dataset
from src.features import add_engineered_features, prepare_feature_matrix


def test_compute_permutation_importance_ranks_the_informative_feature_first():
    # y depends only on the first column; a well-behaved importance measure
    # should rank it clearly above pure noise columns.
    rng = np.random.default_rng(0)
    n = 200
    x_informative = rng.normal(0, 1, n)
    noise_cols = rng.normal(0, 1, (n, 3))
    X = np.column_stack([x_informative, noise_cols])
    y = 5 * x_informative + rng.normal(0, 0.1, n)

    model = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())]).fit(X, y)

    importance = compute_permutation_importance(
        model, X, y, feature_names=["informative", "noise1", "noise2", "noise3"], n_repeats=15,
    )

    assert importance.index[0] == "informative"
    assert importance["informative"] > importance["noise1"]
    assert importance["informative"] > importance["noise2"]
    assert importance["informative"] > importance["noise3"]


def test_aggregate_by_base_variable_sums_matching_features():
    importance = pd.Series({"Vc": 0.1, "Vc2": 0.2, "Vc_Fz": 0.05, "Fz": 0.3, "ap": 0.4})

    result = aggregate_by_base_variable(importance, base_vars=("Vc", "Fz", "ap"))

    # Vc_Fz counts toward both Vc and Fz (see docstring - not split between them)
    assert result["Vc"] == pytest.approx(0.1 + 0.2 + 0.05)
    assert result["Fz"] == pytest.approx(0.3 + 0.05)
    assert result["ap"] == pytest.approx(0.4)


def test_compute_all_feature_importance_against_real_bundle():
    bundle = load_model_bundle()
    df = add_engineered_features(load_dataset())
    X, y, cols = prepare_feature_matrix(df)

    models = {"SVR": bundle["svr_model"], "GPR": bundle["gpr_model"]}
    detailed, by_variable = compute_all_feature_importance(models, X, y, cols, n_repeats=10)

    assert set(detailed.columns) == {"SVR", "GPR"}
    assert set(detailed.index) == set(cols)
    assert set(by_variable.index) == {"Vc", "Fz", "ap"}

    # normalized per model: nothing negative, and each column sums to ~1
    assert (detailed >= 0).all().all()
    for col in detailed.columns:
        assert abs(detailed[col].sum() - 1.0) < 1e-6
