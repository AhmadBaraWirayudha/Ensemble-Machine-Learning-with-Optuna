from src.models import (
    build_svr_pipeline,
    build_gpr_pipeline,
    build_rf_pipeline,
    build_gbm_pipeline,
    PowerLawRegressor,
)


def test_svr_pipeline():

    model = build_svr_pipeline()

    assert model is not None


def test_gpr_pipeline():

    model = build_gpr_pipeline()

    assert model is not None


def test_rf_pipeline():
    model = build_rf_pipeline()
    assert model is not None


def test_gbm_pipeline():
    model = build_gbm_pipeline()
    assert model is not None


def test_power_law_regressor_fits_and_predicts():
    import numpy as np

    X = np.array([[10.0, 0.1, 1.0], [12.5, 0.1, 1.0], [7.5, 0.05, 0.75], [17.5, 0.15, 1.5]])
    y = np.array([0.7, 0.8, 0.5, 1.2])

    model = PowerLawRegressor().fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(y)
    assert (preds > 0).all()  # Ra is physically non-negative
    assert set(model.exponents_.keys()) == {"Vc", "Fz", "ap"}
    assert "Ra = " in model.formula_string()


def test_power_law_regressor_rejects_non_positive_input():
    import numpy as np
    import pytest

    X = np.array([[10.0, 0.1, 1.0], [-1.0, 0.1, 1.0]])
    y = np.array([0.7, 0.8])

    with pytest.raises(ValueError):
        PowerLawRegressor().fit(X, y)
