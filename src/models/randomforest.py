"""Random forest model wrapper."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def build_rf_pipeline(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features="sqrt", random_state=151101):
    """
    Random forest on the 12 engineered features directly - no
    PolynomialFeatures step. Trees split on raw feature values and capture
    nonlinearity/interactions natively, so stacking an explicit polynomial
    expansion in front of them (as the SVR pipeline does) would only add
    thousands of collinear, uninformative columns without giving the model
    any new capability - unlike SVR/GPR, which need the expansion to
    represent nonlinear relationships at all.
    """

    return Pipeline([
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )),
    ])


def train_random_forest(X_train, y_train, **params):
    """Train a random forest model with the given hyperparameters."""
    model = build_rf_pipeline(**params)
    model.fit(X_train, y_train)
    return model
