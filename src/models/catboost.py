"""CatBoost model wrapper - not implemented (see note below)."""


def train_catboost(X_train, y_train, **params):
    """Train a CatBoost model."""
    raise NotImplementedError(
        "catboost is not a dependency of this project - same reasoning as "
        "src/models/xgboost.py. There are also no categorical features in "
        "this dataset (Vc/Fz/ap are all continuous), which is CatBoost's "
        "main differentiator, so the case for adding it here specifically "
        "is weak. sklearn's GradientBoostingRegressor "
        "(src/models/models.py::build_gbm_pipeline) covers boosted trees "
        "without the extra dependency."
    )
