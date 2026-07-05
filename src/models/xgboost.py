"""XGBoost model wrapper - not implemented (see note below)."""


def train_xgboost(X_train, y_train, **params):
    """Train an XGBoost model."""
    raise NotImplementedError(
        "xgboost is not a dependency of this project. On a dataset this "
        "small (119 rows), sklearn's GradientBoostingRegressor (see "
        "src/models/models.py::build_gbm_pipeline) already covers the "
        "boosted-trees approach without adding a new dependency. If you "
        "want XGBoost specifically (e.g. for GPU support or a larger "
        "future dataset), add `xgboost` to requirements.txt and implement "
        "this the same way build_gbm_pipeline is implemented."
    )
