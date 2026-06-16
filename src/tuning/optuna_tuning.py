import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

from src.models import (
    build_svr_pipeline,
    build_gpr_pipeline
)

from src.config import (
    RANDOM_STATE,
    N_SPLITS,
    SVR_TRIALS,
    GPR_TRIALS,
    ENSEMBLE_TRIALS
)


def create_stratified_bins(y, n_bins=5):
    """
    Create stratified bins for regression.
    """

    bins = pd.qcut(
        y,
        q=n_bins,
        duplicates="drop"
    )

    return bins.codes


def tune_svr(X, y):

    bins = create_stratified_bins(y)

    def objective(trial):

        C_value = trial.suggest_float(
            "C",
            0.1,
            10,
            log=True
        )

        epsilon = trial.suggest_float(
            "epsilon",
            0.01,
            0.5,
            log=True
        )

        gamma = trial.suggest_float(
            "gamma",
            1e-3,
            1,
            log=True
        )

        poly_degree = trial.suggest_int(
            "poly_degree",
            2,
            4
        )

        model = build_svr_pipeline(
            C_value=C_value,
            epsilon=epsilon,
            gamma=gamma,
            poly_degree=poly_degree
        )

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        predictions = []
        targets = []

        for train_idx, test_idx in cv.split(X, bins):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            predictions.extend(preds.tolist())
            targets.extend(y_test.tolist())

        return mean_squared_error(targets, predictions)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        )
    )

    study.optimize(
        objective,
        n_trials=SVR_TRIALS,
        show_progress_bar=True
    )

    return study.best_params


def tune_gpr(X, y):

    bins = create_stratified_bins(y)

    def objective(trial):

        amplitude = trial.suggest_float(
            "amplitude",
            1e-2,
            5,
            log=True
        )

        length_scale = trial.suggest_float(
            "length_scale",
            0.05,
            3,
            log=True
        )

        alpha = trial.suggest_float(
            "alpha",
            1e-3,
            5,
            log=True
        )

        noise = trial.suggest_float(
            "noise",
            1e-6,
            1e-2,
            log=True
        )

        model = build_gpr_pipeline(
            amplitude=amplitude,
            length_scale=length_scale,
            alpha=alpha,
            noise=noise
        )

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        predictions = []
        targets = []

        for train_idx, test_idx in cv.split(X, bins):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            predictions.extend(preds.tolist())
            targets.extend(y_test.tolist())

        return mean_squared_error(targets, predictions)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        )
    )

    study.optimize(
        objective,
        n_trials=GPR_TRIALS,
        show_progress_bar=True
    )

    return study.best_params


def optimize_ensemble_weight(
    y_true,
    y_pred_svr,
    y_pred_gpr
):

    def objective(trial):

        alpha = trial.suggest_float(
            "alpha",
            0.0,
            1.0
        )

        y_ensemble = (
            alpha * y_pred_gpr
            + (1 - alpha) * y_pred_svr
        )

        return mean_squared_error(
            y_true,
            y_ensemble
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        )
    )

    study.optimize(
        objective,
        n_trials=ENSEMBLE_TRIALS,
        show_progress_bar=True
    )

    return study.best_params["alpha"]
