import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from src.models import (
    build_svr_pipeline,
    build_gpr_pipeline,
    build_gbm_pipeline,
    build_rf_pipeline,
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


def create_replicate_groups(df):
    """
    Group ID per row identifying which exact (Vc, Fz, ap) design point it
    belongs to. This dataset is a full 5x5x4 factorial DOE (100 unique
    combinations) where 19 of those 100 points were measured twice - so 19
    groups have 2 rows and 81 have 1. Plain StratifiedKFold doesn't know
    about this and can (does, in ~17 of 19 cases at the default seed) split
    a replicate pair across train/test. Use with StratifiedGroupKFold
    instead of StratifiedKFold to guarantee replicates never do that,
    keeping every fold's test set genuinely unseen.
    """

    return df.groupby(["Vc", "Fz", "ap"]).ngroup().values


def tune_svr(
    X,
    y,
    n_trials=None,
    max_poly_degree=4,
    n_splits=None,
    random_state=None,
    use_pruning=True,
    show_progress_bar=True,
    groups=None,
):
    """
    Tune SVR hyperparameters with Optuna.

    poly_degree is searched in [2, max_poly_degree]. Be aware of the cost:
    on this dataset's 12 base features, PolynomialFeatures expansion goes
    from 90 output features at degree=2 to 1819 at degree=4, so a degree=4
    trial's 5-fold CV costs roughly 20x a degree=2 trial (measured: ~1.2s
    vs ~25s per fit). With only 119 training rows, degree=4 is also deep
    into "more features than samples" overfitting territory. Pruning
    (median pruner, on by default) lets clearly-uncompetitive trials -
    which is often the expensive high-degree ones - get abandoned after
    the first fold or two instead of always paying for all 5.

    Pass `groups` (see create_replicate_groups) to use StratifiedGroupKFold
    instead of plain StratifiedKFold - this dataset has 19 exact-duplicate
    (Vc,Fz,ap) replicate measurements, which plain StratifiedKFold can (and
    at this module's default seed, mostly does) split across train/test.
    Defaults to None for backward compatibility with any existing callers.
    """

    n_trials = n_trials if n_trials is not None else SVR_TRIALS
    n_splits = n_splits if n_splits is not None else N_SPLITS
    random_state = random_state if random_state is not None else RANDOM_STATE

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
            max_poly_degree
        )

        model = build_svr_pipeline(
            C_value=C_value,
            epsilon=epsilon,
            gamma=gamma,
            poly_degree=poly_degree
        )

        if groups is not None:
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
            split_iter = cv.split(X, bins, groups)
        else:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
            split_iter = cv.split(X, bins)

        predictions = []
        targets = []

        for fold_idx, (train_idx, test_idx) in enumerate(split_iter):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            predictions.extend(preds.tolist())
            targets.extend(y_test.tolist())

            if use_pruning:
                running_mse = mean_squared_error(targets, predictions)
                trial.report(running_mse, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return mean_squared_error(targets, predictions)

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        if use_pruning
        else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=random_state
        ),
        pruner=pruner,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar
    )

    return study.best_params


def tune_gpr(
    X,
    y,
    n_trials=None,
    n_splits=None,
    random_state=None,
    use_pruning=True,
    show_progress_bar=True,
    groups=None,
):
    """Tune GPR hyperparameters with Optuna. See tune_svr's docstring for
    the `groups` parameter - same StratifiedGroupKFold-vs-StratifiedKFold
    behavior applies here."""

    n_trials = n_trials if n_trials is not None else GPR_TRIALS
    n_splits = n_splits if n_splits is not None else N_SPLITS
    random_state = random_state if random_state is not None else RANDOM_STATE

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

        if groups is not None:
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
            split_iter = cv.split(X, bins, groups)
        else:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
            split_iter = cv.split(X, bins)

        predictions = []
        targets = []

        for fold_idx, (train_idx, test_idx) in enumerate(split_iter):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            predictions.extend(preds.tolist())
            targets.extend(y_test.tolist())

            if use_pruning:
                running_mse = mean_squared_error(targets, predictions)
                trial.report(running_mse, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return mean_squared_error(targets, predictions)

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        if use_pruning
        else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=random_state
        ),
        pruner=pruner,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar
    )

    return study.best_params


def tune_rf(X, y, groups, n_trials=25, n_splits=None, random_state=None, use_pruning=True, show_progress_bar=True):
    """
    Tune RandomForest hyperparameters with Optuna, using StratifiedGroupKFold
    (see create_replicate_groups) rather than plain StratifiedKFold - since
    this is new code, it starts off using the more rigorous CV strategy
    directly rather than inheriting tune_svr/tune_gpr's original choice.
    """

    n_splits = n_splits if n_splits is not None else N_SPLITS
    random_state = random_state if random_state is not None else RANDOM_STATE
    bins = create_stratified_bins(y)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
        max_depth = trial.suggest_int("max_depth", 2, 12)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

        model = build_rf_pipeline(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            random_state=random_state,
        )
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        predictions, targets = [], []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, bins, groups)):
            model.fit(X[train_idx], y[train_idx])
            predictions.extend(model.predict(X[test_idx]).tolist())
            targets.extend(y[test_idx].tolist())
            if use_pruning:
                trial.report(mean_squared_error(targets, predictions), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return mean_squared_error(targets, predictions)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1) if use_pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state), pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study.best_params


def tune_gbm(X, y, groups, n_trials=25, n_splits=None, random_state=None, use_pruning=True, show_progress_bar=True):
    """Tune Gradient Boosting hyperparameters with Optuna. See tune_rf docstring."""

    n_splits = n_splits if n_splits is not None else N_SPLITS
    random_state = random_state if random_state is not None else RANDOM_STATE
    bins = create_stratified_bins(y)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 400, step=25)
        max_depth = trial.suggest_int("max_depth", 1, 5)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)

        model = build_gbm_pipeline(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, min_samples_leaf=min_samples_leaf, random_state=random_state,
        )
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        predictions, targets = [], []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, bins, groups)):
            model.fit(X[train_idx], y[train_idx])
            predictions.extend(model.predict(X[test_idx]).tolist())
            targets.extend(y[test_idx].tolist())
            if use_pruning:
                trial.report(mean_squared_error(targets, predictions), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return mean_squared_error(targets, predictions)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1) if use_pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state), pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study.best_params


def optimize_ensemble_weight(
    y_true,
    y_pred_svr,
    y_pred_gpr,
    n_trials=None,
    random_state=None,
    show_progress_bar=True,
):
    """
    Optimize the weighted-ensemble blend alpha * GPR + (1 - alpha) * SVR.
    Operates purely on already-computed out-of-fold predictions, so unlike
    tune_svr/tune_gpr this never fits a model - even 80+ trials finish in
    well under a second.
    """

    n_trials = n_trials if n_trials is not None else ENSEMBLE_TRIALS
    random_state = random_state if random_state is not None else RANDOM_STATE

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
            seed=random_state
        )
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar
    )

    return study.best_params["alpha"]
