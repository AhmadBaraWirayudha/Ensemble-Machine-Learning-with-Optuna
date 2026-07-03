"""
Training entrypoint: load data -> engineer features -> tune SVR/GPR with
Optuna -> build the weighted + stacking ensembles from out-of-fold
predictions -> refit the base models on all available data -> persist
everything the API needs to serve predictions without retraining.

This is the piece the original prototype (Untitled-2.py) was missing: it
did every step up through the ensembles, but never saved a model, so
there was nothing for a service to load. See src/models/persistence.py.

Usable as a library (`from src.train import main`) or via the CLI wrapper
at scripts/train_model.py.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV

# GPR's internal kernel-hyperparameter optimizer routinely hits its search
# bounds on a dataset this small and warns about it on every fit - the
# original prototype silenced warnings globally for the same reason. Scope
# it to ConvergenceWarning specifically rather than blanket-silencing
# everything.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.config import RANDOM_STATE, N_SPLITS
from src.data_loader import load_dataset
from src.features import add_engineered_features, prepare_feature_matrix
from src.models import build_svr_pipeline, build_gpr_pipeline
from src.models.persistence import save_model_bundle
from src.optuna_tuning import (
    tune_svr,
    tune_gpr,
    optimize_ensemble_weight,
    create_stratified_bins,
)
from src.evaluation import calculate_metrics, generate_metrics_report
from src.visualization import (
    plot_target_distribution,
    plot_actual_vs_predicted,
    plot_model_comparison,
)
from src.utils.pokayoke import validate_input

# Trial-count / search-space presets. "quick" caps poly_degree at 3, which
# matters a lot: on this dataset, PolynomialFeatures expansion goes from 90
# features at degree=2 to 1819 at degree=4 (only 119 training rows), which
# measured about 20x slower per fit and is deep into overfitting territory
# for a sample this small. "full" restores the original 2-4 search space
# from the prototype/configs for anyone willing to let it run longer.
PRESETS = {
    "quick": dict(svr_trials=25, gpr_trials=25, ensemble_trials=40, max_poly_degree=3),
    "full": dict(svr_trials=60, gpr_trials=60, ensemble_trials=80, max_poly_degree=4),
}


def _log(verbose, msg):
    if verbose:
        print(msg, flush=True)


def main(
    preset="quick",
    svr_trials=None,
    gpr_trials=None,
    ensemble_trials=None,
    max_poly_degree=None,
    n_splits=None,
    random_state=None,
    use_pruning=True,
    save=True,
    make_plots=True,
    verbose=True,
    svr_params=None,
    gpr_params=None,
):
    """
    Run the full training pipeline. Explicit arguments override the chosen
    preset's defaults. Returns a dict summarizing the run (metrics,
    recommended model, hyperparameters, and - if save=True - the paths the
    bundle was written to) so this is usable both from the CLI script and
    from tests/notebooks.

    If svr_params / gpr_params are supplied, the corresponding Optuna
    search is skipped entirely and those hyperparameters are used as-is.
    Handy for redeploying a previously-found configuration without paying
    for a fresh search, or for resuming a search that was run externally
    (e.g. a checkpointed/resumable Optuna study).
    """

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose one of {list(PRESETS)}")

    cfg = dict(PRESETS[preset])
    if svr_trials is not None:
        cfg["svr_trials"] = svr_trials
    if gpr_trials is not None:
        cfg["gpr_trials"] = gpr_trials
    if ensemble_trials is not None:
        cfg["ensemble_trials"] = ensemble_trials
    if max_poly_degree is not None:
        cfg["max_poly_degree"] = max_poly_degree

    random_state = random_state if random_state is not None else RANDOM_STATE
    n_splits = n_splits if n_splits is not None else N_SPLITS

    t_start = time.time()

    _log(verbose, f"[1/8] Loading dataset (preset={preset}, config={cfg})...")
    df = load_dataset()
    validate_input(df)
    _log(verbose, f"      {len(df)} clean samples loaded.")

    _log(verbose, "[2/8] Engineering features...")
    df_eng = add_engineered_features(df)
    X, y, feature_columns = prepare_feature_matrix(df_eng)
    bins = create_stratified_bins(y)
    _log(verbose, f"      X shape={X.shape}, features={feature_columns}")

    if svr_params is None:
        _log(verbose, f"[3/8] Tuning SVR ({cfg['svr_trials']} trials, poly_degree<={cfg['max_poly_degree']})...")
        svr_params = tune_svr(
            X, y,
            n_trials=cfg["svr_trials"],
            max_poly_degree=cfg["max_poly_degree"],
            n_splits=n_splits,
            random_state=random_state,
            use_pruning=use_pruning,
            show_progress_bar=verbose,
        )
    else:
        _log(verbose, "[3/8] Using pre-computed svr_params, skipping search")
    _log(verbose, f"      best SVR params: {svr_params}")

    if gpr_params is None:
        _log(verbose, f"[4/8] Tuning GPR ({cfg['gpr_trials']} trials)...")
        gpr_params = tune_gpr(
            X, y,
            n_trials=cfg["gpr_trials"],
            n_splits=n_splits,
            random_state=random_state,
            use_pruning=use_pruning,
            show_progress_bar=verbose,
        )
    else:
        _log(verbose, "[4/8] Using pre-computed gpr_params, skipping search")
    _log(verbose, f"      best GPR params: {gpr_params}")

    _log(verbose, "[5/8] Collecting out-of-fold predictions for ensembling...")
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_true, y_pred_svr, y_pred_gpr = [], [], []

    for train_idx, test_idx in cv.split(X, bins):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_svr = build_svr_pipeline(
            C_value=svr_params["C"],
            epsilon=svr_params["epsilon"],
            gamma=svr_params["gamma"],
            poly_degree=svr_params["poly_degree"],
        )
        fold_gpr = build_gpr_pipeline(**gpr_params)

        fold_svr.fit(X_train, y_train)
        fold_gpr.fit(X_train, y_train)

        y_true.extend(y_test.tolist())
        y_pred_svr.extend(fold_svr.predict(X_test).tolist())
        y_pred_gpr.extend(fold_gpr.predict(X_test).tolist())

    y_true = np.array(y_true)
    y_pred_svr = np.array(y_pred_svr)
    y_pred_gpr = np.array(y_pred_gpr)

    _log(verbose, f"[6/8] Optimizing ensemble weight ({cfg['ensemble_trials']} trials) and fitting stacking meta-learner...")
    best_alpha = optimize_ensemble_weight(
        y_true, y_pred_svr, y_pred_gpr,
        n_trials=cfg["ensemble_trials"],
        random_state=random_state,
        show_progress_bar=verbose,
    )
    y_pred_weighted = best_alpha * y_pred_gpr + (1 - best_alpha) * y_pred_svr

    meta_X = np.column_stack([y_pred_svr, y_pred_gpr])
    meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    meta_model.fit(meta_X, y_true)
    y_pred_stack = meta_model.predict(meta_X)

    metrics = {
        "SVR": calculate_metrics(y_true, y_pred_svr),
        "GPR": calculate_metrics(y_true, y_pred_gpr),
        "Weighted_Ensemble": calculate_metrics(y_true, y_pred_weighted),
        "Stacking_Ensemble": calculate_metrics(y_true, y_pred_stack),
    }
    generate_metrics_report(metrics)

    recommended_model = (
        "Weighted_Ensemble"
        if metrics["Weighted_Ensemble"]["RMSE"] <= metrics["Stacking_Ensemble"]["RMSE"]
        else "Stacking_Ensemble"
    )
    _log(verbose, f"      alpha={best_alpha:.4f} | recommended: {recommended_model}")
    _log(verbose, f"      Weighted RMSE={metrics['Weighted_Ensemble']['RMSE']:.4f}  R2={metrics['Weighted_Ensemble']['R2']:.4f}")
    _log(verbose, f"      Stacking RMSE={metrics['Stacking_Ensemble']['RMSE']:.4f}  R2={metrics['Stacking_Ensemble']['R2']:.4f}")

    if make_plots:
        _log(verbose, "[7/8] Saving plots...")
        plot_target_distribution(df)
        plot_actual_vs_predicted(y_true, y_pred_svr, "SVR")
        plot_actual_vs_predicted(y_true, y_pred_gpr, "GPR")
        plot_actual_vs_predicted(y_true, y_pred_weighted, "Weighted_Ensemble")
        plot_actual_vs_predicted(y_true, y_pred_stack, "Stacking_Ensemble")
        plot_model_comparison(y_true, y_pred_svr, y_pred_gpr, y_pred_weighted)
    else:
        _log(verbose, "[7/8] Skipping plots (make_plots=False).")

    result = {
        "metrics": metrics,
        "recommended_model": recommended_model,
        "best_alpha": float(best_alpha),
        "svr_params": svr_params,
        "gpr_params": gpr_params,
        "feature_columns": feature_columns,
        "n_train_samples": len(df),
        "elapsed_seconds": time.time() - t_start,
    }

    if save:
        _log(verbose, "[8/8] Refitting final models on all data and saving bundle...")
        final_svr = build_svr_pipeline(
            C_value=svr_params["C"],
            epsilon=svr_params["epsilon"],
            gamma=svr_params["gamma"],
            poly_degree=svr_params["poly_degree"],
        )
        final_gpr = build_gpr_pipeline(**gpr_params)
        final_svr.fit(X, y)
        final_gpr.fit(X, y)

        bundle_path, metadata_path, baseline_path = save_model_bundle(
            svr_model=final_svr,
            gpr_model=final_gpr,
            meta_model=meta_model,
            best_alpha=best_alpha,
            feature_columns=feature_columns,
            svr_params=svr_params,
            gpr_params=gpr_params,
            metrics=metrics,
            recommended_model=recommended_model,
            training_df=df,
            n_train_samples=len(df),
            random_state=random_state,
        )
        result["bundle_path"] = str(bundle_path)
        result["metadata_path"] = str(metadata_path)
        result["baseline_path"] = str(baseline_path)
        _log(verbose, f"      saved: {bundle_path}")
        _log(verbose, f"      saved: {metadata_path}")
        _log(verbose, f"      saved: {baseline_path}")
    else:
        _log(verbose, "[8/8] Skipping save (save=False).")

    _log(verbose, f"\nTraining pipeline completed in {result['elapsed_seconds']:.1f}s")

    return result


if __name__ == "__main__":
    main()
