"""
Training entrypoint: load data -> engineer features -> tune SVR/GPR/
RandomForest/GradientBoosting with Optuna (+ fit the parameter-free
power-law baseline) -> build the weighted + stacking ensembles from
out-of-fold predictions -> refit the base models on all available data ->
persist everything the API needs to serve predictions without retraining.

Two corrections layered on top of the original design here, both found by
actually interrogating the dataset rather than assuming the original
approach was well-calibrated (see UPGRADE_NOTES.md for the full story):

1. CV uses StratifiedGroupKFold, grouping by exact (Vc,Fz,ap) identity.
   19 of this dataset's 100 unique design points were measured twice: a
   plain StratifiedKFold split doesn't know that and, at this module's
   default seed, splits 17 of those 19 replicate pairs across train/test.
2. RandomForest and GradientBoosting are now tuned and included as base
   learners alongside the original SVR/GPR. On a fair side-by-side
   comparison (identical CV folds), they substantially outperform SVR/GPR
   on this dataset (~0.60-0.63 R2 vs ~0.39-0.41) - the heavy manual
   feature engineering plus polynomial expansion that SVR/GPR rely on
   turns out to suit this narrow-DOE-grid, replicate-heavy dataset less
   well than trees, which split naturally on the ~4-5 discrete levels each
   raw parameter takes.

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
from sklearn.model_selection import StratifiedGroupKFold

# GPR's internal kernel-hyperparameter optimizer routinely hits its search
# bounds on a dataset this small and warns about it on every fit - the
# original prototype silenced warnings globally for the same reason. Scope
# it to ConvergenceWarning specifically rather than blanket-silencing
# everything.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.config import RANDOM_STATE, N_SPLITS, FEATURE_DIR
from src.data_loader import load_dataset
from src.features import add_engineered_features, prepare_feature_matrix
from src.models import (
    build_svr_pipeline,
    build_gpr_pipeline,
    build_rf_pipeline,
    build_gbm_pipeline,
    PowerLawRegressor,
)
from src.models.persistence import save_model_bundle, save_feature_importance
from src.optuna_tuning import (
    tune_svr,
    tune_gpr,
    tune_rf,
    tune_gbm,
    optimize_ensemble_weight,
    create_stratified_bins,
    create_replicate_groups,
)
from src.evaluation import calculate_metrics, generate_metrics_report
from src.visualization import (
    plot_target_distribution,
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_feature_importance,
)
from src.feature_importance import compute_all_feature_importance
from src.utils.pokayoke import validate_input

# Trial-count / search-space presets. "quick" caps SVR's poly_degree at 3,
# which matters a lot: on this dataset, PolynomialFeatures expansion goes
# from 90 features at degree=2 to 1819 at degree=4 (only 119 training
# rows), which measured about 20x slower per fit and is deep into
# overfitting territory for a sample this small. "full" restores the
# original 2-4 search space from the prototype/configs for anyone willing
# to let it run longer. RF/GBM trial counts are the same in both presets -
# tree ensembles are cheap here (seconds, not tens of seconds, per trial).
PRESETS = {
    "quick": dict(svr_trials=25, gpr_trials=25, rf_trials=25, gbm_trials=25, ensemble_trials=40, max_poly_degree=3),
    "full": dict(svr_trials=60, gpr_trials=60, rf_trials=40, gbm_trials=40, ensemble_trials=80, max_poly_degree=4),
}

# Models combined into the stacking meta-learner. PowerLawRegressor is
# deliberately excluded here (but still trained, evaluated, and reported
# standalone) - side-by-side testing found it contributes only a small,
# hard-to-interpret *negative* weight when included (RidgeCV effectively
# partially cancels it against another model rather than adding signal),
# with no improvement to stacked RMSE. It's kept as an interpretable
# baseline in its own right (see PowerLawRegressor.formula_string()), not
# because it strengthens the ensemble numerically.
STACKED_MODEL_KEYS = ["svr_model", "gpr_model", "rf_model", "gbm_model"]


def _log(verbose, msg):
    if verbose:
        print(msg, flush=True)


def main(
    preset="quick",
    svr_trials=None,
    gpr_trials=None,
    rf_trials=None,
    gbm_trials=None,
    ensemble_trials=None,
    max_poly_degree=None,
    n_splits=None,
    random_state=None,
    use_pruning=True,
    save=True,
    make_plots=True,
    compute_importance=True,
    verbose=True,
    svr_params=None,
    gpr_params=None,
    rf_params=None,
    gbm_params=None,
):
    """
    Run the full training pipeline. Explicit arguments override the chosen
    preset's defaults. Returns a dict summarizing the run (metrics,
    recommended model, hyperparameters, and - if save=True - the paths the
    bundle was written to) so this is usable both from the CLI script and
    from tests/notebooks.

    If svr_params / gpr_params / rf_params / gbm_params are supplied, the
    corresponding Optuna search is skipped entirely and those
    hyperparameters are used as-is. Handy for redeploying a previously-
    found configuration without paying for a fresh search, or for
    resuming a search that was run externally (e.g. a checkpointed/
    resumable Optuna study, which is how this project's own shipped model
    was tuned inside a sandboxed environment with tight per-command time
    limits - see UPGRADE_NOTES.md).
    """

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose one of {list(PRESETS)}")

    cfg = dict(PRESETS[preset])
    for key, val in [("svr_trials", svr_trials), ("gpr_trials", gpr_trials),
                      ("rf_trials", rf_trials), ("gbm_trials", gbm_trials),
                      ("ensemble_trials", ensemble_trials), ("max_poly_degree", max_poly_degree)]:
        if val is not None:
            cfg[key] = val

    random_state = random_state if random_state is not None else RANDOM_STATE
    n_splits = n_splits if n_splits is not None else N_SPLITS

    t_start = time.time()

    _log(verbose, f"[1/9] Loading dataset (preset={preset}, config={cfg})...")
    df = load_dataset()
    validate_input(df)
    _log(verbose, f"      {len(df)} clean samples loaded.")

    _log(verbose, "[2/9] Engineering features...")
    df_eng = add_engineered_features(df)
    X, y, feature_columns = prepare_feature_matrix(df_eng)
    bins = create_stratified_bins(y)
    groups = create_replicate_groups(df)
    _log(verbose, f"      X shape={X.shape}, features={feature_columns}")
    _log(verbose, f"      {len(np.unique(groups))} unique (Vc,Fz,ap) design points "
                  f"({len(y) - len(np.unique(groups))} are replicate measurements)")

    if svr_params is None:
        _log(verbose, f"[3/9] Tuning SVR ({cfg['svr_trials']} trials, poly_degree<={cfg['max_poly_degree']})...")
        svr_params = tune_svr(
            X, y, n_trials=cfg["svr_trials"], max_poly_degree=cfg["max_poly_degree"],
            n_splits=n_splits, random_state=random_state, use_pruning=use_pruning,
            show_progress_bar=verbose, groups=groups,
        )
    else:
        _log(verbose, "[3/9] Using pre-computed svr_params, skipping search")
    _log(verbose, f"      best SVR params: {svr_params}")

    if gpr_params is None:
        _log(verbose, f"[4/9] Tuning GPR ({cfg['gpr_trials']} trials)...")
        gpr_params = tune_gpr(
            X, y, n_trials=cfg["gpr_trials"], n_splits=n_splits, random_state=random_state,
            use_pruning=use_pruning, show_progress_bar=verbose, groups=groups,
        )
    else:
        _log(verbose, "[4/9] Using pre-computed gpr_params, skipping search")
    _log(verbose, f"      best GPR params: {gpr_params}")

    if rf_params is None:
        _log(verbose, f"[4b/9] Tuning RandomForest ({cfg['rf_trials']} trials)...")
        rf_params = tune_rf(
            X, y, groups, n_trials=cfg["rf_trials"], n_splits=n_splits, random_state=random_state,
            use_pruning=use_pruning, show_progress_bar=verbose,
        )
    else:
        _log(verbose, "[4b/9] Using pre-computed rf_params, skipping search")
    _log(verbose, f"      best RF params: {rf_params}")

    if gbm_params is None:
        _log(verbose, f"[4c/9] Tuning GradientBoosting ({cfg['gbm_trials']} trials)...")
        gbm_params = tune_gbm(
            X, y, groups, n_trials=cfg["gbm_trials"], n_splits=n_splits, random_state=random_state,
            use_pruning=use_pruning, show_progress_bar=verbose,
        )
    else:
        _log(verbose, "[4c/9] Using pre-computed gbm_params, skipping search")
    _log(verbose, f"      best GBM params: {gbm_params}")

    _log(verbose, "[5/9] Collecting out-of-fold predictions (StratifiedGroupKFold - replicates never split across train/test)...")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_true = []
    oof = {"svr_model": [], "gpr_model": [], "rf_model": [], "gbm_model": [], "power_law_model": []}

    for train_idx, test_idx in cv.split(X, bins, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_models = {
            "svr_model": build_svr_pipeline(C_value=svr_params["C"], epsilon=svr_params["epsilon"], gamma=svr_params["gamma"], poly_degree=svr_params["poly_degree"]),
            "gpr_model": build_gpr_pipeline(**gpr_params),
            "rf_model": build_rf_pipeline(**rf_params),
            "gbm_model": build_gbm_pipeline(**gbm_params),
            "power_law_model": PowerLawRegressor(),
        }

        y_true.extend(y_test.tolist())
        for key, model in fold_models.items():
            model.fit(X_train, y_train)
            oof[key].extend(model.predict(X_test).tolist())

    y_true = np.array(y_true)
    for key in oof:
        oof[key] = np.array(oof[key])

    _log(verbose, f"[6/9] Optimizing weighted ensemble (SVR+GPR, {cfg['ensemble_trials']} trials) and fitting stacking meta-learner (SVR+GPR+RF+GBM)...")
    best_alpha = optimize_ensemble_weight(
        y_true, oof["svr_model"], oof["gpr_model"],
        n_trials=cfg["ensemble_trials"], random_state=random_state, show_progress_bar=verbose,
    )
    y_pred_weighted = best_alpha * oof["gpr_model"] + (1 - best_alpha) * oof["svr_model"]

    stack_X = np.column_stack([oof[key] for key in STACKED_MODEL_KEYS])
    meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta_model.fit(stack_X, y_true)
    y_pred_stack = meta_model.predict(stack_X)

    metrics = {
        "SVR": calculate_metrics(y_true, oof["svr_model"]),
        "GPR": calculate_metrics(y_true, oof["gpr_model"]),
        "RandomForest": calculate_metrics(y_true, oof["rf_model"]),
        "GradientBoosting": calculate_metrics(y_true, oof["gbm_model"]),
        "PowerLaw": calculate_metrics(y_true, oof["power_law_model"]),
        "Weighted_Ensemble": calculate_metrics(y_true, y_pred_weighted),
        "Stacking_Ensemble": calculate_metrics(y_true, y_pred_stack),
    }
    generate_metrics_report(metrics)

    recommended_model = min(metrics, key=lambda name: metrics[name]["RMSE"])
    _log(verbose, f"      alpha={best_alpha:.4f}")
    _log(verbose, f"      meta-learner weights: {dict(zip(STACKED_MODEL_KEYS, np.round(meta_model.coef_, 3)))}")
    for name, m in metrics.items():
        marker = " <-- recommended" if name == recommended_model else ""
        _log(verbose, f"      {name:18s} RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}{marker}")

    if make_plots:
        _log(verbose, "[7/9] Saving plots...")
        plot_target_distribution(df)
        plot_actual_vs_predicted(y_true, oof["svr_model"], "SVR")
        plot_actual_vs_predicted(y_true, oof["gpr_model"], "GPR")
        plot_actual_vs_predicted(y_true, oof["rf_model"], "RandomForest")
        plot_actual_vs_predicted(y_true, oof["gbm_model"], "GradientBoosting")
        plot_actual_vs_predicted(y_true, oof["power_law_model"], "PowerLaw")
        plot_actual_vs_predicted(y_true, y_pred_weighted, "Weighted_Ensemble")
        plot_actual_vs_predicted(y_true, y_pred_stack, "Stacking_Ensemble")
        plot_model_comparison(y_true, oof["svr_model"], oof["gpr_model"], y_pred_weighted)
    else:
        _log(verbose, "[7/9] Skipping plots (make_plots=False).")

    result = {
        "metrics": metrics,
        "recommended_model": recommended_model,
        "best_alpha": float(best_alpha),
        "svr_params": svr_params,
        "gpr_params": gpr_params,
        "rf_params": rf_params,
        "gbm_params": gbm_params,
        "feature_columns": feature_columns,
        "n_train_samples": len(df),
        "n_unique_design_points": int(len(np.unique(groups))),
        "elapsed_seconds": time.time() - t_start,
    }

    if save:
        _log(verbose, "[8/9] Refitting final models on all data...")
        final_models = {
            "svr_model": build_svr_pipeline(C_value=svr_params["C"], epsilon=svr_params["epsilon"], gamma=svr_params["gamma"], poly_degree=svr_params["poly_degree"]),
            "gpr_model": build_gpr_pipeline(**gpr_params),
            "rf_model": build_rf_pipeline(**rf_params),
            "gbm_model": build_gbm_pipeline(**gbm_params),
            "power_law_model": PowerLawRegressor(),
        }
        for model in final_models.values():
            model.fit(X, y)

        _log(verbose, f"      {final_models['power_law_model'].formula_string()}")

        bundle_path, metadata_path, baseline_path = save_model_bundle(
            svr_model=final_models["svr_model"],
            gpr_model=final_models["gpr_model"],
            rf_model=final_models["rf_model"],
            gbm_model=final_models["gbm_model"],
            power_law_model=final_models["power_law_model"],
            meta_model=meta_model,
            best_alpha=best_alpha,
            feature_columns=feature_columns,
            svr_params=svr_params,
            gpr_params=gpr_params,
            rf_params=rf_params,
            gbm_params=gbm_params,
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

        if compute_importance:
            _log(verbose, "[9/9] Computing permutation importance...")
            importance_models = {
                "SVR": final_models["svr_model"],
                "GPR": final_models["gpr_model"],
                "RandomForest": final_models["rf_model"],
                "GradientBoosting": final_models["gbm_model"],
            }
            detailed, by_variable = compute_all_feature_importance(
                importance_models, X, y, feature_columns, random_state=random_state,
            )

            FEATURE_DIR.mkdir(parents=True, exist_ok=True)
            detailed.to_csv(FEATURE_DIR / "feature_importance_detailed.csv")
            by_variable.to_csv(FEATURE_DIR / "feature_importance_by_variable.csv")
            importance_path = save_feature_importance(detailed, by_variable)
            result["feature_importance_path"] = str(importance_path)

            if make_plots:
                plot_feature_importance(
                    by_variable,
                    title="Feature Importance by Machining Parameter",
                    filename="feature_importance_by_variable.png",
                )

            top_by_model = {col: by_variable[col].idxmax() for col in by_variable.columns}
            _log(verbose, f"      most important parameter per model: {top_by_model}")
            _log(verbose, f"      saved: {importance_path}")
        else:
            _log(verbose, "[9/9] Skipping feature importance (compute_importance=False).")
    else:
        _log(verbose, "[8/9] Skipping refit + save (save=False).")
        _log(verbose, "[9/9] Skipping feature importance (requires save=True).")

    _log(verbose, f"\nTraining pipeline completed in {result['elapsed_seconds']:.1f}s")

    return result


if __name__ == "__main__":
    main()
