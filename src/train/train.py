import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeCV

from src.config import (
    RANDOM_STATE,
    N_SPLITS
)

from src.data_loader import load_dataset

from src.features import (
    add_engineered_features,
    prepare_feature_matrix
)

from src.models import (
    build_svr_pipeline,
    build_gpr_pipeline
)

from src.optuna_tuning import (
    tune_svr,
    tune_gpr,
    optimize_ensemble_weight,
    create_stratified_bins
)

from src.evaluation import (
    calculate_metrics,
    generate_metrics_report
)

from src.visualization import (
    plot_target_distribution,
    plot_actual_vs_predicted,
    plot_model_comparison
)


def main():

    print("Loading dataset...")

    df = load_dataset()

    print("Generating engineered features...")

    df = add_engineered_features(df)

    X, y, feature_names = prepare_feature_matrix(df)

    bins = create_stratified_bins(y)

    # =====================================================
    # OPTUNA TUNING
    # =====================================================

    print("Tuning SVR...")

    svr_params = tune_svr(X, y)

    print("Best SVR Parameters")
    print(svr_params)

    print("Tuning GPR...")

    gpr_params = tune_gpr(X, y)

    print("Best GPR Parameters")
    print(gpr_params)

    # =====================================================
    # BUILD MODELS
    # =====================================================

    svr_model = build_svr_pipeline(
        C_value=svr_params["C"],
        epsilon=svr_params["epsilon"],
        gamma=svr_params["gamma"],
        poly_degree=svr_params["poly_degree"]
    )

    gpr_model = build_gpr_pipeline(
        amplitude=gpr_params["amplitude"],
        length_scale=gpr_params["length_scale"],
        alpha=gpr_params["alpha"],
        noise=gpr_params["noise"]
    )

    # =====================================================
    # CROSS VALIDATION
    # =====================================================

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    y_true = []
    y_pred_svr = []
    y_pred_gpr = []

    for train_idx, test_idx in cv.split(X, bins):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        svr_model.fit(X_train, y_train)
        gpr_model.fit(X_train, y_train)

        pred_svr = svr_model.predict(X_test)
        pred_gpr = gpr_model.predict(X_test)

        y_true.extend(y_test.tolist())

        y_pred_svr.extend(pred_svr.tolist())
        y_pred_gpr.extend(pred_gpr.tolist())

    y_true = np.array(y_true)

    y_pred_svr = np.array(y_pred_svr)
    y_pred_gpr = np.array(y_pred_gpr)

    # =====================================================
    # ENSEMBLE
    # =====================================================

    print("Optimizing ensemble weight...")

    best_alpha = optimize_ensemble_weight(
        y_true,
        y_pred_svr,
        y_pred_gpr
    )

    y_pred_ensemble = (
        best_alpha * y_pred_gpr
        + (1 - best_alpha) * y_pred_svr
    )

    # =====================================================
    # STACKING
    # =====================================================

    meta_X = np.column_stack([
        y_pred_svr,
        y_pred_gpr
    ])

    meta_model = RidgeCV(
        alphas=[0.1, 1.0, 10.0]
    )

    meta_model.fit(meta_X, y_true)

    y_pred_stack = meta_model.predict(meta_X)

    # =====================================================
    # EVALUATION
    # =====================================================

    results = {
        "SVR": calculate_metrics(
            y_true,
            y_pred_svr
        ),

        "GPR": calculate_metrics(
            y_true,
            y_pred_gpr
        ),

        "Weighted_Ensemble": calculate_metrics(
            y_true,
            y_pred_ensemble
        ),

        "Stacking_Ensemble": calculate_metrics(
            y_true,
            y_pred_stack
        )
    }

    generate_metrics_report(results)

    # =====================================================
    # VISUALIZATION
    # =====================================================

    plot_target_distribution(df)

    plot_actual_vs_predicted(
        y_true,
        y_pred_svr,
        "SVR"
    )

    plot_actual_vs_predicted(
        y_true,
        y_pred_gpr,
        "GPR"
    )

    plot_actual_vs_predicted(
        y_true,
        y_pred_ensemble,
        "ENSEMBLE"
    )

    plot_model_comparison(
        y_true,
        y_pred_svr,
        y_pred_gpr,
        y_pred_ensemble
    )

    print("\nTraining Pipeline Completed")


if __name__ == "__main__":
    main()
