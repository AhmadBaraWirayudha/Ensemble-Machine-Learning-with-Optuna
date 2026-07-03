
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna
import queue
import logging
import time
warnings.filterwarnings("ignore")

from sklearn.gaussian_process.kernels import RBF, Product, Sum, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import RidgeCV

import tkinter as tk
from tkinter import ttk, filedialog
from threading import Thread
from PIL import Image, ImageTk
import glob

from openpyxl import Workbook
from openpyxl.chart import ScatterChart, Reference, Series

# LOGGER FOR GUI
gui_log_queue = queue.Queue()

def gui_log(msg):
    print(msg)
    gui_log_queue.put(msg + "\n")

# CONFIG
DATA_PATH = Path(r"D:\SKRIPSI REBORN\data data dan training\experimental\Sheet2.csv")
OUT_DIR = Path("DATA_Hasil_Bara")
FIG_DIR = OUT_DIR / "Plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 151101
POLY_DEGREE = 2


# Optuna trials
SVR_TRIALS = 60
GPR_TRIALS = 60
ENS_TRIALS = 80


# MODEL BUILDERS
def svr_rbf(C=1.5, epsilon=0.2, gamma='scale', poly_degree=2):
    return Pipeline([
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("scale", RobustScaler()),
            ("power", PowerTransformer(method='yeo-johnson', standardize=True, copy=True)),
            ("svr", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))
        ])


def make_svr_pipeline(C, eps, gamma, poly_degree):
    return TransformedTargetRegressor(
            regressor=Pipeline([
                ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
                ("scale", RobustScaler()),
                ("power", PowerTransformer(method='yeo-johnson', standardize=True, copy=True)),
                ("svr", SVR(kernel="rbf", C=C, epsilon=eps, gamma=gamma))
            ]),
            transformer=PowerTransformer(standardize=True)
        )

def gpr_rbf(amplitude=1.0, length_scale=1.0, alpha=1.0, noise=1e-4):
    kernel = (
        C(amplitude)
        * (RBF(length_scale=length_scale) * RationalQuadratic(alpha=alpha))
        + WhiteKernel(noise_level=noise)
    )

    return Pipeline([
            ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),  # keep fixed or tune
            ("scale", RobustScaler()),
            ("power", PowerTransformer(method='yeo-johnson')),
            ("gpr", GaussianProcessRegressor(kernel=kernel, normalize_y=True))
        ])


def wrap(pipe):
    # useful for SVR target transform
    return TransformedTargetRegressor(regressor=pipe, transformer=PowerTransformer())

def main_pipeline():
    gui_log("Starting ML Pipeline")
    

   
    # Load Data
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=lambda x: x.strip())
    # ensure we select correct columns
    if "Ax" in df.columns and "ap" not in df.columns:
        df = df.rename(columns={"Ax": "ap"})
    df = df[["Vc", "Fz", "ap", "Ra"]].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
    X = df[["Vc", "Fz", "ap"]].values
    y = df["Ra"].values
    gui_log(f"Loaded {len(df)} samples with {X.shape[1]} features")

    # Feature Engineering
    def add_features(df):
        df = df.copy()
        # Squared features
        df["Vc2"] = df["Vc"]**2
        df["Fz2"] = df["Fz"]**2
        df["ap2"] = df["ap"]**2
        # Interactions
        df["Vc_Fz"] = df["Vc"] * df["Fz"]
        df["Vc_ap"] = df["Vc"] * df["ap"]
        df["Fz_ap"] = df["Fz"] * df["ap"]
        # Ratios (purely mathematical)
        df["Vc/Fz"] = df["Vc"] / df["Fz"]
        df["Fz/ap"] = df["Fz"] / df["ap"]
        df["Vc/ap"] = df["Vc"] / df["ap"]
        #log transformation
        df["log_Vc"] = np.log(df["Vc"])
        df["log_Fz"] = np.log(df["Fz"])
        df["log_ap"] = np.log(df["ap"])
        return df
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df = add_features(df)
    
    X = df[["Vc","Fz","ap",
            "Vc2","Fz2","ap2",
            "Vc_Fz","Vc_ap","Fz_ap",
            "Vc/Fz","Fz/ap","Vc/ap"]].values
    y = df["Ra"].values

    # Helpers
    def metrics_report(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mbe = np.mean(y_pred - y_true)
        r2 = r2_score(y_true, y_pred)
        # Safe MAPE
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
        # Relative error scale
        mean_y = np.mean(np.abs(y_true))
        rmse_pct = rmse / mean_y * 100
        mae_pct = mae / mean_y * 100
        mbe_pct = mbe / mean_y * 100
        return {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "RMSE_%": float(rmse_pct),
            "MAE": float(mae),
            "MAE_%": float(mae_pct),
            "MBE": float(mbe),
            "MBE_%": float(mbe_pct),
            "MAPE": float(mape),
            "R2": float(r2)
        }
    def savefig(fig, name):
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = FIG_DIR / name
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        gui_log(f"[Saved] {path}")


    # Optuna Tuning: SVR
    gui_log("Tuning SVR (RBF)")
    def svr_objective(trial):
        C = trial.suggest_float("C", 0.1, 10, log=True)
        eps = trial.suggest_float("epsilon", 0.01, 0.5, log=True)
        gamma = trial.suggest_float("gamma", 1e-3, 1, log=True)
        poly_degree = trial.suggest_int("poly_degree", 2, 4)
        model = make_svr_pipeline(C, eps, gamma, poly_degree)
        # stratified bins
        bins = pd.qcut(y, q=5, duplicates="drop")
        bins = bins.codes
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        preds, truths = [], []
        for tr, te in cv.split(X, bins):
            model.fit(X[tr], y[tr])
            preds.extend(model.predict(X[te]).tolist())
            truths.extend(y[te].tolist())
        return mean_squared_error(truths, preds)
    svr_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    svr_study.optimize(svr_objective, n_trials=SVR_TRIALS, show_progress_bar=False)
    svr_params = svr_study.best_params
    gui_log(f"best SVR parameter: {svr_params}")


    # Optuna Tuning: GPR
    gui_log("Tuning GPR (RBF)")
    def gpr_objective(trial):

        amp = trial.suggest_float("amplitude", 1e-2, 5, log=True)
        lscale = trial.suggest_float("length_scale", 0.05, 3, log=True)
        alpha = trial.suggest_float("alpha", 1e-3, 5, log=True)
        noise = trial.suggest_float("noise", 1e-6, 1e-2, log=True)
        model = gpr_rbf(amplitude=amp, length_scale=lscale, alpha=alpha, noise=noise)
        bins = pd.qcut(y, q=5, duplicates="drop")
        bins = bins.codes
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        preds, truths = [], []
        for tr, te in cv.split(X, bins):
            model.fit(X[tr], y[tr])
            preds.extend(model.predict(X[te]).tolist())
            truths.extend(y[te].tolist())
        return mean_squared_error(truths, preds)
    gpr_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    gpr_study.optimize(gpr_objective, n_trials=GPR_TRIALS, show_progress_bar=False)
    gpr_params = gpr_study.best_params
    gui_log(f"best GPR parameter: {gpr_params}")


    # 5 Strat Kfold
    svr_model = wrap(svr_rbf(**svr_params))
    gpr_model = gpr_rbf(**gpr_params)
    y_true = []
    y_pred_svr = []
    y_pred_gpr = []
    std_gpr = []
    bins = pd.qcut(y, q=5, duplicates="drop")
    bins = bins.codes
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(X, bins):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        svr_model.fit(Xtr, ytr)
        gpr_model.fit(Xtr, ytr)
        p_svr = svr_model.predict(Xte)
        p_gpr, p_std = gpr_model.predict(Xte, return_std=True)
        y_true.extend(yte.tolist())
        y_pred_svr.extend(p_svr.tolist())
        y_pred_gpr.extend(p_gpr.tolist())
        std_gpr.extend(p_std.tolist())
    # convert after loop
    y_true = np.array(y_true)
    y_pred_svr = np.array(y_pred_svr)
    y_pred_gpr = np.array(y_pred_gpr)
    std_gpr = np.array(std_gpr)


    # Weighted Ensemble Alpha Optimization dengan OPTUNA
    gui_log("Optimizing ensemble weight")
    def ensemble_objective(trial):
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        y_ens = alpha * y_pred_gpr + (1 - alpha) * y_pred_svr
        return mean_squared_error(y_true, y_ens)
    ens_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    ens_study.optimize(ensemble_objective, n_trials=80, show_progress_bar=False)
    best_alpha = ens_study.best_params["alpha"]
    gui_log(
        f"Best ensemble weight α (GPR weight): {best_alpha:.3f} | "
        f"SVR weight: {1-best_alpha:.3f}"
    )
    y_pred_weighted = (
        best_alpha * y_pred_gpr
        + (1 - best_alpha) * y_pred_svr
    )


    #  STACKING META-LEARNER
    meta_X = np.column_stack([y_pred_svr, y_pred_gpr])
    meta_y = y_true
    meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
    meta.fit(meta_X, meta_y)
    # stacking prediction = also OOF prediction used for scoring
    y_pred_stack = meta.predict(meta_X)
    # normalized stacking weights
    meta_coefs = meta.coef_
    meta_weights = meta_coefs / meta_coefs.sum() if meta_coefs.sum() != 0 else meta_coefs

    gui_log(f"Stacking Weights = {meta_weights}")



   
    # Metrics
    # core prediction arrays are numpy arrays
    y_true = np.asarray(y_true)
    y_pred_svr = np.asarray(y_pred_svr)
    y_pred_gpr = np.asarray(y_pred_gpr)
    y_pred_weighted = np.asarray(y_pred_weighted)
    y_pred_stack = np.asarray(y_pred_stack)
    std_gpr = np.asarray(std_gpr)

    #  metrics helper
    def metrics_report(y_true, y_pred):
        # convert to numpy avoid list/array problem
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mean_y = np.mean(np.abs(y_true)) if y_true.size else 1.0 
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100)
        mbe = float(np.mean(y_pred - y_true))
        r2 = float(r2_score(y_true, y_pred))

        rmse_pct = (rmse / mean_y) * 100 if mean_y != 0 else np.nan
        mae_pct = (mae / mean_y) * 100 if mean_y != 0 else np.nan
        mbe_pct = (mbe / mean_y) * 100 if mean_y != 0 else np.nan

        return {
            "MSE": float(mean_squared_error(y_true, y_pred)),
            "RMSE": rmse,
            "RMSE_%": rmse_pct,
            "MAE": mae,
            "MAE_%": mae_pct,
            "MBE": mbe,
            "MBE_%": mbe_pct,
            "MAPE": mape,
            "R2": r2
        }

    # Build results dict and DataFrame
    results = {
        "SVR_RBF": metrics_report(y_true, y_pred_svr),
        "GPR_RBF": metrics_report(y_true, y_pred_gpr),
        f"Weighted_Ensemble (α={best_alpha:.2f})": metrics_report(y_true, y_pred_weighted),
        "Stacking_Ensemble": metrics_report(y_true, y_pred_stack)
    }

    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(OUT_DIR / "metrics_dual_optuna_fixed.csv")
    gui_log(f"\n Final Results:\n {metrics_df}")

    
    # Feature Importance
    from sklearn.inspection import permutation_importance

    feature_names = ["Vc", "Fz", "ap"]
    importance_results = {}

    # SVR: Permutation Importance bara
    # Fit full SVR pipeline
    try:
        svr_final = wrap(svr_rbf(**svr_params)) if 'svr_params_clean' in globals() else wrap(svr_rbf(**svr_params))
        svr_final.fit(X, y)
        perm_result = permutation_importance(svr_final, X, y, n_repeats=10, random_state=RANDOM_STATE, scoring='neg_mean_squared_error')
        svr_importances = perm_result.importances_mean
        # normalize
        if svr_importances.sum() != 0:
            svr_importances = svr_importances / svr_importances.sum()
        else:
            svr_importances = np.zeros_like(svr_importances)
        importance_results["SVR (Permutation)"] = svr_importances
    except Exception as e:
        gui_log(f"Warning - SVR permutation importance failed: {e}")
        importance_results["SVR (Permutation)"] = np.zeros(len(feature_names))
        perm_result = None  

    def find_poly(estimator):
        """Search recursively for a fitted PolynomialFeatures inside any pipeline/wrapper."""
        if isinstance(estimator, PolynomialFeatures):
            return estimator
        if isinstance(estimator, Pipeline):
            for step_name, step_est in estimator.named_steps.items():
                result = find_poly(step_est)
                if result is not None:
                    return result
        if hasattr(estimator, "regressor_"):
            return find_poly(estimator.regressor_)
        if hasattr(estimator, "regressor"):
            return find_poly(estimator.regressor)
        return None

   
    # Permutation importance (returns final feature count
    perm_result = permutation_importance(
        svr_final, X, y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring='neg_mean_squared_error'
    )

    raw_importances = perm_result.importances_mean
    raw_stderr = perm_result.importances_std
    n_final = len(raw_importances) 
    # Extract fitted inner pipeline
    fitted_pipe = svr_final.regressor_      
    # get polynomial names
    poly = fitted_pipe.named_steps["poly"]
    # how many input features PolynomialFeatures actually saw
    n_poly_in = poly.n_features_in_
    # generate correct base names for polynomial features
    base_names = [f"x{i}" for i in range(n_poly_in)]
    # get correct feature names
    poly_names = poly.get_feature_names_out(base_names)

    poly_names = poly_names[:n_final]

    # group by original variables
    groups = {
        "Vc": [i for i, name in enumerate(poly_names) if "Vc" in name or "x0" in name],
        "Fz": [i for i, name in enumerate(poly_names) if "Fz" in name or "x1" in name],
        "ap": [i for i, name in enumerate(poly_names) if "ap" in name or "x2" in name],
    }

    gui_log(f"Groups (trimmed): {groups}")
    gui_log(f"raw_importances length: {n_final}")

    def safe_sum(arr, idx):
        return arr[idx].sum() if len(idx) > 0 else 0.0

    # collapse to 3 features
    svr_importances = np.array([
        safe_sum(raw_importances, groups["Vc"]),
        safe_sum(raw_importances, groups["Fz"]),
        safe_sum(raw_importances, groups["ap"])
    ])

    # normalize
    if svr_importances.sum() > 0:
        svr_importances /= svr_importances.sum()

    importance_results["SVR (Permutation)"] = svr_importances

    # stderr per group
    stderr = np.array([
        raw_stderr[groups["Vc"]].mean() if len(groups["Vc"]) else 0.0,
        raw_stderr[groups["Fz"]].mean() if len(groups["Fz"]) else 0.0,
        raw_stderr[groups["ap"]].mean() if len(groups["ap"]) else 0.0,
    ])
    importance_results["SVR (Permutation)"] = svr_importances



    # GPR: Inverse Length-scale Importance bara
    try:
        gpr_final = gpr_rbf(**gpr_params)
        gpr_final.fit(X, y)

        # recursive extraction of RBF kernel
        from sklearn.gaussian_process.kernels import RBF
        gpr_kernel = gpr_final.named_steps["gpr"].kernel_
        def extract_rbf_kernel(kernel):
            if isinstance(kernel, RBF):
                return kernel
            # composite kernels
            for attr in ("k1", "k2", "kernel", "kernels"):
                if hasattr(kernel, attr):
                    sub = getattr(kernel, attr)
                    res = extract_rbf_kernel(sub)
                    if res is not None:
                        return res
            # check tuple/list containers
            if isinstance(kernel, (list, tuple)):
                for k in kernel:
                    res = extract_rbf_kernel(k)
                    if res is not None:
                        return res
            return None

        rbf_kernel = extract_rbf_kernel(gpr_kernel)
        if rbf_kernel is not None:
            lscale = np.atleast_1d(rbf_kernel.length_scale)
            gpr_importances = (1.0 / lscale)
            if gpr_importances.sum() != 0:
                gpr_importances = gpr_importances / gpr_importances.sum()
            else:
                gpr_importances = np.zeros(len(feature_names))
        else:
            gpr_importances = np.zeros(len(feature_names))
        importance_results["GPR (1/LengthScale)"] = gpr_importances
    except Exception as e:
        gui_log(f"Warning - GPR importance extraction failed: {e}")
        importance_results["GPR (1/LengthScale)"] = np.zeros(len(feature_names))

    # Save
    imp_df = pd.DataFrame(importance_results, index=feature_names)
    imp_df.to_csv(OUT_DIR / "feature_importance.csv")
    gui_log(f"\n Feature Importance (normalized):\n {imp_df}")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(7,5))
    imp_df.plot(kind="bar", ax=ax)
    ax.set_title("Feature Sensitivity: SVR (Permutation) vs GPR (Inverse Length-Scale)")
    ax.set_ylabel("Normalized Importance")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    savefig(fig, "feature_importance.png")
    gui_log(" Feature importance plot saved.")


    # VISUALISASI

    def savefig(fig, name):
        path = FIG_DIR / name
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        gui_log(f"[Saved] {path}")

    # =========================================================
    # 1. DISTRIBUSI DATA EKSPERIMEN
    # =========================================================
    
    # Histogram Ra
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df["Ra"], bins=15, edgecolor="black")
    ax.set_title("Distribusi Kekasaran Permukaan (Ra)")
    ax.set_xlabel("Ra (µm)")
    ax.set_ylabel("Frekuensi")
    ax.grid(alpha=0.3)
    savefig(fig, "histogram_Ra.png")

    # Scatter Ra vs parameter pemesinan
    for param in ["Vc", "Fz", "ap"]:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(df[param], df["Ra"], edgecolor="black", alpha=0.7)
        ax.set_title(f"Hubungan Kekasaran Permukaan terhadap {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("Ra (µm)")
        ax.grid(alpha=0.3)
        savefig(fig, f"scatter_Ra_vs_{param}.png")

    # =========================================================
    # 2. ACTUAL vs PREDICTED Ra – SVR
    # =========================================================

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(y_true, y_pred_svr, edgecolor="black", alpha=0.7)

    mn, mx = y_true.min(), y_true.max()
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)

    ax.set_title("Actual vs Predicted Kekasaran Permukaan (SVR)")
    ax.set_xlabel("Actual Ra (µm)")
    ax.set_ylabel("Predicted Ra (µm)")
    ax.grid(alpha=0.3)
    savefig(fig, "actual_vs_predicted_SVR.png")

    # =========================================================
    # 3. ACTUAL vs PREDICTED Ra – GPR
    # =========================================================

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(y_true, y_pred_gpr, edgecolor="black", alpha=0.7)
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)

    ax.set_title("Actual vs Predicted Kekasaran Permukaan (GPR)")
    ax.set_xlabel("Actual Ra (µm)")
    ax.set_ylabel("Predicted Ra (µm)")
    ax.grid(alpha=0.3)
    savefig(fig, "actual_vs_predicted_GPR.png")

    # =========================================================
    # 4. PERBANDINGAN PREDIKSI MODEL TERHADAP DATA AKTUAL
    # =========================================================

    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(y_true, label="Actual", linewidth=2)
    ax.plot(y_pred_svr, "--", label="SVR", linewidth=1)
    ax.plot(y_pred_gpr, "--", label="GPR", linewidth=1)
    ax.plot(y_pred_weighted, label="Ensemble", linewidth=1.5)

    ax.set_title("Perbandingan Prediksi Model terhadap Data Aktual")
    ax.set_xlabel("Indeks Sampel")
    ax.set_ylabel("Ra (µm)")
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, "model_comparison.png")

    # =========================================================
    # 5. SENSITIVITAS PARAMETER PEMESINAN
    # =========================================================

    fig, ax = plt.subplots(figsize=(6,4))
    imp_df.mean(axis=1).plot(kind="bar", edgecolor="black", ax=ax)

    ax.set_title("Sensitivitas Parameter Pemesinan terhadap Kekasaran Permukaan")
    ax.set_xlabel("Parameter Pemesinan")
    ax.set_ylabel("Normalized Importance")
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, "sensitivitas_parameter.png")

    


global stop_time
stop_time = time.time()


def run_full_pipeline():
    gui_log("Running optimization")
    try:
        main_pipeline()  # your full training code
        gui_log("Optimization complete.")
    except Exception as e:
        gui_log(f"Error: {e}")

start_time = None
stop_time = None

# -----------------------------------------------------
# GUI DASHBOARD
# -----------------------------------------------------

gui_log_queue = queue.Queue()

def gui_log(msg):
    print(msg)
    gui_log_queue.put(str(msg) + "\n")


# -----------------------------------------------------
# GUI FUNCTIONS (define BEFORE buttons!)
# -----------------------------------------------------

def show_metrics():
    """Opens metrics CSV inside a popup window."""
    f = OUT_DIR / "metrics_dual_optuna_fixed.csv"
    if f.exists():
        df = pd.read_csv(f)
        win = tk.Toplevel(root)
        win.title("Metrics")
        txt = tk.Text(win, width=120, height=30)
        txt.pack(fill="both", expand=True)
        txt.insert(tk.END, df.to_string())
    else:
        gui_log("⚠ Metrics CSV not found.")


def show_plots():
    """Opens a scrollable window showing all saved plots from FIG_DIR."""
    files = sorted(glob.glob(str(FIG_DIR / "*.png")))
    if not files:
        gui_log("⚠ No plot images found.")
        return

    # TOP-LEVEL WINDOW
    win = tk.Toplevel(root)
    win.title("Plots Viewer")
    win.geometry("900x700")

    # OUTER FRAME
    outer = ttk.Frame(win)
    outer.pack(fill="both", expand=True)

    # CANVAS for scrolling
    canvas = tk.Canvas(outer)
    canvas.pack(side="left", fill="both", expand=True)

    # Vertical scrollbar
    v_scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    v_scroll.pack(side="right", fill="y")

    # Horizontal scrollbar
    h_scroll = ttk.Scrollbar(win, orient="horizontal", command=canvas.xview)
    h_scroll.pack(side="bottom", fill="x")

    canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

    # INTERNAL FRAME (holds plot images)
    inner = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")

    # Allow resizing content inside the canvas
    def update_scrollregion(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    inner.bind("<Configure>", update_scrollregion)

    # LOAD AND DISPLAY IMAGES
    for file in files:
        img = Image.open(file)
        img = img.resize((800, 600))  # nice display size
        img_tk = ImageTk.PhotoImage(img)

        lbl = tk.Label(inner, image=img_tk)
        lbl.image = img_tk  # prevent garbage collection
        lbl.pack(pady=10)

    gui_log(f"Loaded {len(files)} plots.")

def update_timer():
    """Updates timer label every second while optimization is running."""
    global elapsed_seconds
    if spinner_running:
        elapsed_seconds += 1
        h = elapsed_seconds // 3600
        m = (elapsed_seconds % 3600) // 60
        s = elapsed_seconds % 60
        timer_label.config(text=f"Run Time: {h:02d}:{m:02d}:{s:02d}")
        root.after(1000, update_timer)
    else:
        timer_label.config(text="Run Time: 00:00:00")
        elapsed_seconds = 0


# -----------------------------------------------------
# OPTUNA → GUI logging hook
# -----------------------------------------------------
class OptunaToGUI(logging.Handler):
    def emit(self, record):
        gui_log(self.format(record))

optuna_logger = logging.getLogger("optuna")
for handler in optuna_logger.handlers[:]:
    optuna_logger.removeHandler(handler)

optuna_logger.setLevel(logging.INFO)
optuna_logger.addHandler(OptunaToGUI())


# Prevent duplicate handlers
if not any(isinstance(h, OptunaToGUI) for h in optuna_logger.handlers):
    optuna_logger.addHandler(OptunaToGUI())
# -----------------------------------------------------
# SPINNER
# -----------------------------------------------------
spinner_running = False
spinner_cycle = ["B", "A", "R", "A", "B", "A", "R", "A", "B", "A", "R", "A"]
spinner_index = 0

def update_spinner():
    global spinner_index
    if spinner_running:
        spinner_label.config(text="Running" + spinner_cycle[spinner_index])
        spinner_index = (spinner_index + 1) % len(spinner_cycle)
        root.after(100, update_spinner)
    else:
        spinner_label.config(text="")

def update_timer():
    if start_time is not None and stop_time is None:
        elapsed = time.time() - start_time
        timer_label.config(text=f"Runtime: {elapsed:,.2f} sec")
    elif stop_time is not None:
        elapsed = stop_time - start_time
        timer_label.config(text=f"Final Runtime: {elapsed:,.2f} sec")
    root.after(200, update_timer)

def run_pipeline_thread():
    global spinner_running, start_time
    spinner_running = True
    start_time = time.time()   # <--- START TIMER HERE
    update_spinner()
    Thread(target=run_pipeline_safe).start()



    update_spinner()
    update_timer()

    Thread(target=run_pipeline_safe, daemon=True).start()



def run_pipeline_safe():
    global spinner_running, start_time, stop_time

    try:
        main_pipeline()

    except Exception as e:
        gui_log(f"ERROR: {e}")

    finally:
        spinner_running = False

        stop_time = time.time()

        if start_time is None:
            gui_log("[TIMER] ERROR: start_time not set!")
        else:
            duration = stop_time - start_time
            gui_log(f"[TIMER] Finished in {duration:.2f} sec")



def update_log():
    """Fetch queued log messages and push into GUI."""
    while not gui_log_queue.empty():
        log_text.insert(tk.END, gui_log_queue.get())
        log_text.see(tk.END)
    root.after(200, update_log)


# -----------------------------------------------------
# MAIN WINDOW
# -----------------------------------------------------
root = tk.Tk()
root.title("Skripsi Bara")
root.state("zoomed")
root.minsize(800, 500)

root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=0)
root.rowconfigure(2, weight=0)
root.columnconfigure(0, weight=1)

# MAIN FRAME
main = ttk.Frame(root)
main.grid(row=0, column=0, sticky="nsew")

main.rowconfigure(0, weight=1)
main.columnconfigure(0, weight=1)

# LOG TEXT BOX
log_text = tk.Text(main, bg="black", fg="lime", wrap="word")
log_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

# BUTTON FRAME
button_frame = ttk.Frame(root)
button_frame.grid(row=1, column=0, pady=5)

btn_run = ttk.Button(button_frame, text="Run ML", command=run_pipeline_thread)
btn_run.grid(row=0, column=0, padx=8)

btn_metrics = ttk.Button(button_frame, text="Metric Hasil", command=show_metrics)
btn_metrics.grid(row=0, column=1, padx=8)

btn_plots = ttk.Button(button_frame, text="Tampilkan Plot", command=show_plots)
btn_plots.grid(row=0, column=2, padx=8)

# SPINNER LABEL
spinner_label = ttk.Label(root, text="", font=("Consolas", 14))
spinner_label.grid(row=2, column=0, pady=5)

#TIME LABEL
timer_label = ttk.Label(root, text="Runtime: 0.00 sec", font=("Consolas", 13))
timer_label.grid(row=3, column=0, pady=5)


# ---------- GIF + TIMER BAR ----------
bottom_frame = ttk.Frame(root)
bottom_frame.grid(row=3, column=0, pady=10)

gif_label = ttk.Label(bottom_frame)
gif_label.grid(row=0, column=0, padx=10)

timer_label = ttk.Label(bottom_frame, text="Run Time: 00:00:00", font=("Consolas", 14))
timer_label.grid(row=0, column=1, padx=20)

update_log()
update_timer()
root.mainloop()



