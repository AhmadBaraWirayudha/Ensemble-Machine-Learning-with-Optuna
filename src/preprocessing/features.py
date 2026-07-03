import numpy as np
import pandas as pd

# Canonical feature set actually used to fit the ensemble. This matches the
# 12-feature matrix that was trained and evaluated in the original
# Untitled-2.py prototype (Vc, Fz, ap + squares + pairwise interactions +
# pairwise ratios). log_Vc/log_Fz/log_ap are computed below for exploratory
# use but were never part of the matrix the prototype actually trained on,
# so they're intentionally left out of the default model inputs here too -
# this keeps the served model faithful to the one whose metrics are in
# reports/, rather than silently retraining on a different feature set.
# Add them to this list (and retrain) if you want to experiment with them.
MODEL_FEATURE_COLUMNS = [
    "Vc",
    "Fz",
    "ap",
    "Vc2",
    "Fz2",
    "ap2",
    "Vc_Fz",
    "Vc_ap",
    "Fz_ap",
    "Vc_Fz_ratio",
    "Fz_ap_ratio",
    "Vc_ap_ratio",
]

# The raw machining parameters a caller (or the API) supplies.
RAW_INPUT_COLUMNS = ["Vc", "Fz", "ap"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate engineered machining features.
    """

    df = df.copy()

    # =====================================================
    # SQUARED FEATURES
    # =====================================================

    df["Vc2"] = df["Vc"] ** 2
    df["Fz2"] = df["Fz"] ** 2
    df["ap2"] = df["ap"] ** 2

    # =====================================================
    # INTERACTION FEATURES
    # =====================================================

    df["Vc_Fz"] = df["Vc"] * df["Fz"]
    df["Vc_ap"] = df["Vc"] * df["ap"]
    df["Fz_ap"] = df["Fz"] * df["ap"]

    # =====================================================
    # RATIO FEATURES
    # =====================================================

    epsilon = 1e-9

    df["Vc_Fz_ratio"] = df["Vc"] / (df["Fz"] + epsilon)
    df["Fz_ap_ratio"] = df["Fz"] / (df["ap"] + epsilon)
    df["Vc_ap_ratio"] = df["Vc"] / (df["ap"] + epsilon)

    # =====================================================
    # LOG FEATURES
    # =====================================================

    df["log_Vc"] = np.log(df["Vc"] + epsilon)
    df["log_Fz"] = np.log(df["Fz"] + epsilon)
    df["log_ap"] = np.log(df["ap"] + epsilon)

    return df


def prepare_feature_matrix(df: pd.DataFrame, feature_columns=None):
    """
    Prepare feature matrix X and target vector y from a dataframe that
    already has engineered columns (i.e. has been through
    add_engineered_features) and still has the "Ra" target column.
    """

    feature_columns = list(feature_columns or MODEL_FEATURE_COLUMNS)

    X = df[feature_columns].values
    y = df["Ra"].values

    return X, y, feature_columns


def build_feature_row(vc: float, fz: float, ap: float, feature_columns=None):
    """
    Build a single model-ready feature row from raw machining parameters.

    This runs the exact same add_engineered_features() code path used at
    training time, so a live prediction request is guaranteed to be
    engineered identically to the training data - no separate "inference
    feature logic" to accidentally let drift out of sync with training.

    Returns a 1-row numpy array shaped (1, n_features), ready to hand to
    a fitted pipeline's .predict().
    """

    feature_columns = list(feature_columns or MODEL_FEATURE_COLUMNS)

    row = pd.DataFrame([{"Vc": vc, "Fz": fz, "ap": ap}])
    row = add_engineered_features(row)

    return row[feature_columns].values
