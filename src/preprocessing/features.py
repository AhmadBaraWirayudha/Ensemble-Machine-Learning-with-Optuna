import numpy as np
import pandas as pd


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


def prepare_feature_matrix(df: pd.DataFrame):
    """
    Prepare feature matrix X and target vector y.
    """

    feature_columns = [
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
        "log_Vc",
        "log_Fz",
        "log_ap",
    ]

    X = df[feature_columns].values
    y = df["Ra"].values

    return X, y, feature_columns
