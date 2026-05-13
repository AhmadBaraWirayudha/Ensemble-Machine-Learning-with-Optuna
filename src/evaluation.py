import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from src.config import METRIC_DIR


def calculate_metrics(
    y_true,
    y_pred
):
    """
    Calculate regression metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_true, y_pred)

    mbe = np.mean(y_pred - y_true)

    r2 = r2_score(y_true, y_pred)

    mape = np.mean(
        np.abs(
            (y_true - y_pred)
            / np.maximum(np.abs(y_true), 1e-9)
        )
    ) * 100

    mean_y = np.mean(np.abs(y_true))

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "RMSE_%": float((rmse / mean_y) * 100),
        "MAE": float(mae),
        "MAE_%": float((mae / mean_y) * 100),
        "MBE": float(mbe),
        "MBE_%": float((mbe / mean_y) * 100),
        "MAPE": float(mape),
        "R2": float(r2)
    }


def generate_metrics_report(results_dict):

    metrics_df = pd.DataFrame(results_dict).T

    output_path = (
        METRIC_DIR
        / "metrics_report.csv"
    )

    metrics_df.to_csv(output_path)

    print("\nMetrics Summary")
    print(metrics_df)

    return metrics_df
