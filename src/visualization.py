import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURE_DIR


def save_figure(fig, filename):

    output_path = FIGURE_DIR / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"[Saved] {output_path}")


def plot_target_distribution(df):

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(
        df["Ra"],
        bins=15,
        edgecolor="black"
    )

    ax.set_title(
        "Surface Roughness Distribution"
    )

    ax.set_xlabel("Ra")
    ax.set_ylabel("Frequency")

    ax.grid(alpha=0.3)

    save_figure(
        fig,
        "target_distribution.png"
    )


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    model_name
):

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        y_true,
        y_pred,
        edgecolor="black",
        alpha=0.7
    )

    mn = min(y_true)
    mx = max(y_true)

    ax.plot(
        [mn, mx],
        [mn, mx],
        "k--",
        linewidth=1
    )

    ax.set_title(
        f"Actual vs Predicted ({model_name})"
    )

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    ax.grid(alpha=0.3)

    save_figure(
        fig,
        f"actual_vs_predicted_{model_name}.png"
    )


def plot_model_comparison(
    y_true,
    y_pred_svr,
    y_pred_gpr,
    y_pred_ensemble
):

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        y_true,
        label="Actual",
        linewidth=2
    )

    ax.plot(
        y_pred_svr,
        "--",
        label="SVR"
    )

    ax.plot(
        y_pred_gpr,
        "--",
        label="GPR"
    )

    ax.plot(
        y_pred_ensemble,
        label="Ensemble",
        linewidth=1.5
    )

    ax.set_title(
        "Model Prediction Comparison"
    )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Surface Roughness")

    ax.legend()

    ax.grid(alpha=0.3)

    save_figure(
        fig,
        "model_comparison.png"
    )


def plot_feature_importance(
    importance_df: pd.DataFrame
):

    fig, ax = plt.subplots(figsize=(7, 5))

    importance_df.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Feature Importance"
    )

    ax.set_ylabel(
        "Normalized Importance"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    save_figure(
        fig,
        "feature_importance.png"
    )
