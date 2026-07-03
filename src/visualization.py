"""
Flat-import compatibility shim.

The real implementation lives in src/metrics/visualization.py. Re-exported
here so `from src.visualization import plot_target_distribution` etc. keep
working.
"""

from src.metrics.visualization import (
    save_figure,
    plot_target_distribution,
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_feature_importance,
)

__all__ = [
    "save_figure",
    "plot_target_distribution",
    "plot_actual_vs_predicted",
    "plot_model_comparison",
    "plot_feature_importance",
]
