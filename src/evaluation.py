"""
Flat-import compatibility shim.

The real implementation lives in src/metrics/evaluation.py. Re-exported
here so `from src.evaluation import calculate_metrics` keeps working.
"""

from src.metrics.evaluation import calculate_metrics, generate_metrics_report

__all__ = ["calculate_metrics", "generate_metrics_report"]
