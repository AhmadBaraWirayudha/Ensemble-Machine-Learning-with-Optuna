"""
Flat-import compatibility shim, matching the pattern of src/data_loader.py,
src/features.py, etc. The real implementation lives in
src/metrics/feature_importance.py.
"""

from src.metrics.feature_importance import (
    compute_permutation_importance,
    aggregate_by_base_variable,
    compute_all_feature_importance,
    BASE_VARIABLES,
)

__all__ = [
    "compute_permutation_importance",
    "aggregate_by_base_variable",
    "compute_all_feature_importance",
    "BASE_VARIABLES",
]
