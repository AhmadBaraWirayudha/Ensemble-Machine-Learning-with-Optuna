"""
Flat-import compatibility shim.

The real implementation lives in src/preprocessing/features.py. Re-exported
here so `from src.features import add_engineered_features` keeps working.
"""

from src.preprocessing.features import (
    add_engineered_features,
    prepare_feature_matrix,
    build_feature_row,
    MODEL_FEATURE_COLUMNS,
    RAW_INPUT_COLUMNS,
)

__all__ = [
    "add_engineered_features",
    "prepare_feature_matrix",
    "build_feature_row",
    "MODEL_FEATURE_COLUMNS",
    "RAW_INPUT_COLUMNS",
]
