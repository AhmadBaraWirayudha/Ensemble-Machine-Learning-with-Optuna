"""
Flat-import compatibility shim.

The real implementation lives in src/tuning/optuna_tuning.py. Re-exported
here so `from src.optuna_tuning import tune_svr` etc. keep working.
"""

from src.tuning.optuna_tuning import (
    create_stratified_bins,
    tune_svr,
    tune_gpr,
    optimize_ensemble_weight,
)

__all__ = [
    "create_stratified_bins",
    "tune_svr",
    "tune_gpr",
    "optimize_ensemble_weight",
]
