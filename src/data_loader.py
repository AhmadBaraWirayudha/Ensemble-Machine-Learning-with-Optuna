"""
Flat-import compatibility shim.

The real implementation lives in src/preprocessing/data_loader.py. This
module re-exports it so `from src.data_loader import load_dataset` keeps
working for existing tests, scripts, and the training entrypoint.
"""

from src.preprocessing.data_loader import load_dataset

__all__ = ["load_dataset"]
