"""
Prediction logging - thin re-export over monitoring/storage.py (SQLite).

This used to be its own JSONL-based implementation; kept as a separate
module (rather than having every caller import storage.py directly) so
existing import sites (`from monitoring.request_log import log_prediction`)
didn't need to change when the backend did. See UPGRADE_NOTES.md for why
JSONL was swapped for SQLite.
"""

from monitoring.storage import log_prediction, read_prediction_log

__all__ = ["log_prediction", "read_prediction_log"]
