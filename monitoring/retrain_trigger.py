#!/usr/bin/env python3
"""
Closes the loop between the two halves of this upgrade: check
monitoring/drift_monitor.py's verdict, and if it says DRIFT_DETECTED,
retrain and redeploy - automatically, with a rollback safety net rather
than blindly trusting that a retrain is always an improvement.

Usage
-----
  # Check drift against the API's request log; retrain only if needed
  python -m monitoring.retrain_trigger

  # See what it would do without actually retraining
  python -m monitoring.retrain_trigger --dry-run

  # Retrain regardless of the drift verdict (e.g. a scheduled weekly job)
  python -m monitoring.retrain_trigger --force-retrain

Safety net: before overwriting the current model, the previous bundle is
copied to models/saved_models/archive/<timestamp>/. After retraining, the
new model's out-of-fold RMSE is compared against the old one's recorded
RMSE. If it's more than --max-regression-pct worse, the old bundle is
restored from that backup rather than promoting a worse model - an
automated pipeline silently deploying a regression is worse than it doing
nothing, especially for a quality-prediction model. Every attempt (skipped,
promoted, or rolled back) is recorded in the retrain_events table of
logs/production.db (see monitoring/storage.py).

Exit codes: 0 = no action needed / retrain promoted successfully,
1 = retrained but rolled back a regression, 2 = error.
"""

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    SAVED_MODEL_DIR,
    MODEL_ARCHIVE_DIR,
    MODEL_BUNDLE_PATH,
)
from src.models.persistence import load_metadata, load_baseline_data
from monitoring.drift_monitor import analyze_drift, read_prediction_log, MIN_SAMPLES
from monitoring.storage import log_retrain_event, read_retrain_log

ARCHIVED_FILENAMES = [
    "model_bundle.joblib",
    "model_metadata.json",
    "training_baseline.csv",
    "feature_importance.json",
]


def backup_current_model() -> Optional[Path]:
    """Copy the current model bundle + sidecar files to a timestamped
    archive folder. Returns the archive path, or None if there was no
    existing model to back up (e.g. first-ever training run)."""

    if not MODEL_BUNDLE_PATH.exists():
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = MODEL_ARCHIVE_DIR / ts
    archive_path.mkdir(parents=True, exist_ok=True)

    for filename in ARCHIVED_FILENAMES:
        src = SAVED_MODEL_DIR / filename
        if src.exists():
            shutil.copy2(src, archive_path / filename)

    return archive_path


def restore_from_backup(archive_path: Path) -> None:
    """Roll back by copying archived files back over the current ones."""

    for filename in ARCHIVED_FILENAMES:
        backup_file = archive_path / filename
        if backup_file.exists():
            shutil.copy2(backup_file, SAVED_MODEL_DIR / filename)


def get_recommended_rmse(metadata: dict) -> float:
    return metadata["metrics"][metadata["recommended_model"]]["RMSE"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, help="CSV of production readings to check for drift (default: the API's request log)")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES, help="Minimum samples needed before drift can be assessed")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick", help="Training preset to use for the retrain (default: quick)")
    parser.add_argument("--max-regression-pct", type=float, default=15.0, help="Roll back if the new model's RMSE is worse than the old one's by more than this percent (default: 15)")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain regardless of the drift verdict (e.g. a periodic scheduled job)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen without actually retraining")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    def log(msg):
        if not args.quiet:
            print(msg, flush=True)

    # --- Step 1: what does the drift monitor say? ---
    if not args.force_retrain:
        baseline_df = load_baseline_data()
        current_df = (
            pd.read_csv(args.input) if args.input else read_prediction_log()
        )

        if len(current_df) < args.min_samples:
            log(f"Only {len(current_df)} samples logged (need {args.min_samples}) - nothing to check yet, skipping.")
            log_retrain_event({"action": "skipped", "reason": "insufficient_data", "n_samples": len(current_df)})
            return 0

        analysis = analyze_drift(baseline_df, current_df)
        log(f"Drift verdict: {analysis['overall_verdict']}")

        if analysis["overall_verdict"] != "DRIFT_DETECTED":
            log("No significant drift - not retraining.")
            log_retrain_event({"action": "skipped", "reason": "no_drift", "drift_verdict": analysis["overall_verdict"]})
            return 0
    else:
        log("--force-retrain set, skipping drift check.")
        analysis = {"overall_verdict": "FORCED"}

    if args.dry_run:
        log(f"[dry-run] Would retrain now (preset={args.preset}) and evaluate for promotion. Stopping here.")
        log_retrain_event({"action": "dry_run", "drift_verdict": analysis["overall_verdict"]})
        return 0

    # --- Step 2: back up the current model before touching it ---
    old_metadata = None
    try:
        old_metadata = load_metadata()
    except FileNotFoundError:
        pass  # first-ever training run - nothing to compare against or back up

    archive_path = backup_current_model()
    if archive_path:
        log(f"Backed up current model to {archive_path}")

    # --- Step 3: retrain ---
    log(f"Retraining (preset={args.preset})...")
    from src.train import main as run_training

    result = run_training(preset=args.preset, verbose=not args.quiet)
    new_rmse = result["metrics"][result["recommended_model"]]["RMSE"]

    # --- Step 4: promote, or roll back if it's a regression ---
    if old_metadata is None:
        log(f"No previous model to compare against - promoting (RMSE={new_rmse:.4f}).")
        log_retrain_event({
            "action": "promoted", "reason": "first_model",
            "drift_verdict": analysis["overall_verdict"], "new_rmse": new_rmse,
        })
        return 0

    old_rmse = get_recommended_rmse(old_metadata)
    pct_change = ((new_rmse - old_rmse) / old_rmse) * 100 if old_rmse else 0.0

    if pct_change > args.max_regression_pct:
        log(f"New model is {pct_change:.1f}% worse (RMSE {old_rmse:.4f} -> {new_rmse:.4f}) "
            f"- exceeds --max-regression-pct={args.max_regression_pct}. Rolling back.")
        restore_from_backup(archive_path)
        log_retrain_event({
            "action": "rolled_back", "reason": "regression",
            "drift_verdict": analysis["overall_verdict"],
            "old_rmse": old_rmse, "new_rmse": new_rmse, "pct_change": pct_change,
        })
        return 1

    log(f"New model RMSE {old_rmse:.4f} -> {new_rmse:.4f} ({pct_change:+.1f}%) - promoting.")
    log_retrain_event({
        "action": "promoted", "reason": "improved_or_acceptable",
        "drift_verdict": analysis["overall_verdict"],
        "old_rmse": old_rmse, "new_rmse": new_rmse, "pct_change": pct_change,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
