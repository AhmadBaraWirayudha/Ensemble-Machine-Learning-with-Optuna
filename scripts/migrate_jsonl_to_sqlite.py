#!/usr/bin/env python3
"""
One-time migration: import any existing logs/prediction_log.jsonl and
logs/retrain_log.jsonl (from before the SQLite storage swap - see
UPGRADE_NOTES.md) into logs/production.db. Safe to run even if those
files don't exist (nothing to do) or have already been migrated (records
are just appended again - run this once, not repeatedly, or pass
--dry-run first to check what it would do).

Usage:
    python scripts/migrate_jsonl_to_sqlite.py
    python scripts/migrate_jsonl_to_sqlite.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PREDICTION_LOG_PATH, RETRAIN_LOG_PATH
from monitoring.storage import log_prediction, log_retrain_event


def _read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report what would be migrated without writing anything")
    args = parser.parse_args()

    predictions = _read_jsonl(PREDICTION_LOG_PATH)
    retrain_events = _read_jsonl(RETRAIN_LOG_PATH)

    print(f"Found {len(predictions)} prediction(s) in {PREDICTION_LOG_PATH}")
    print(f"Found {len(retrain_events)} retrain event(s) in {RETRAIN_LOG_PATH}")

    if args.dry_run:
        print("--dry-run set, not writing anything.")
        return 0

    for record in predictions:
        log_prediction(
            vc=record.get("Vc"), fz=record.get("Fz"), ap=record.get("ap"),
            prediction=record, job_id=record.get("job_id"), source="jsonl_migration",
        )

    for record in retrain_events:
        log_retrain_event(record)

    print(f"Migrated {len(predictions)} prediction(s) and {len(retrain_events)} retrain event(s) into SQLite.")
    print("The old .jsonl files were not deleted - remove them by hand once you've confirmed the migration looks right.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
