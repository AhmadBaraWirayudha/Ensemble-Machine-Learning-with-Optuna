# Runtime logs

- `prediction_log.jsonl` - every prediction the API has served. Empty
  until the first `/predict` request. See `monitoring/request_log.py`.
- `retrain_log.jsonl` - every auto-retrain attempt (skipped/promoted/
  rolled-back). Empty until `monitoring/retrain_trigger.py` first runs.

Not committed to git (see `.gitignore`); these `.gitkeep`/`README.md`
files just preserve the directory structure.
