# Reports Directory

This folder stores all generated outputs from the machine learning pipeline.

---

# Structure

```text
reports/
├── figures/
├── metrics/
├── feature_importance/
├── drift/
└── README.md
```

---

# figures/

Contains generated visualization outputs, written by `src/train/train.py`
via `src/metrics/visualization.py`.

- actual vs predicted plots (one per model + ensemble)
- target distribution histogram
- model comparison plot

Format: `.png`.

---

# metrics/

Contains evaluation metrics and experiment summaries.

- `metrics_report.csv` - MSE, RMSE, MAE, MAPE, MBE, R² per model, written
  by `src/metrics/evaluation.py::generate_metrics_report`.

---

# feature_importance/

**Currently unused** (placeholder `.gitkeep` only). The original prototype
computed permutation importance (SVR) and inverse length-scale sensitivity
(GPR) here; that analysis wasn't reimplemented as part of the API/drift
monitoring upgrade - see `UPGRADE_NOTES.md` for why, and it's listed there
as a reasonable follow-up (e.g. a `/model/feature-importance` endpoint).

---

# drift/

Timestamped JSON reports from `python -m monitoring.drift_monitor`
(or `GET /drift/report`, which returns the same analysis without writing
a file). Not present in earlier versions of this project - added
alongside `monitoring/drift_monitor.py`.

---

# Note on request logs

The API's prediction request log (`prediction_log.jsonl`, read by the
drift monitor) lives at the **repo-root `logs/`** directory, not under
`reports/` - see `monitoring/request_log.py`.

---

# Notes

Generated reports should be reproducible from source code inside `src/`,
`app/`, and `monitoring/`.
