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

Permutation importance, written by `src/train/train.py` via
`src/metrics/feature_importance.py`:

- `feature_importance_detailed.csv` - per engineered feature (12 rows), one column per model (SVR, GPR, Random Forest, Gradient Boosting)
- `feature_importance_by_variable.csv` - rolled up to the raw Vc/Fz/ap parameter each feature derives from

The same data is served by the API at `GET /model/feature-importance`
(from `models/saved_models/feature_importance.json`, saved alongside the
CSVs here). The original prototype computed something similar but inside
the polynomial-expanded feature space, requiring fragile reverse-mapping
of auto-generated names back to the original variables - this version
runs permutation importance on the whole fitted pipeline instead, so that
mapping step isn't needed. See `UPGRADE_NOTES.md`.

---

# drift/

Timestamped JSON reports from `python -m monitoring.drift_monitor`
(or `GET /drift/report`, which returns the same analysis without writing
a file). Not present in earlier versions of this project - added
alongside `monitoring/drift_monitor.py`.

---

# Note on request/measurement data

The API's prediction, measurement, and retrain-event history all live in
`logs/production.db` (SQLite) at the **repo-root `logs/`** directory, not
under `reports/` - see `monitoring/storage.py`.

---

# Notes

Generated reports should be reproducible from source code inside `src/`,
`app/`, and `monitoring/`.
