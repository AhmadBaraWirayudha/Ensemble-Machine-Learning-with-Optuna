# Tests Directory

This folder contains tests for the project: the original ML pipeline
tests plus new coverage for model persistence, the prediction API, and
drift monitoring added alongside those features.

---

# Structure

```text
tests/
├── test_data_loader.py    (original)
├── test_features.py       (original)
├── test_models.py         (original)
├── test_evaluation.py     (original)
├── test_objective.py      (original, placeholder)
├── test_preprocessing.py  (original, placeholder)
├── test_persistence.py    (new: model bundle save/load)
├── test_inference.py      (new: prediction helper used by the API)
├── test_api.py            (new: FastAPI endpoints)
├── test_auth.py                (new: opt-in API key authentication)
├── test_drift_monitor.py  (new: PSI/KS drift statistics)
├── test_feature_importance.py  (new: permutation importance)
├── test_retrain_trigger.py     (new: auto-retrain backup/promote/rollback)
├── test_storage.py             (new: SQLite backend - predictions, measurements, accuracy)
├── test_time3233_integration.py  (new: TIME3233 reader - export parsing, HTTP submission)
├── test_freecad_integration.py   (new: FreeCAD macro core - unit conversions, extraction logic)
└── README.md
```

---

# Test Coverage

| File | Purpose |
|---|---|
| `test_data_loader.py` | Dataset loading validation |
| `test_features.py` | Feature engineering validation |
| `test_models.py` | Model creation validation |
| `test_evaluation.py` | Metrics validation |
| `test_persistence.py` | Model bundle save/load round-trips correctly |
| `test_inference.py` | Prediction helper returns sane, deterministic outputs |
| `test_api.py` | API endpoints: health, predict, batch, validation, drift report, retrain history |
| `test_auth.py` | Auth is off by default; when `CNC_API_KEY` is set, every data endpoint requires it and `/health` never does |
| `test_drift_monitor.py` | PSI/KS drift statistics behave correctly on known distributions |
| `test_feature_importance.py` | Permutation importance ranks a known-informative feature correctly |
| `test_retrain_trigger.py` | Backup/restore round-trip, and promote-vs-rollback decision logic (training itself is mocked out so these run in ~1s instead of minutes) |
| `test_storage.py` | SQLite predictions/measurements round-trip; accuracy report joins correctly by job_id and picks the latest measurement on re-measure |
| `test_time3233_integration.py` | Ra-column detection heuristic, Excel/CSV export parsing, serial regex pattern, HTTP submission (mocked) |
| `test_freecad_integration.py` | Unit conversions (checked against hand calculations), HTTP client (mocked), attribute-extraction logic (against mock Path objects) |

Most of these tests run against real artifacts (the actual training CSV,
an actual trained model bundle) rather than mocks, matching how the
original tests in this directory were written. `test_persistence.py` and
`test_persistence.py::test_save_and_load_round_trip` additionally use a
`tmp_path` fixture so bundle save/load is verified in isolation too.
`test_api.py` and `test_auth.py` redirect the production database to a
temp file (via `monkeypatch`, since it's just a SQLite file path -
see `monitoring/storage.py`) so running the suite doesn't add fake
traffic to the real `logs/production.db` that the drift monitor and
accuracy report read from.

**Prerequisite:** `test_persistence.py`, `test_inference.py`, and the
model-dependent parts of `test_api.py` need a trained model to already
exist. Run `python scripts/train_model.py` once before testing if
`models/saved_models/` is empty. `test_retrain_trigger.py` mocks out
`src.train.main` entirely, so it doesn't need this and doesn't take
minutes to run. `test_storage.py`, `test_time3233_integration.py`, and
`test_freecad_integration.py` don't need it either - they use an isolated
temp database and mocked HTTP calls respectively.

---

# Running tests

Install (already included in the top-level `requirements.txt`):

```bash
pip install pytest httpx
```

Run all tests:

```bash
pytest
```

A `pytest.ini` at the repo root sets `pythonpath = .`, so this works from
the repo root regardless of invocation style (`pytest`, `python -m
pytest`, an IDE's test runner, etc.) - previously only `python -m pytest`
happened to work, because only `-m` invocation adds the current directory
to `sys.path`; plain `pytest` did not, so the command this file recommended
didn't actually run.

---

# Testing Goals

- Verify reproducibility
- Prevent regression bugs
- Validate preprocessing pipeline
- Validate model outputs
- Ensure metric correctness
- Validate the API's contract (status codes, validation, response shape)
- Validate drift detection behaves correctly on known stable/shifted data

