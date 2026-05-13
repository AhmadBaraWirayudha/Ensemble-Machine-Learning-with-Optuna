# Tests Directory

This folder contains unit tests for the project.

---

# Structure

```text
tests/
├── test_data_loader.py
├── test_features.py
├── test_models.py
├── test_evaluation.py
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

---

# Recommended Testing Framework

Use:

```bash
pytest
```

Install:

```bash
pip install pytest
```

Run all tests:

```bash
pytest
```

---

# Testing Goals

- Verify reproducibility
- Prevent regression bugs
- Validate preprocessing pipeline
- Validate model outputs
- Ensure metric correctness
