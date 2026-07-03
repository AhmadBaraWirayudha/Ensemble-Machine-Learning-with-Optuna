# Dataset Directory

This folder stores all datasets used in the project.

---

# Structure

```text
data/
├── raw/
├── processed/
└── README.md
```

---

# raw/

Contains original datasets without modification.

Actual file in this repo: `raw_data.csv` (119 rows after cleaning; see
`src/preprocessing/data_loader.py`). Earlier config files in this repo
referred to a `Sheet2.csv` that was never actually present - `raw_data.csv`
has always been the real source of truth.

---

# processed/

Contains cleaned or feature-engineered datasets generated during preprocessing.

Examples:

- normalized datasets
- train/test split datasets
- feature-engineered datasets

**Currently unused** (placeholder `.gitkeep` only) - `load_dataset()` and
`add_engineered_features()` clean and engineer features in-memory on every
run rather than writing intermediate files here. `interim/` is the same:
present for future use, nothing writes to it yet.

---

# Notes

- Do not overwrite raw datasets.
- Keep processed datasets reproducible.
- Large datasets should not be committed directly to GitHub if they exceed repository limits.
