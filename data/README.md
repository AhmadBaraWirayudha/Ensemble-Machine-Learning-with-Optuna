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

Example:

- `Sheet2.csv`

---

# processed/

Contains cleaned or feature-engineered datasets generated during preprocessing.

Examples:

- normalized datasets
- train/test split datasets
- feature-engineered datasets

---

# Notes

- Do not overwrite raw datasets.
- Keep processed datasets reproducible.
- Large datasets should not be committed directly to GitHub if they exceed repository limits.
