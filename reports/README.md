# Reports Directory

This folder stores all generated outputs from the machine learning pipeline.

---

# Structure

```text
reports/
├── figures/
├── metrics/
├── feature_importance/
├── logs/
└── README.md
```

---

# figures/

Contains generated visualization outputs.

Examples:

- actual vs predicted plots
- histogram distributions
- feature importance plots
- model comparison plots

Recommended formats:

- `.png`
- `.jpg`
- `.svg`

---

# metrics/

Contains evaluation metrics and experiment summaries.

Examples:

- RMSE
- MAE
- MAPE
- R² score

Recommended formats:

- `.csv`
- `.xlsx`

---

# feature_importance/

Contains feature sensitivity and importance outputs.

Examples:

- permutation importance
- inverse length-scale sensitivity
- normalized importance scores

---

# logs/

Stores training and optimization logs.

Examples:

- Optuna logs
- runtime logs
- training history

---

# Notes

Generated reports should be reproducible from source code inside `src/`.
