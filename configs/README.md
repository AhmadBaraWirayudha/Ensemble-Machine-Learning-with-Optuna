# Configurations Directory

This folder stores configuration files used throughout the project.

---

# Structure

```text
configs/
├── model_config.yaml
├── training_config.yaml
├── paths_config.yaml
└── README.md
```

---

# model_config.yaml

Stores machine learning model settings.

Examples:

- kernel types
- polynomial degree
- scaling methods
- ensemble configuration

---

# training_config.yaml

Stores training parameters.

Examples:

- random seed
- number of folds
- Optuna trials
- validation settings

---

# paths_config.yaml

Stores directory and file paths.

Examples:

- dataset path
- report directory
- figure directory
- model save path

---

# Benefits

Using configuration files improves:

- reproducibility
- modularity
- scalability
- experiment tracking
