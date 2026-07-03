# Configurations Directory

This folder stores configuration files intended for the project.

> **Current status:** none of the YAML files here are actually read by any
> code - `src/config.py` (the file the codebase actually imports paths and
> hyperparameter defaults from) uses plain Python constants instead, and
> the one YAML-loading function that existed (`src/utils/config_loader.py`)
> is an unimplemented stub. Some files also describe columns
> (`feed_rate`, `spindle_speed`) that don't match the real dataset's
> `Vc`/`Fz`/`ap`. Left as-is rather than guessing at intent; see
> `UPGRADE_NOTES.md` at the repo root. Wiring these up for real (e.g. via
> `src/utils/config_loader.py` -> `src/config.py`) would be a reasonable
> follow-up if you want training runs driven by config files instead of
> CLI flags.

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
