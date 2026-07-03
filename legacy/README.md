# Legacy prototype code

This folder holds the original Tkinter desktop prototype. It's kept for
history/reference, not as something to run.

- **Untitled-2.py** - the original monolithic script: data loading,
  feature engineering, Optuna tuning, the SVR/GPR/ensemble training, and
  the Tkinter GUI, all in one file. It never persisted a trained model to
  disk (no joblib/pickle call anywhere in it) - every GUI click retrained
  from scratch and only saved metrics CSVs and PNG plots. It also points
  at a hardcoded Windows path (`D:\SKRIPSI REBORN\...\Sheet2.csv`) that
  doesn't exist in this repo.
- **gui.py** - a cleaner Tkinter wrapper around `src.train.main`. As
  shipped it couldn't actually run either: `src/__init__.py` was named
  `src/_init_.py` (so `src` wasn't a real package), `src/config.py` was
  missing (misplaced at `src/tuning/config.py`), and every subpackage's
  `__init__.py` was empty, so none of the cross-module imports resolved.

All of that logic now lives in `src/` (fixed and working) plus
`scripts/train_model.py`, which additionally persists a trained model
bundle - something neither file here ever did. The desktop GUI itself has
been replaced by the REST API in `app/`; see the top-level README for how
to run it.
