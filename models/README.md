# Models Directory

This folder stores trained machine learning models and related artifacts.

> **Actual structure differs slightly from the original plan below:**
> `saved_models/` is real and populated by `python scripts/train_model.py`
> (via `src/models/persistence.py`). `checkpoints/` and `scalers/` were
> part of the original plan but were never needed in practice - the
> RobustScaler/PowerTransformer steps are fitted *inside* each model's
> sklearn `Pipeline` (see `src/models/models.py`) and saved as part of
> that pipeline object, rather than as separate standalone artifacts, and
> training here is fast enough (a couple of minutes on the default preset)
> that checkpointing intermediate state hasn't been necessary.

---

# saved_models/

Populated by `python scripts/train_model.py`:

| File | Contents |
|---|---|
| `model_bundle.joblib` | Fitted SVR pipeline, fitted GPR pipeline, RidgeCV stacking meta-learner, ensemble weight (`alpha`), feature list, hyperparameters, metrics, training timestamp - everything `app/main.py` loads at startup. |
| `model_metadata.json` | The same metadata as above, minus the model objects themselves - human-readable, and what `GET /model/info` returns. |
| `training_baseline.csv` | The cleaned training data (`Vc`, `Fz`, `ap`, `Ra`), used by `monitoring/drift_monitor.py` as its reference distribution. |

See `src/models/persistence.py` for the save/load code and
`UPGRADE_NOTES.md` for why this didn't exist before this upgrade (nothing
in the original prototype ever called `joblib.dump`).

---

# Notes

Avoid uploading extremely large model files directly to GitHub.

Recommended alternatives:

- Hugging Face Hub
- Google Drive
- GitHub Releases

(Not a concern at the current size - the whole bundle is under 1 MB - but
worth knowing if this grows into a larger model.)
