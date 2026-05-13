# Models Directory

This folder stores trained machine learning models and related artifacts.

---

# Structure

```text
models/
├── saved_models/
├── checkpoints/
├── scalers/
└── README.md
```

---

# saved_models/

Stores finalized trained models.

Examples:

- SVR model
- GPR model
- ensemble model

Recommended formats:

- `.pkl`
- `.joblib`

---

# checkpoints/

Stores intermediate training checkpoints.

Useful for:

- long optimization runs
- experiment recovery
- incremental training

---

# scalers/

Stores preprocessing transformers.

Examples:

- RobustScaler
- PowerTransformer

---

# Notes

Avoid uploading extremely large model files directly to GitHub.

Recommended alternatives:

- Hugging Face Hub
- Google Drive
- GitHub Releases
