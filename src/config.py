from pathlib import Path

# =========================================================
# PROJECT PATHS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
METRIC_DIR = REPORT_DIR / "metrics"
FEATURE_DIR = REPORT_DIR / "feature_importance"
MODEL_DIR = ROOT_DIR / "models"
SAVED_MODEL_DIR = MODEL_DIR / "saved_models"
LOG_DIR = ROOT_DIR / "logs"
DRIFT_REPORT_DIR = REPORT_DIR / "drift"

# =========================================================
# DATASET
# =========================================================

# NOTE: earlier configs pointed at "Sheet2.csv", which does not exist in
# this repo. The dataset actually shipped under version control is
# data/raw/raw_data.csv - that is the real source of truth.
DATA_PATH = DATA_DIR / "raw" / "raw_data.csv"

# =========================================================
# MODEL PERSISTENCE
# =========================================================

# Fitted SVR/GPR pipelines + stacking meta-learner + ensemble weight,
# saved together as one joblib bundle so the API loads a single artifact.
MODEL_BUNDLE_PATH = SAVED_MODEL_DIR / "model_bundle.joblib"

# Human-readable copy of the bundle's metadata (metrics, params, feature
# list, training timestamp) so it can be inspected/served without touching
# pickled sklearn objects.
MODEL_METADATA_PATH = SAVED_MODEL_DIR / "model_metadata.json"

# Cleaned training data (raw machining parameters), kept as its own CSV so
# the drift monitor has a reference distribution to compare against without
# needing to unpickle the model bundle.
BASELINE_DATA_PATH = SAVED_MODEL_DIR / "training_baseline.csv"

# Where the API appends every prediction request it serves, one JSON object
# per line. The drift monitor can read this as its "production data" source.
PREDICTION_LOG_PATH = LOG_DIR / "prediction_log.jsonl"

# =========================================================
# RANDOMNESS
# =========================================================

RANDOM_STATE = 151101

# =========================================================
# FEATURE ENGINEERING
# =========================================================

POLY_DEGREE = 2

# =========================================================
# OPTUNA SETTINGS
# =========================================================

SVR_TRIALS = 60
GPR_TRIALS = 60
ENSEMBLE_TRIALS = 80

# =========================================================
# CROSS VALIDATION
# =========================================================

N_SPLITS = 5

# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
