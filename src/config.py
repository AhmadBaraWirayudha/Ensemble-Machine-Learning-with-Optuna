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

# =========================================================
# DATASET
# =========================================================

DATA_PATH = DATA_DIR / "Sheet2.csv"

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
