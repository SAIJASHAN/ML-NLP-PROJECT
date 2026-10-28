"""Centralized project configuration and constants."""

from pathlib import Path

# Paths
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DRIFT_DATA_DIR = DATA_DIR / "drift"

MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODEL_DIR = MODELS_DIR / "baseline_v1"
REGISTRY_DIR = MODELS_DIR / "registry"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Remote BANKING77 Dataset URLs (PolyAI official repository)
BANKING77_TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
)
BANKING77_TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
)
BANKING77_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json"
)

# Expected BANKING77 dataset properties
EXPECTED_INTENT_COUNT = 77
EXPECTED_TRAIN_SAMPLES = 10003
EXPECTED_TEST_SAMPLES = 3080

# Modeling Hyperparameters
RANDOM_SEED = 42
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR_TF = True
LR_C = 10.0
LR_MAX_ITER = 1000
LR_CLASS_WEIGHT = "balanced"
LR_SOLVER = "lbfgs"

# Serving & Decision Thresholds
LOW_CONFIDENCE_THRESHOLD = 0.35
TOP_K_PREDICTIONS = 3

# Release Gate Criteria
RELEASE_GATE_MIN_MACRO_F1 = 0.82
RELEASE_GATE_MAX_F1_DROP = 0.005  # Max allowed degradation vs champion (0.5%)

# MLflow Tracking & Registry Configuration
DEFAULT_MLFLOW_DB_PATH = (PROJECT_ROOT / "mlflow.db").resolve().as_posix()
MLFLOW_TRACKING_URI = f"sqlite:///{DEFAULT_MLFLOW_DB_PATH}"
MLFLOW_EXPERIMENT_NAME = "ComplaintOps-Banking77"
MLFLOW_MODEL_NAME = "BankingIntentClassifier"

