from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "lightgbm_model.pkl"
REPORT_PATH = ARTIFACTS_DIR / "model_comparison.json"

DEFAULT_HIGH_RISK_THRESHOLD = 0.7
DEFAULT_BATCH_RISK_SHARE = 0.3