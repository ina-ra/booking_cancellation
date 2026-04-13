import json
import pickle

from app.core.config import MODEL_PATH, REPORT_PATH


class ModelRegistry:
    def __init__(self):
        self.model = None
        self.categorical_columns = []
        self.model_name = "unknown"

    def load(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        if not REPORT_PATH.exists():
            raise FileNotFoundError(f"Report file not found: {REPORT_PATH}")

        with open(MODEL_PATH, "rb") as file:
            self.model = pickle.load(file)

        with open(REPORT_PATH, "r", encoding="utf-8") as file:
            report = json.load(file)

        self.categorical_columns = report.get("categorical_columns", [])
        best_model = report.get("best_model", {})
        self.model_name = best_model.get("name", "LightGBM")

    def is_ready(self) -> bool:
        return self.model is not None


model_registry = ModelRegistry()