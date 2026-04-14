import json
import pickle

from src.config import settings


class ModelRegistry:
    def __init__(self):
        self.model = None
        self.categorical_columns = []
        self.model_name = "unknown"

    def load(self):
        if not settings.lightgbm_model_pickle_path.exists():
            raise FileNotFoundError(f"Model file not found: {settings.lightgbm_model_pickle_path}")
        if not settings.model_report_path.exists():
            raise FileNotFoundError(f"Report file not found: {settings.model_report_path}")

        with open(settings.lightgbm_model_pickle_path, "rb") as file:
            self.model = pickle.load(file)
        with open(settings.model_report_path, "r", encoding="utf-8") as file:
            report = json.load(file)

        self.categorical_columns = report.get("categorical_columns", [])
        best_model = report.get("best_model", {})
        self.model_name = best_model.get("name", "LightGBM")

    def is_ready(self) -> bool:
        return self.model is not None


model_registry = ModelRegistry()

