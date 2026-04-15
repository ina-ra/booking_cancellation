from src.infrastructure.ml.artifacts import load_model_report, load_pickled_model


class ModelRegistry:
    def __init__(self):
        self.model = None
        self.categorical_columns = []
        self.model_name = "unknown"

    def load(self):
        self.model = load_pickled_model()
        report = load_model_report()

        self.categorical_columns = report.get("categorical_columns", [])
        best_model = report.get("best_model", {})
        self.model_name = best_model.get("name", "LightGBM")

    def is_ready(self) -> bool:
        return self.model is not None


model_registry = ModelRegistry()

