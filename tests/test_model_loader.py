from src.infrastructure.ml.model_loader import ModelRegistry


class DummyModel:
    def predict_proba(self, x):
        return [[0.2, 0.8]]


def test_model_registry_is_not_ready_before_load():
    registry = ModelRegistry()
    assert registry.is_ready() is False


def test_model_registry_load(monkeypatch):
    monkeypatch.setattr(
        "src.infrastructure.ml.model_loader.load_pickled_model",
        lambda: DummyModel(),
    )
    monkeypatch.setattr(
        "src.infrastructure.ml.model_loader.load_model_report",
        lambda: {
            "categorical_columns": ["type of meal", "room type"],
            "best_model": {"name": "LightGBM"},
        },
    )

    registry = ModelRegistry()
    registry.load()

    assert registry.model is not None
    assert registry.categorical_columns == ["type of meal", "room type"]
    assert registry.model_name == "LightGBM"
    assert registry.is_ready() is True
