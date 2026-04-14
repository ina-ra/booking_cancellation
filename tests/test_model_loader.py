import json
import pickle

from src.infrastructure.ml.model_loader import ModelRegistry


class DummyModel:
    def predict_proba(self, x):
        return [[0.2, 0.8]]


def test_model_registry_is_not_ready_before_load():
    registry = ModelRegistry()
    assert registry.is_ready() is False


def test_model_registry_load(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pkl"
    report_path = tmp_path / "report.json"

    with open(model_path, "wb") as file:
        pickle.dump(DummyModel(), file)

    report = {
        "categorical_columns": ["type of meal", "room type"],
        "best_model": {"name": "LightGBM"},
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file)

    monkeypatch.setattr(
        "src.infrastructure.ml.model_loader.settings",
        type(
            "FakeSettings",
            (),
            {
                "lightgbm_model_pickle_path": model_path,
                "model_report_path": report_path,
            },
        )(),
    )

    registry = ModelRegistry()
    registry.load()

    assert registry.model is not None
    assert registry.categorical_columns == ["type of meal", "room type"]
    assert registry.model_name == "LightGBM"
    assert registry.is_ready() is True