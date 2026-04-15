import json
import pickle
from pathlib import Path

from src.infrastructure.ml.artifacts import (
    load_model_report,
    load_pickled_model,
    upload_training_artifacts,
)


class DummyModel:
    def predict_proba(self, x):
        return [[0.1, 0.9]]


def test_load_model_report_from_local_file(monkeypatch):
    report_path = Path("artifacts/model_report.json")

    monkeypatch.setattr(
        "src.infrastructure.ml.artifacts.settings",
        type("FakeSettings", (), {"model_report_path": report_path})(),
    )
    monkeypatch.setattr(
        "src.infrastructure.ml.artifacts.artifact_storage",
        type("FakeStorage", (), {"is_enabled": lambda self: False})(),
    )
    monkeypatch.setattr(
        "src.infrastructure.ml.artifacts._read_text_file",
        lambda path: json.dumps({"best_model": {"name": "LightGBM"}}),
    )
    monkeypatch.setattr(Path, "exists", lambda self: self == report_path)

    report = load_model_report()

    assert report["best_model"]["name"] == "LightGBM"


def test_load_pickled_model_from_s3(monkeypatch):
    payload = pickle.dumps(DummyModel())

    monkeypatch.setattr(
        "src.infrastructure.ml.artifacts.settings",
        type("FakeSettings", (), {"lightgbm_model_pickle_object_name": "artifacts/model.pkl"})(),
    )
    monkeypatch.setattr(
        "src.infrastructure.ml.artifacts.artifact_storage",
        type(
            "FakeStorage",
            (),
            {
                "is_enabled": lambda self: True,
                "download_bytes": lambda self, object_name: payload,
            },
        )(),
    )

    model = load_pickled_model()

    assert isinstance(model, DummyModel)


def test_upload_training_artifacts_to_s3(monkeypatch):
    text_path = Path("artifacts/lightgbm_model.txt")
    pickle_path = Path("artifacts/lightgbm_model.pkl")
    report_path = Path("artifacts/model_report.json")

    uploads = []

    monkeypatch.setattr(
        "src.infrastructure.ml.artifacts.settings",
        type(
            "FakeSettings",
            (),
            {
                "lightgbm_model_text_path": text_path,
                "lightgbm_model_text_object_name": "artifacts/lightgbm_model.txt",
                "lightgbm_model_pickle_path": pickle_path,
                "lightgbm_model_pickle_object_name": "artifacts/lightgbm_model.pkl",
                "model_report_path": report_path,
                "model_report_object_name": "artifacts/model_report.json",
            },
        )(),
    )
    monkeypatch.setattr(
        "src.infrastructure.ml.artifacts.artifact_storage",
        type(
            "FakeStorage",
            (),
            {
                "is_enabled": lambda self: True,
                "upload_file": lambda self, local_path, object_name: uploads.append(
                    (local_path.name, object_name)
                ),
            },
        )(),
    )

    upload_training_artifacts()

    assert uploads == [
        ("lightgbm_model.txt", "artifacts/lightgbm_model.txt"),
        ("lightgbm_model.pkl", "artifacts/lightgbm_model.pkl"),
        ("model_report.json", "artifacts/model_report.json"),
    ]
