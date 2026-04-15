import pickle

from src.infrastructure.ml.artifacts import (
    ensure_s3_storage,
    load_model_report,
    load_pickled_model,
    upload_training_artifacts,
)


class DummyModel:
    def predict_proba(self, x):
        return [[0.1, 0.9]]


class DummyBooster:
    def model_to_string(self):
        return "tree dump"


class UploadableModel:
    def __init__(self):
        self.booster_ = DummyBooster()


def test_ensure_s3_storage_raises_when_s3_is_disabled(monkeypatch):
    monkeypatch.setattr(
        "src.infrastructure.ml.artifacts.artifact_storage",
        type("FakeStorage", (), {"is_enabled": lambda self: False})(),
    )

    try:
        ensure_s3_storage()
    except RuntimeError as error:
        assert "S3 storage is required" in str(error)
    else:
        raise AssertionError("ensure_s3_storage() should raise when S3 is disabled")


def test_load_model_report_from_s3(monkeypatch):
    payload = b'{"best_model": {"name": "LightGBM"}}'

    monkeypatch.setattr(
        "src.infrastructure.ml.artifacts.settings",
        type("FakeSettings", (), {"model_report_object_name": "artifacts/model_report.json"})(),
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
    uploads = []
    model = UploadableModel()
    report = {"best_model": {"name": "LightGBM"}}

    monkeypatch.setattr(
        "src.infrastructure.ml.artifacts.settings",
        type(
            "FakeSettings",
            (),
            {
                "lightgbm_model_text_object_name": "artifacts/lightgbm_model.txt",
                "lightgbm_model_pickle_object_name": "artifacts/lightgbm_model.pkl",
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
                "upload_text": lambda self, payload, object_name, content_type: uploads.append(
                    ("text", payload, object_name, content_type)
                ),
                "upload_bytes": lambda self, payload, object_name, content_type: uploads.append(
                    ("bytes", payload, object_name, content_type)
                ),
            },
        )(),
    )

    upload_training_artifacts(model, report)

    assert uploads[0] == (
        "text",
        "tree dump",
        "artifacts/lightgbm_model.txt",
        "text/plain; charset=utf-8",
    )
    assert uploads[1][0] == "bytes"
    assert uploads[1][2] == "artifacts/lightgbm_model.pkl"
    assert uploads[1][3] == "application/octet-stream"
    assert uploads[2] == (
        "text",
        '{\n  "best_model": {\n    "name": "LightGBM"\n  }\n}',
        "artifacts/model_report.json",
        "application/json",
    )
