from pathlib import Path

from src.config import Settings, build_settings


def test_build_settings_returns_settings_instance():
    settings = build_settings()
    assert isinstance(settings, Settings)


def test_build_settings_paths_and_defaults():
    settings = build_settings()

    assert isinstance(settings.base_dir, Path)
    assert settings.data_dir.name == "data"
    assert settings.artifacts_dir.name == "artifacts"
    assert settings.docs_dir.name == "docs"

    assert settings.target_column == "booking status"
    assert settings.id_column == "Booking_ID"
    assert settings.date_column == "date of reservation"

    assert isinstance(settings.default_high_risk_threshold, float)
    assert isinstance(settings.default_batch_risk_share, float)
    assert isinstance(settings.random_state, int)
    assert isinstance(settings.test_size, float)


def test_build_settings_s3_configuration(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("S3_BUCKET", "booking-cancellation-artifacts")
    monkeypatch.setenv("S3_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("S3_SECRET_KEY", "minioadmin")
    monkeypatch.setenv("S3_ARTIFACTS_PREFIX", "ml-artifacts")
    monkeypatch.setenv("S3_BATCH_OUTPUTS_PREFIX", "batch-runs")
    monkeypatch.setenv("S3_AUTO_CREATE_BUCKET", "true")
    monkeypatch.setenv("S3_USE_PATH_STYLE", "true")

    settings = build_settings()

    assert settings.s3_enabled is True
    assert settings.lightgbm_model_pickle_object_name == "ml-artifacts/lightgbm_model.pkl"
    assert settings.model_report_object_name == "ml-artifacts/model_comparison.json"
    assert settings.batch_outputs_prefix == "batch-runs"
    assert settings.s3_auto_create_bucket is True
    assert settings.s3_use_path_style is True
