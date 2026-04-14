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