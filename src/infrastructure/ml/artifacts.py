import json
import pickle
from pathlib import Path
from typing import Any, cast

from src.config import settings
from src.infrastructure.storage import artifact_storage


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_binary_file(path: Path) -> bytes:
    return path.read_bytes()


def load_model_report() -> dict[str, Any]:
    if artifact_storage.is_enabled():
        payload = artifact_storage.download_bytes(settings.model_report_object_name)
        return cast(dict[str, Any], json.loads(payload.decode("utf-8")))

    if not settings.model_report_path.exists():
        raise FileNotFoundError(f"Report file not found: {settings.model_report_path}")

    return cast(dict[str, Any], json.loads(_read_text_file(settings.model_report_path)))


def load_pickled_model():
    if artifact_storage.is_enabled():
        payload = artifact_storage.download_bytes(settings.lightgbm_model_pickle_object_name)
        return pickle.loads(payload)

    if not settings.lightgbm_model_pickle_path.exists():
        raise FileNotFoundError(f"Model file not found: {settings.lightgbm_model_pickle_path}")

    return pickle.loads(_read_binary_file(settings.lightgbm_model_pickle_path))


def upload_training_artifacts():
    if not artifact_storage.is_enabled():
        return

    artifact_storage.upload_file(
        settings.lightgbm_model_text_path,
        settings.lightgbm_model_text_object_name,
    )
    artifact_storage.upload_file(
        settings.lightgbm_model_pickle_path,
        settings.lightgbm_model_pickle_object_name,
    )
    artifact_storage.upload_file(
        settings.model_report_path,
        settings.model_report_object_name,
    )
