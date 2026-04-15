import json
import pickle
from typing import Any, cast

from src.config import settings
from src.infrastructure.storage import artifact_storage


def ensure_s3_storage() -> None:
    if not artifact_storage.is_enabled():
        raise RuntimeError("S3 storage is required but not configured.")


def load_model_report() -> dict[str, Any]:
    ensure_s3_storage()
    payload = artifact_storage.download_bytes(settings.model_report_object_name)
    return cast(dict[str, Any], json.loads(payload.decode("utf-8")))


def load_pickled_model() -> Any:
    ensure_s3_storage()
    payload = artifact_storage.download_bytes(settings.lightgbm_model_pickle_object_name)
    return pickle.loads(payload)


def upload_training_artifacts(model: Any, report: dict[str, Any]) -> None:
    ensure_s3_storage()
    artifact_storage.upload_text(
        model.booster_.model_to_string(),
        settings.lightgbm_model_text_object_name,
        content_type="text/plain; charset=utf-8",
    )
    artifact_storage.upload_bytes(
        pickle.dumps(model),
        settings.lightgbm_model_pickle_object_name,
        content_type="application/octet-stream",
    )
    artifact_storage.upload_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        settings.model_report_object_name,
        content_type="application/json",
    )
