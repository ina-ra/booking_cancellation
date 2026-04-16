import json

import pandas as pd

from src.config import settings
from src.infrastructure.storage.s3 import artifact_storage


def build_run_prefix(run_date: str) -> str:
    return f"{settings.batch_outputs_prefix}/{run_date}"


def build_scores_object_name(run_date: str) -> str:
    return f"{build_run_prefix(run_date)}/booking_risk_scores.csv"


def build_high_risk_object_name(run_date: str) -> str:
    return f"{build_run_prefix(run_date)}/high_risk_bookings.csv"


def build_success_marker_object_name(run_date: str) -> str:
    return f"{build_run_prefix(run_date)}/_SUCCESS.json"


def batch_run_exists(run_date: str) -> bool:
    if not settings.s3_enabled:
        return False

    return artifact_storage.object_exists(build_success_marker_object_name(run_date))


def upload_batch_outputs(
    run_date: str,
    scoring_table: pd.DataFrame,
    high_risk_table: pd.DataFrame,
) -> None:
    if not settings.s3_enabled:
        return

    artifact_storage.upload_text(
        scoring_table.to_csv(index=False),
        build_scores_object_name(run_date),
        content_type="text/csv; charset=utf-8",
    )
    artifact_storage.upload_text(
        high_risk_table.to_csv(index=False),
        build_high_risk_object_name(run_date),
        content_type="text/csv; charset=utf-8",
    )
    artifact_storage.upload_text(
        json.dumps({"run_date": run_date, "status": "success"}, ensure_ascii=False, indent=2),
        build_success_marker_object_name(run_date),
        content_type="application/json",
    )
