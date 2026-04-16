import pandas as pd

from src.infrastructure.storage.batch_runs import (
    batch_run_exists,
    build_high_risk_object_name,
    build_scores_object_name,
    build_success_marker_object_name,
    upload_batch_outputs,
)


def test_batch_run_object_names_include_run_date():
    assert build_scores_object_name("2026-04-16") == (
        "batch-runs/2026-04-16/booking_risk_scores.csv"
    )
    assert build_high_risk_object_name("2026-04-16") == (
        "batch-runs/2026-04-16/high_risk_bookings.csv"
    )
    assert build_success_marker_object_name("2026-04-16") == "batch-runs/2026-04-16/_SUCCESS.json"


def test_batch_run_exists_checks_success_marker(monkeypatch):
    fake_settings = type(
        "FakeSettings",
        (),
        {"s3_enabled": True, "batch_outputs_prefix": "batch-runs"},
    )()
    monkeypatch.setattr("src.infrastructure.storage.batch_runs.settings", fake_settings)
    monkeypatch.setattr(
        "src.infrastructure.storage.batch_runs.artifact_storage.object_exists",
        lambda object_name: object_name == "batch-runs/2026-04-16/_SUCCESS.json",
    )

    assert batch_run_exists("2026-04-16") is True


def test_upload_batch_outputs_writes_three_objects(monkeypatch):
    uploads = []
    fake_settings = type(
        "FakeSettings",
        (),
        {"s3_enabled": True, "batch_outputs_prefix": "batch-runs"},
    )()
    monkeypatch.setattr("src.infrastructure.storage.batch_runs.settings", fake_settings)
    monkeypatch.setattr(
        "src.infrastructure.storage.batch_runs.artifact_storage.upload_text",
        lambda payload, object_name, content_type: uploads.append(
            (payload, object_name, content_type)
        ),
    )

    scoring_table = pd.DataFrame([{"booking_id": "A1", "probability_of_cancellation": 0.8}])
    high_risk_table = pd.DataFrame([{"booking_id": "A1", "probability_of_cancellation": 0.8}])

    upload_batch_outputs("2026-04-16", scoring_table, high_risk_table)

    assert len(uploads) == 3
    assert uploads[0][1] == "batch-runs/2026-04-16/booking_risk_scores.csv"
    assert uploads[1][1] == "batch-runs/2026-04-16/high_risk_bookings.csv"
    assert uploads[2][1] == "batch-runs/2026-04-16/_SUCCESS.json"
