from pathlib import Path

import pytest

from src.config import settings
from src.interfaces.cli.predict_cli import build_batch_output_path, parse_run_date


def test_build_batch_output_path_without_run_date_returns_original_path():
    original = Path("artifacts/booking_risk_scores.csv")

    assert build_batch_output_path(original, None) == original


def test_build_batch_output_path_with_run_date_uses_partitioned_directory():
    result = build_batch_output_path(Path("artifacts/booking_risk_scores.csv"), "2026-04-16")
    expected = settings.artifacts_dir / "batch_runs" / "2026-04-16" / "booking_risk_scores.csv"

    assert result == expected


def test_parse_run_date_accepts_iso_date():
    assert parse_run_date("2026-04-16") == "2026-04-16"


def test_parse_run_date_rejects_invalid_format():
    with pytest.raises(ValueError):
        parse_run_date("16-04-2026")
