import pytest

from src.interfaces.cli.check_batch_run_cli import main


def test_precheck_exits_with_skip_code_when_run_already_exists(monkeypatch):
    monkeypatch.setattr(
        "src.interfaces.cli.check_batch_run_cli.parse_args",
        lambda: type("Args", (), {"mode": "precheck", "run_date": "2026-04-16"})(),
    )
    monkeypatch.setattr(
        "src.interfaces.cli.check_batch_run_cli.batch_run_exists",
        lambda run_date: True,
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 99


def test_precheck_returns_when_run_does_not_exist(monkeypatch):
    monkeypatch.setattr(
        "src.interfaces.cli.check_batch_run_cli.parse_args",
        lambda: type("Args", (), {"mode": "precheck", "run_date": "2026-04-16"})(),
    )
    monkeypatch.setattr(
        "src.interfaces.cli.check_batch_run_cli.batch_run_exists",
        lambda run_date: False,
    )

    assert main() is None


def test_verify_outputs_checks_all_required_objects(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "src.interfaces.cli.check_batch_run_cli.parse_args",
        lambda: type("Args", (), {"mode": "verify-outputs", "run_date": "2026-04-16"})(),
    )
    monkeypatch.setattr(
        "src.interfaces.cli.check_batch_run_cli.artifact_storage.object_exists",
        lambda object_name: seen.append(object_name) or True,
    )

    main()

    assert seen == [
        "batch-runs/2026-04-16/booking_risk_scores.csv",
        "batch-runs/2026-04-16/high_risk_bookings.csv",
        "batch-runs/2026-04-16/_SUCCESS.json",
    ]
