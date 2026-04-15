import pandas as pd
import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.infrastructure.db.models import ModelRun, MonitoringMetric, PredictionRecord
from src.infrastructure.db.repositories import (
    save_model_run,
    save_monitoring_metrics,
    save_prediction_batch,
)


class FakeSession:
    def __init__(self, fail_on_add=False):
        self.fail_on_add = fail_on_add
        self.added = []
        self.committed = False
        self.rolled_back = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add(self, item):
        if self.fail_on_add:
            raise SQLAlchemyError("db error")
        self.added.append(item)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def make_session_factory(session):
    return lambda: session


def test_save_model_run_persists_model_run(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(
        "src.infrastructure.db.repositories.get_session_factory",
        lambda: make_session_factory(session),
    )

    save_model_run(
        model_name="LightGBM",
        model_version="v1",
        metrics={"accuracy": 0.91, "f1": 0.82, "roc_auc": 0.88},
        parameters={"n_estimators": 700},
    )

    assert session.committed is True
    assert len(session.added) == 1
    saved_item = session.added[0]
    assert isinstance(saved_item, ModelRun)
    assert saved_item.model_name == "LightGBM"
    assert saved_item.model_version == "v1"
    assert saved_item.parameters_json == '{"n_estimators": 700}'


def test_save_model_run_rolls_back_on_error(monkeypatch):
    session = FakeSession(fail_on_add=True)
    monkeypatch.setattr(
        "src.infrastructure.db.repositories.get_session_factory",
        lambda: make_session_factory(session),
    )

    with pytest.raises(SQLAlchemyError):
        save_model_run(
            model_name="LightGBM",
            model_version="v1",
            metrics={"accuracy": 0.91, "f1": 0.82, "roc_auc": 0.88},
            parameters={"n_estimators": 700},
        )

    assert session.rolled_back is True


def test_save_prediction_batch_persists_prediction_records(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(
        "src.infrastructure.db.repositories.get_session_factory",
        lambda: make_session_factory(session),
    )
    scores = pd.DataFrame(
        [
            {
                "booking_id": "INN001",
                "probability_of_cancellation": 0.9,
                "rank": 1,
                "risk_percentile": 50.0,
                "is_high_risk": 1,
                "risk_segment": "top_30_percent",
            },
            {
                "booking_id": "INN002",
                "probability_of_cancellation": 0.1,
                "rank": 2,
                "risk_percentile": 100.0,
                "is_high_risk": 0,
                "risk_segment": "regular",
            },
        ]
    )

    save_prediction_batch(scores, model_name="LightGBM")

    assert session.committed is True
    assert len(session.added) == 2
    assert all(isinstance(item, PredictionRecord) for item in session.added)
    assert session.added[0].booking_id == "INN001"
    assert session.added[0].is_high_risk is True


def test_save_monitoring_metrics_persists_monitoring_rows(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(
        "src.infrastructure.db.repositories.get_session_factory",
        lambda: make_session_factory(session),
    )

    save_monitoring_metrics(
        run_type="train",
        model_name="LightGBM",
        metrics={"records_before_preprocessing": 120.0, "eval_accuracy": 0.91},
    )

    assert session.committed is True
    assert len(session.added) == 2
    assert all(isinstance(item, MonitoringMetric) for item in session.added)
    assert session.added[0].run_type == "train"
    assert session.added[1].metric_name == "eval_accuracy"


def test_save_monitoring_metrics_rolls_back_on_error(monkeypatch):
    session = FakeSession(fail_on_add=True)
    monkeypatch.setattr(
        "src.infrastructure.db.repositories.get_session_factory",
        lambda: make_session_factory(session),
    )

    with pytest.raises(SQLAlchemyError):
        save_monitoring_metrics(
            run_type="train",
            model_name="LightGBM",
            metrics={"eval_accuracy": 0.91},
        )

    assert session.rolled_back is True
