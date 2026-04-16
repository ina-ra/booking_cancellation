import pytest

from src.infrastructure.db.connection import (
    _deduplicate_monitoring_metrics,
    _deduplicate_predictions,
    ensure_database_schema,
    get_engine,
    get_session_factory,
)


def test_get_engine_raises_when_postgres_is_not_configured(monkeypatch):
    monkeypatch.setattr(
        "src.infrastructure.db.connection.settings",
        type("FakeSettings", (), {"postgres_url": None})(),
    )

    with pytest.raises(ValueError, match="Postgres is not configured"):
        get_engine()


def test_get_engine_returns_sqlalchemy_engine(monkeypatch):
    created = {}

    def fake_create_engine(url, pool_pre_ping):
        created["url"] = url
        created["pool_pre_ping"] = pool_pre_ping
        return "engine"

    monkeypatch.setattr(
        "src.infrastructure.db.connection.settings",
        type("FakeSettings", (), {"postgres_url": "postgresql://example"})(),
    )
    monkeypatch.setattr("src.infrastructure.db.connection.create_engine", fake_create_engine)

    result = get_engine()

    assert result == "engine"
    assert created == {"url": "postgresql://example", "pool_pre_ping": True}


def test_get_session_factory_uses_engine(monkeypatch):
    monkeypatch.setattr("src.infrastructure.db.connection.get_engine", lambda: "engine")
    monkeypatch.setattr(
        "src.infrastructure.db.connection.sessionmaker",
        lambda bind, autoflush, autocommit: {
            "bind": bind,
            "autoflush": autoflush,
            "autocommit": autocommit,
        },
    )

    result = get_session_factory()

    assert result == {
        "bind": "engine",
        "autoflush": False,
        "autocommit": False,
    }


def test_ensure_database_schema_calls_create_all(monkeypatch):
    calls = []

    class FakeMetadata:
        def create_all(self, bind):
            calls.append(bind)

    monkeypatch.setattr("src.infrastructure.db.connection.get_engine", lambda: "engine")
    monkeypatch.setattr(
        "src.infrastructure.db.connection._ensure_idempotency_columns",
        lambda engine: None,
    )
    monkeypatch.setattr(
        "src.infrastructure.db.connection.Base",
        type("FakeBase", (), {"metadata": FakeMetadata()})(),
    )

    ensure_database_schema()

    assert calls == ["engine"]


class FakeConnection:
    def __init__(self):
        self.statements = []

    def execute(self, statement):
        self.statements.append(str(statement))


def test_deduplicate_predictions_executes_delete_query():
    connection = FakeConnection()

    _deduplicate_predictions(connection)

    assert "DELETE FROM predictions" in connection.statements[0]
    assert "PARTITION BY booking_id, model_name, run_date" in connection.statements[0]


def test_deduplicate_monitoring_metrics_executes_delete_query():
    connection = FakeConnection()

    _deduplicate_monitoring_metrics(connection)

    assert "DELETE FROM monitoring_metrics" in connection.statements[0]
    assert "PARTITION BY run_type, metric_name, model_name, run_date" in connection.statements[0]
