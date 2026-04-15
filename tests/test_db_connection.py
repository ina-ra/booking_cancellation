import pytest

from src.infrastructure.db.connection import ensure_database_schema, get_engine, get_session_factory


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
        "src.infrastructure.db.connection.Base",
        type("FakeBase", (), {"metadata": FakeMetadata()})(),
    )

    ensure_database_schema()

    assert calls == ["engine"]
