from src.interfaces.main import startup_event


def test_startup_event_loads_model_registry(monkeypatch):
    calls = []
    fake_registry = type("FakeRegistry", (), {"load": lambda self: calls.append("loaded")})()

    monkeypatch.setattr("src.interfaces.main.model_registry", fake_registry)

    startup_event()

    assert calls == ["loaded"]
