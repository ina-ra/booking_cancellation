from fastapi.testclient import TestClient

from src.interfaces.main import app

client = TestClient(app)


class ReadyRegistry:
    model_name = "LightGBM"

    def is_ready(self):
        return True


class NotReadyRegistry:
    model_name = "unknown"

    def is_ready(self):
        return False


def test_health_returns_status_ok(monkeypatch):
    monkeypatch.setattr("src.interfaces.api.routes.model_registry", ReadyRegistry())

    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "LightGBM"


def test_predict_returns_503_when_model_not_loaded(monkeypatch, sample_booking_payload):
    monkeypatch.setattr("src.interfaces.api.routes.model_registry", NotReadyRegistry())

    response = client.post("/predict", json=sample_booking_payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded"


def test_predict_returns_prediction(monkeypatch, sample_booking_payload):
    monkeypatch.setattr("src.interfaces.api.routes.model_registry", ReadyRegistry())

    def fake_predict_one_use_case(payload, model_registry):
        return {
            "booking_id": "INN00001",
            "probability_of_cancellation": 0.42,
            "is_high_risk": 0,
            "risk_segment": "regular",
        }

    monkeypatch.setattr(
        "src.interfaces.api.routes.predict_one_use_case",
        fake_predict_one_use_case,
    )

    response = client.post("/predict", json=sample_booking_payload)
    assert response.status_code == 200

    body = response.json()
    assert body["booking_id"] == "INN00001"
    assert body["probability_of_cancellation"] == 0.42
    assert body["is_high_risk"] == 0
    assert body["risk_segment"] == "regular"


def test_predict_batch_returns_predictions(monkeypatch, sample_batch_payload):
    monkeypatch.setattr("src.interfaces.api.routes.model_registry", ReadyRegistry())

    def fake_predict_batch_use_case(payloads, risk_share, model_registry):
        return [
            {
                "booking_id": "INN00099",
                "probability_of_cancellation": 0.91,
                "is_high_risk": 1,
                "risk_segment": "top_30_percent",
            },
            {
                "booking_id": "INN00001",
                "probability_of_cancellation": 0.15,
                "is_high_risk": 0,
                "risk_segment": "regular",
            },
        ]

    monkeypatch.setattr(
        "src.interfaces.api.routes.predict_batch_use_case",
        fake_predict_batch_use_case,
    )

    response = client.post(
        "/predict/batch",
        json={"risk_share": 0.3, "bookings": sample_batch_payload},
    )
    assert response.status_code == 200

    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["booking_id"] == "INN00099"