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


def test_frontend_health_returns_status_ok(monkeypatch):
    monkeypatch.setattr("src.interfaces.api.routes.model_registry", ReadyRegistry())

    response = client.get("/frontend-api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["modelLoaded"] is True
    assert body["modelName"] == "LightGBM"


def test_frontend_predict_returns_mapped_prediction(monkeypatch):
    monkeypatch.setattr("src.interfaces.api.routes.model_registry", ReadyRegistry())

    def fake_predict_one_use_case(payload, model_registry):
        assert payload["Booking_ID"] == "INN02501"
        assert payload["number of adults"] == 1
        assert payload["market segment type"] == "Online"
        return {
            "booking_id": "INN02501",
            "probability_of_cancellation": 0.42,
            "is_high_risk": 0,
            "risk_segment": "regular",
        }

    monkeypatch.setattr(
        "src.interfaces.api.routes.predict_one_use_case",
        fake_predict_one_use_case,
    )

    response = client.post(
        "/frontend-api/predict",
        json={
            "bookingId": "INN02501",
            "reservationDate": "2018-04-08",
            "adults": 1,
            "children": 0,
            "weekendNights": 0,
            "weekNights": 1,
            "meal": "Meal Plan 1",
            "parking": "0",
            "roomType": "Room_Type 1",
            "leadTime": 4,
            "marketSegment": "Online",
            "repeated": "0",
            "previousCanceled": 0,
            "previousNotCanceled": 0,
            "averagePrice": 95,
            "specialRequests": 1,
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert body["bookingId"] == "INN02501"
    assert body["probabilityOfCancellation"] == 0.42
    assert body["risk"] == 42
    assert body["isHighRisk"] is False
    assert body["riskSegment"] == "regular"


def test_frontend_predict_batch_returns_summary_and_details(monkeypatch):
    monkeypatch.setattr("src.interfaces.api.routes.model_registry", ReadyRegistry())

    def fake_predict_batch_use_case(payloads, risk_share, model_registry):
        assert risk_share == 0.3
        assert payloads[0]["Booking_ID"] == "ROW-0001"
        assert payloads[1]["Booking_ID"] == "INN04192"
        return [
            {
                "booking_id": "INN04192",
                "probability_of_cancellation": 0.67,
                "is_high_risk": 1,
                "risk_segment": "top_30_percent",
            },
            {
                "booking_id": "ROW-0001",
                "probability_of_cancellation": 0.21,
                "is_high_risk": 0,
                "risk_segment": "regular",
            },
        ]

    monkeypatch.setattr(
        "src.interfaces.api.routes.predict_batch_use_case",
        fake_predict_batch_use_case,
    )

    response = client.post(
        "/frontend-api/predict/batch",
        json={
            "riskShare": 0.3,
            "bookings": [
                {
                    "bookingId": "",
                    "reservationDate": "2018-04-08",
                    "adults": 1,
                    "children": 0,
                    "weekendNights": 0,
                    "weekNights": 1,
                    "meal": "Meal Plan 1",
                    "parking": "0",
                    "roomType": "Room_Type 1",
                    "leadTime": 4,
                    "marketSegment": "Online",
                    "repeated": "0",
                    "previousCanceled": 0,
                    "previousNotCanceled": 0,
                    "averagePrice": 95,
                    "specialRequests": 1,
                },
                {
                    "bookingId": "INN04192",
                    "reservationDate": "2018-09-14",
                    "adults": 2,
                    "children": 0,
                    "weekendNights": 1,
                    "weekNights": 2,
                    "meal": "Meal Plan 1",
                    "parking": "0",
                    "roomType": "Room_Type 4",
                    "leadTime": 118,
                    "marketSegment": "Online",
                    "repeated": "0",
                    "previousCanceled": 0,
                    "previousNotCanceled": 0,
                    "averagePrice": 142,
                    "specialRequests": 0,
                },
            ],
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert body["summary"] == {
        "total": 2,
        "highRiskCount": 1,
        "averageProbability": 44,
    }
    assert body["predictions"][0]["bookingId"] == "INN04192"
    assert body["predictions"][0]["risk"] == 67
    assert body["predictions"][1]["bookingId"] == "ROW-0001"
    assert body["predictions"][1]["riskSegment"] == "regular"
