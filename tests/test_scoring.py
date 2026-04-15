import numpy as np
import pandas as pd

from src.application.scoring import (
    build_batch_predictions,
    build_scoring_table,
    build_single_prediction,
)


class FakeModel:
    def __init__(self, probabilities):
        self._probabilities = probabilities

    def predict_proba(self, features):
        return np.array([[1 - p, p] for p in self._probabilities])


def test_build_scoring_table_sorts_and_marks_high_risk(sample_probabilities):
    booking_ids = pd.Series(["A", "B", "C"])
    result = build_scoring_table(booking_ids, sample_probabilities, risk_share=0.34)

    assert list(result["booking_id"]) == ["B", "C", "A"]
    assert list(result["is_high_risk"]) == [1, 1, 0]
    assert list(result["risk_segment"]) == ["top_30_percent", "top_30_percent", "regular"]


def test_build_scoring_table_creates_ids_if_missing(sample_probabilities):
    result = build_scoring_table(None, sample_probabilities, risk_share=0.2)

    assert "booking_id" in result.columns
    assert len(result) == 3
    assert result.iloc[0]["booking_id"] == 2


def test_build_single_prediction_returns_expected_structure(monkeypatch, sample_booking_payload):
    fake_features = pd.DataFrame([{"lead time": 45, "average price": 120.5}])

    def fake_prepare_features(df, categorical_columns):
        booking_ids = pd.Series(["INN00001"])
        summary = {"processed_shape": (1, 2)}
        return booking_ids, fake_features, summary

    monkeypatch.setattr("src.application.scoring.prepare_features", fake_prepare_features)

    model = FakeModel([0.85])
    result = build_single_prediction(sample_booking_payload, model, categorical_columns=[])

    assert result["booking_id"] == "INN00001"
    assert result["probability_of_cancellation"] == 0.85
    assert result["is_high_risk"] == 1
    assert result["risk_segment"] == "high_risk"


def test_build_single_prediction_uses_booking_entity(monkeypatch, sample_booking_payload):
    fake_features = pd.DataFrame([{"lead time": 45, "average price": 120.5}])

    def fake_prepare_features(df, categorical_columns):
        assert df.iloc[0]["Booking_ID"] == "INN00001"
        assert df.iloc[0]["lead time"] == 45
        return pd.Series(["INN00001"]), fake_features, {"processed_shape": (1, 2)}

    monkeypatch.setattr("src.application.scoring.prepare_features", fake_prepare_features)

    result = build_single_prediction(
        sample_booking_payload,
        FakeModel([0.25]),
        categorical_columns=[],
    )

    assert result["booking_id"] == "INN00001"
    assert result["probability_of_cancellation"] == 0.25


def test_build_batch_predictions_returns_records(monkeypatch, sample_batch_payload):
    fake_features = pd.DataFrame(
        [
            {"lead time": 45, "average price": 120.5},
            {"lead time": 180, "average price": 220.0},
        ]
    )

    def fake_prepare_features(df, categorical_columns):
        booking_ids = pd.Series(["INN00001", "INN00099"])
        summary = {"processed_shape": (2, 2)}
        return booking_ids, fake_features, summary

    monkeypatch.setattr("src.application.scoring.prepare_features", fake_prepare_features)

    model = FakeModel([0.2, 0.9])
    result = build_batch_predictions(
        sample_batch_payload,
        model,
        categorical_columns=[],
        risk_share=0.5,
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["booking_id"] == "INN00099"
    assert result[0]["is_high_risk"] == 1
    assert result[1]["booking_id"] == "INN00001"
    assert result[0]["rank"] == 1
    assert result[0]["risk_percentile"] == 50.0
