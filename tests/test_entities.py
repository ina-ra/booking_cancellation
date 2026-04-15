from src.domain.entities import BatchScoringResult, Booking, BookingRiskScore, TrainingResult


def test_booking_from_payload_and_to_payload(sample_booking_payload):
    booking = Booking.from_payload(sample_booking_payload)

    assert booking.booking_id == "INN00001"
    assert booking.lead_time == 45
    assert booking.to_payload() == sample_booking_payload


def test_batch_scoring_result_counts_high_risk_records():
    result = BatchScoringResult(
        scores=[
            BookingRiskScore(
                "A",
                0.9,
                rank=1,
                risk_percentile=50.0,
                is_high_risk=1,
                risk_segment="top",
            ),
            BookingRiskScore(
                "B",
                0.1,
                rank=2,
                risk_percentile=100.0,
                is_high_risk=0,
                risk_segment="regular",
            ),
        ]
    )

    assert result.total_bookings == 2
    assert result.high_risk_count == 1
    assert result.to_dict_list()[0]["booking_id"] == "A"


def test_training_result_stores_training_outputs():
    result = TrainingResult(
        model_name="LightGBM",
        metrics={"accuracy": 0.9},
        parameters={"n_estimators": 700},
        report={"best_model": {"name": "LightGBM"}},
    )

    assert result.model_name == "LightGBM"
    assert result.metrics["accuracy"] == 0.9
    assert result.parameters["n_estimators"] == 700
