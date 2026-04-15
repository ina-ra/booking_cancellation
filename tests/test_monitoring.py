import pandas as pd

from src.application.monitoring import (
    build_batch_monitoring_metrics,
    build_training_monitoring_metrics,
)


def test_build_training_monitoring_metrics_returns_expected_values():
    preprocessing_summary = {
        "original_shape": (120, 10),
        "processed_shape": (100, 12),
        "duplicates_count": 5,
        "invalid_dates_count": 2,
    }
    evaluation_metrics = {
        "accuracy": 0.9,
        "roc_auc": 0.87,
    }

    result = build_training_monitoring_metrics(preprocessing_summary, evaluation_metrics)

    assert result == {
        "records_before_preprocessing": 120.0,
        "records_after_preprocessing": 100.0,
        "duplicates_count": 5.0,
        "invalid_dates_count": 2.0,
        "eval_accuracy": 0.9,
        "eval_roc_auc": 0.87,
    }


def test_build_batch_monitoring_metrics_returns_expected_values():
    preprocessing_summary = {
        "original_shape": (4, 10),
        "processed_shape": (3, 12),
        "invalid_dates_count": 1,
    }
    scoring_table = pd.DataFrame(
        {
            "is_high_risk": [1, 0, 1],
            "probability_of_cancellation": [0.91, 0.35, 0.72],
            "risk_percentile": [33.33, 66.67, 100.0],
        }
    )

    result = build_batch_monitoring_metrics(preprocessing_summary, scoring_table)

    assert result["records_before_preprocessing"] == 4.0
    assert result["records_after_preprocessing"] == 3.0
    assert result["predictions_count"] == 3.0
    assert result["high_risk_count"] == 2.0
    assert round(result["high_risk_share"], 4) == 0.6667
    assert result["mean_cancellation_probability"] == float(
        scoring_table["probability_of_cancellation"].mean()
    )
    assert result["max_cancellation_probability"] == 0.91
    assert result["min_cancellation_probability"] == 0.35
    assert result["avg_risk_percentile"] == float(scoring_table["risk_percentile"].mean())
    assert result["invalid_dates_count"] == 1.0
