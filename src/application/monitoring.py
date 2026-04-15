import pandas as pd


def build_training_monitoring_metrics(
    preprocessing_summary: dict,
    evaluation_metrics: dict[str, float],
) -> dict[str, float]:
    metrics = {
        "records_before_preprocessing": float(preprocessing_summary["original_shape"][0]),
        "records_after_preprocessing": float(preprocessing_summary["processed_shape"][0]),
        "duplicates_count": float(preprocessing_summary["duplicates_count"]),
        "invalid_dates_count": float(preprocessing_summary["invalid_dates_count"]),
    }
    metrics.update({f"eval_{name}": float(value) for name, value in evaluation_metrics.items()})
    return metrics


def build_batch_monitoring_metrics(
    preprocessing_summary: dict,
    scoring_table: pd.DataFrame,
) -> dict[str, float]:
    return {
        "records_before_preprocessing": float(preprocessing_summary["original_shape"][0]),
        "records_after_preprocessing": float(preprocessing_summary["processed_shape"][0]),
        "predictions_count": float(len(scoring_table)),
        "high_risk_count": float(int(scoring_table["is_high_risk"].sum())),
        "high_risk_share": float(scoring_table["is_high_risk"].mean()),
        "mean_cancellation_probability": float(scoring_table["probability_of_cancellation"].mean()),
        "max_cancellation_probability": float(scoring_table["probability_of_cancellation"].max()),
        "min_cancellation_probability": float(scoring_table["probability_of_cancellation"].min()),
        "avg_risk_percentile": float(scoring_table["risk_percentile"].mean()),
        "invalid_dates_count": float(preprocessing_summary["invalid_dates_count"]),
    }
