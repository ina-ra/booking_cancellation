import json

from sqlalchemy.exc import SQLAlchemyError

from src.infrastructure.db.connection import get_session_factory
from src.infrastructure.db.models import ModelRun, MonitoringMetric, PredictionRecord


def save_model_run(model_name: str, model_version: str, metrics: dict, parameters: dict):
    session_factory = get_session_factory()

    with session_factory() as session:
        try:
            session.add(
                ModelRun(
                    model_name=model_name,
                    model_version=model_version,
                    accuracy=metrics["accuracy"],
                    f1=metrics["f1"],
                    roc_auc=metrics["roc_auc"],
                    parameters_json=json.dumps(parameters, ensure_ascii=False),
                )
            )
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise


def save_prediction_batch(scores, model_name: str = "LightGBM", run_date: str | None = None):
    session_factory = get_session_factory()
    effective_run_date = run_date or "adhoc"

    with session_factory() as session:
        try:
            if hasattr(session, "query"):
                session.query(PredictionRecord).filter_by(
                    model_name=model_name,
                    run_date=effective_run_date,
                ).delete()
            for item in scores.to_dict(orient="records"):
                session.add(
                    PredictionRecord(
                        booking_id=str(item["booking_id"]),
                        probability_of_cancellation=float(item["probability_of_cancellation"]),
                        rank=int(item["rank"]),
                        risk_percentile=float(item["risk_percentile"]),
                        is_high_risk=bool(item["is_high_risk"]),
                        risk_segment=str(item["risk_segment"]),
                        model_name=model_name,
                        run_date=effective_run_date,
                    )
                )
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise


def save_monitoring_metrics(
    run_type: str,
    model_name: str,
    metrics: dict[str, float],
    run_date: str | None = None,
):
    session_factory = get_session_factory()
    effective_run_date = run_date or "adhoc"

    with session_factory() as session:
        try:
            if hasattr(session, "query"):
                session.query(MonitoringMetric).filter_by(
                    run_type=run_type,
                    model_name=model_name,
                    run_date=effective_run_date,
                ).delete()
            for metric_name, metric_value in metrics.items():
                session.add(
                    MonitoringMetric(
                        run_type=run_type,
                        metric_name=metric_name,
                        metric_value=float(metric_value),
                        model_name=model_name,
                        run_date=effective_run_date,
                    )
                )
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise
