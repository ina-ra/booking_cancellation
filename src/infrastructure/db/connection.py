from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from src.config import settings
from src.infrastructure.db.models import Base


def get_engine():
    if not settings.postgres_url:
        raise ValueError("Postgres is not configured. Set POSTGRES_* environment variables.")

    return create_engine(settings.postgres_url, pool_pre_ping=True)


def get_session_factory():
    engine = get_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def ensure_database_schema():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_idempotency_columns(engine)


def _ensure_idempotency_columns(engine) -> None:
    inspector = inspect(engine)

    prediction_columns = {column["name"] for column in inspector.get_columns("predictions")}
    monitoring_columns = {column["name"] for column in inspector.get_columns("monitoring_metrics")}
    prediction_constraints = {
        constraint["name"] for constraint in inspector.get_unique_constraints("predictions")
    }
    monitoring_constraints = {
        constraint["name"] for constraint in inspector.get_unique_constraints("monitoring_metrics")
    }

    with engine.begin() as connection:
        if "run_date" not in prediction_columns:
            connection.execute(
                text(
                    "ALTER TABLE predictions "
                    "ADD COLUMN run_date VARCHAR(20) NOT NULL DEFAULT 'adhoc'"
                )
            )
        if "uq_predictions_booking_model_run_date" not in prediction_constraints:
            _deduplicate_predictions(connection)
            connection.execute(
                text(
                    "ALTER TABLE predictions "
                    "ADD CONSTRAINT uq_predictions_booking_model_run_date "
                    "UNIQUE (booking_id, model_name, run_date)"
                )
            )

        if "run_date" not in monitoring_columns:
            connection.execute(
                text(
                    "ALTER TABLE monitoring_metrics "
                    "ADD COLUMN run_date VARCHAR(20) NOT NULL DEFAULT 'adhoc'"
                )
            )
        if "uq_monitoring_run_metric_model_run_date" not in monitoring_constraints:
            _deduplicate_monitoring_metrics(connection)
            connection.execute(
                text(
                    "ALTER TABLE monitoring_metrics "
                    "ADD CONSTRAINT uq_monitoring_run_metric_model_run_date "
                    "UNIQUE (run_type, metric_name, model_name, run_date)"
                )
            )


def _deduplicate_predictions(connection) -> None:
    connection.execute(
        text(
            "DELETE FROM predictions "
            "WHERE id IN ("
            "    SELECT id FROM ("
            "        SELECT id, "
            "               ROW_NUMBER() OVER ("
            "                   PARTITION BY booking_id, model_name, run_date "
            "                   ORDER BY created_at DESC, id DESC"
            "               ) AS row_num "
            "        FROM predictions"
            "    ) duplicates "
            "    WHERE duplicates.row_num > 1"
            ")"
        )
    )


def _deduplicate_monitoring_metrics(connection) -> None:
    connection.execute(
        text(
            "DELETE FROM monitoring_metrics "
            "WHERE id IN ("
            "    SELECT id FROM ("
            "        SELECT id, "
            "               ROW_NUMBER() OVER ("
            "                   PARTITION BY run_type, metric_name, model_name, run_date "
            "                   ORDER BY created_at DESC, id DESC"
            "               ) AS row_num "
            "        FROM monitoring_metrics"
            "    ) duplicates "
            "    WHERE duplicates.row_num > 1"
            ")"
        )
    )
