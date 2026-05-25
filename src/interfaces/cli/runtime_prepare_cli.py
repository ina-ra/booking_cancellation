import argparse
import time

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import SQLAlchemyError

from src.application.training import train_lightgbm_pipeline
from src.config import settings
from src.infrastructure.db.connection import ensure_database_schema, get_engine
from src.infrastructure.ml.artifacts import training_artifacts_exist
from src.infrastructure.storage import artifact_storage


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[runtime-prepare] {timestamp} | {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare runtime dependencies such as Postgres schema and model artifacts."
    )
    parser.add_argument(
        "--step",
        choices=["init-db", "seed-model"],
        required=True,
        help="Preparation step to execute.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=30,
        help="How many times dependency checks should be retried before failing.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=int,
        default=2,
        help="How long to wait between dependency check attempts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force model training even if artifacts are already present in S3.",
    )
    return parser.parse_args()


def wait_for_postgres(max_attempts: int, delay_seconds: int) -> None:
    if not settings.postgres_enabled:
        raise ValueError("Postgres is not configured. Set POSTGRES_* environment variables.")

    _log(
        "Waiting for Postgres "
        f"at {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}."
    )
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            engine = get_engine()
            with engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            _log(f"Postgres is reachable on attempt {attempt}.")
            return
        except SQLAlchemyError as error:
            last_error = error
            _log(
                f"Waiting for Postgres ({attempt}/{max_attempts})... "
                f"Last error: {type(error).__name__}."
            )
            time.sleep(delay_seconds)

    raise RuntimeError("Postgres did not become reachable in time.") from last_error


def wait_for_s3(max_attempts: int, delay_seconds: int) -> None:
    if not artifact_storage.is_enabled():
        raise ValueError("S3 storage is not configured. Set S3_* environment variables.")

    _log(
        "Waiting for S3 storage "
        f"at {settings.s3_endpoint_url} bucket={settings.s3_bucket}."
    )
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            artifact_storage.ensure_bucket_exists()
            _log(f"S3 storage is reachable on attempt {attempt}.")
            return
        except (ClientError, BotoCoreError, RuntimeError) as error:
            last_error = error
            _log(
                f"Waiting for S3 storage ({attempt}/{max_attempts})... "
                f"Last error: {type(error).__name__}."
            )
            time.sleep(delay_seconds)

    raise RuntimeError("S3 storage did not become reachable in time.") from last_error


def run_init_db(max_attempts: int, delay_seconds: int) -> None:
    _log("Starting init-db step.")
    wait_for_postgres(max_attempts=max_attempts, delay_seconds=delay_seconds)
    ensure_database_schema()
    _log("Postgres schema initialized successfully.")


def run_seed_model(max_attempts: int, delay_seconds: int, force: bool) -> None:
    _log(
        "Starting seed-model step "
        f"(force={force}, artifacts_prefix={settings.s3_artifacts_prefix})."
    )
    wait_for_postgres(max_attempts=max_attempts, delay_seconds=delay_seconds)
    wait_for_s3(max_attempts=max_attempts, delay_seconds=delay_seconds)

    if not force and training_artifacts_exist():
        _log("Model artifacts already exist in S3. Skipping training.")
        return

    _log("Training pipeline is starting because artifacts are missing or force=true.")
    train_lightgbm_pipeline()
    _log("Model artifacts are ready.")


def main():
    args = parse_args()

    if args.step == "init-db":
        run_init_db(max_attempts=args.max_attempts, delay_seconds=args.delay_seconds)
        return

    if args.step == "seed-model":
        run_seed_model(
            max_attempts=args.max_attempts,
            delay_seconds=args.delay_seconds,
            force=args.force,
        )
        return

    raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
