from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow.providers.docker.operators.docker import DockerOperator

from airflow import DAG


def build_batch_environment() -> dict[str, str]:
    return {
        "POSTGRES_HOST": "postgres",
        "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
        "POSTGRES_DB": os.environ["POSTGRES_DB"],
        "POSTGRES_USER": os.environ["POSTGRES_USER"],
        "POSTGRES_PASSWORD": os.environ["POSTGRES_PASSWORD"],
        "POSTGRES_SSLMODE": "disable",
        "S3_ENDPOINT_URL": "http://minio:9000",
        "S3_BUCKET": os.environ["S3_BUCKET"],
        "S3_ACCESS_KEY": os.environ["S3_ACCESS_KEY"],
        "S3_SECRET_KEY": os.environ["S3_SECRET_KEY"],
        "S3_REGION": os.getenv("S3_REGION", "us-east-1"),
        "S3_ARTIFACTS_PREFIX": os.getenv("S3_ARTIFACTS_PREFIX", "artifacts"),
        "S3_BATCH_OUTPUTS_PREFIX": os.getenv("S3_BATCH_OUTPUTS_PREFIX", "batch-runs"),
        "S3_AUTO_CREATE_BUCKET": os.getenv("S3_AUTO_CREATE_BUCKET", "true"),
        "S3_USE_PATH_STYLE": os.getenv("S3_USE_PATH_STYLE", "true"),
        "DEFAULT_BATCH_RISK_SHARE": os.getenv("DEFAULT_BATCH_RISK_SHARE", "0.3"),
    }


with DAG(
    dag_id="booking_batch_scoring",
    description="Daily batch scoring for booking cancellation risk.",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 4, 14, tz="UTC"),
    catchup=True,
    max_active_runs=1,
    default_args={
        "owner": "ina-ra",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["batch", "docker", "booking"],
) as dag:
    check_batch_not_processed = DockerOperator(
        task_id="check_batch_not_processed",
        image="booking-cancellation-app:latest",
        command=(
            "python -m src.interfaces.cli.check_batch_run_cli "
            "--mode precheck --run-date {{ ds }}"
        ),
        docker_url="unix://var/run/docker.sock",
        api_version="auto",
        auto_remove="success",
        network_mode="booking_cancellation_default",
        mount_tmp_dir=False,
        environment=build_batch_environment(),
    )

    run_batch_scoring = DockerOperator(
        task_id="run_batch_scoring",
        image="booking-cancellation-app:latest",
        command="python -m src.interfaces.cli.predict_cli --run-date {{ ds }}",
        docker_url="unix://var/run/docker.sock",
        api_version="auto",
        auto_remove="success",
        network_mode="booking_cancellation_default",
        mount_tmp_dir=False,
        environment=build_batch_environment(),
    )

    verify_batch_outputs = DockerOperator(
        task_id="verify_batch_outputs",
        image="booking-cancellation-app:latest",
        command=(
            "python -m src.interfaces.cli.check_batch_run_cli "
            "--mode verify-outputs --run-date {{ ds }}"
        ),
        docker_url="unix://var/run/docker.sock",
        api_version="auto",
        auto_remove="success",
        network_mode="booking_cancellation_default",
        mount_tmp_dir=False,
        environment=build_batch_environment(),
    )

    check_batch_not_processed >> run_batch_scoring >> verify_batch_outputs
