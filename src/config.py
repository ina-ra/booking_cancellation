import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_flag(name: str, default: bool = False) -> bool:
    # Parse boolean-like flags from .env values such as true/false or 1/0.
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    artifacts_dir: Path
    docs_dir: Path
    raw_booking_data_path: Path
    processed_booking_data_path: Path
    new_bookings_data_path: Path
    lightgbm_model_text_path: Path
    lightgbm_model_pickle_path: Path
    model_report_path: Path
    risk_scores_path: Path
    high_risk_bookings_path: Path
    default_high_risk_threshold: float
    default_batch_risk_share: float
    postgres_host: str | None
    postgres_port: int | None
    postgres_db: str | None
    postgres_user: str | None
    postgres_password: str | None
    postgres_sslmode: str | None
    s3_endpoint_url: str | None
    s3_bucket: str | None
    s3_access_key: str | None
    s3_secret_key: str | None
    s3_region: str
    s3_artifacts_prefix: str
    s3_batch_outputs_prefix: str
    s3_auto_create_bucket: bool
    s3_use_path_style: bool
    target_column: str
    id_column: str
    date_column: str
    random_state: int
    test_size: float

    @property
    def postgres_enabled(self) -> bool:
        return all(
            [
                self.postgres_host,
                self.postgres_port,
                self.postgres_db,
                self.postgres_user,
                self.postgres_password,
            ]
        )

    @property
    def postgres_url(self) -> str | None:
        if not self.postgres_enabled:
            return None

        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?sslmode={self.postgres_sslmode or 'require'}"
        )

    @property
    def s3_enabled(self) -> bool:
        return all(
            [
                self.s3_endpoint_url,
                self.s3_bucket,
                self.s3_access_key,
                self.s3_secret_key,
            ]
        )

    @property
    def lightgbm_model_text_object_name(self) -> str:
        return f"{self.s3_artifacts_prefix}/lightgbm_model.txt"

    @property
    def lightgbm_model_pickle_object_name(self) -> str:
        return f"{self.s3_artifacts_prefix}/lightgbm_model.pkl"

    @property
    def model_report_object_name(self) -> str:
        return f"{self.s3_artifacts_prefix}/model_comparison.json"

    @property
    def batch_outputs_prefix(self) -> str:
        return self.s3_batch_outputs_prefix.strip("/")


def build_settings() -> Settings:
    # Resolve the project root once so every other path can be derived from it.
    base_dir = Path(__file__).resolve().parents[1]
    load_dotenv(base_dir / ".env")

    # Centralized local directories used across training, inference, and docs.
    data_dir = base_dir / "data"
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    artifacts_dir = base_dir / "artifacts"
    docs_dir = base_dir / "docs"

    return Settings(
        # Project directories and file locations.
        base_dir=base_dir,
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        artifacts_dir=artifacts_dir,
        docs_dir=docs_dir,
        raw_booking_data_path=raw_data_dir / "booking.csv",
        processed_booking_data_path=processed_data_dir / "booking_clean.csv",
        new_bookings_data_path=raw_data_dir / "new_bookings.csv",
        lightgbm_model_text_path=artifacts_dir / "lightgbm_model.txt",
        lightgbm_model_pickle_path=artifacts_dir / "lightgbm_model.pkl",
        model_report_path=artifacts_dir / "model_comparison.json",
        risk_scores_path=artifacts_dir / "booking_risk_scores.csv",
        high_risk_bookings_path=artifacts_dir / "high_risk_bookings.csv",
        # Default ML thresholds and train/test split parameters.
        default_high_risk_threshold=float(os.getenv("DEFAULT_HIGH_RISK_THRESHOLD", "0.7")),
        default_batch_risk_share=float(os.getenv("DEFAULT_BATCH_RISK_SHARE", "0.3")),
        # Remote Postgres connection settings.
        postgres_host=os.getenv("POSTGRES_HOST"),
        postgres_port=(
            int(os.getenv("POSTGRES_PORT", "5432"))
            if os.getenv("POSTGRES_HOST")
            else None
        ),
        postgres_db=os.getenv("POSTGRES_DB"),
        postgres_user=os.getenv("POSTGRES_USER"),
        postgres_password=os.getenv("POSTGRES_PASSWORD"),
        postgres_sslmode=os.getenv("POSTGRES_SSLMODE", "require"),
        # S3-compatible storage settings used for model artifacts.
        s3_endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        s3_bucket=os.getenv("S3_BUCKET"),
        s3_access_key=os.getenv("S3_ACCESS_KEY"),
        s3_secret_key=os.getenv("S3_SECRET_KEY"),
        s3_region=os.getenv("S3_REGION", "us-east-1"),
        s3_artifacts_prefix=os.getenv("S3_ARTIFACTS_PREFIX", "artifacts").strip("/"),
        s3_batch_outputs_prefix=os.getenv("S3_BATCH_OUTPUTS_PREFIX", "batch-runs").strip("/"),
        s3_auto_create_bucket=_env_flag("S3_AUTO_CREATE_BUCKET", default=True),
        s3_use_path_style=_env_flag("S3_USE_PATH_STYLE", default=True),
        # Dataset-specific column names and reproducibility parameters.
        target_column="booking status",
        id_column="Booking_ID",
        date_column="date of reservation",
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
    )


settings = build_settings()

