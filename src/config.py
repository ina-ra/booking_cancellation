import os
from dataclasses import dataclass
from pathlib import Path


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
    target_column: str
    id_column: str
    date_column: str
    random_state: int
    test_size: float


def build_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    artifacts_dir = base_dir / "artifacts"
    docs_dir = base_dir / "docs"

    return Settings(
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
        default_high_risk_threshold=float(os.getenv("DEFAULT_HIGH_RISK_THRESHOLD", "0.7")),
        default_batch_risk_share=float(os.getenv("DEFAULT_BATCH_RISK_SHARE", "0.3")),
        target_column="booking status",
        id_column="Booking_ID",
        date_column="date of reservation",
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
    )


settings = build_settings()

