import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from src.application.monitoring import build_batch_monitoring_metrics
from src.application.scoring import build_scoring_table, prepare_features
from src.config import settings
from src.infrastructure.db.repositories import save_monitoring_metrics, save_prediction_batch
from src.infrastructure.ml.artifacts import load_model_report, load_pickled_model
from src.infrastructure.storage import upload_batch_outputs


def load_metadata():
    report = load_model_report()
    return report.get("categorical_columns", [])


def parse_run_date(value: str) -> str:
    datetime.strptime(value, "%Y-%m-%d")
    return value


def build_batch_output_path(default_path: Path, run_date: str | None) -> Path:
    if not run_date:
        return default_path

    return settings.artifacts_dir / "batch_runs" / run_date / default_path.name


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch LightGBM scoring for new bookings.")
    parser.add_argument("--input", default=str(settings.new_bookings_data_path))
    parser.add_argument("--scores-output", default=str(settings.risk_scores_path))
    parser.add_argument("--high-risk-output", default=str(settings.high_risk_bookings_path))
    parser.add_argument("--risk-share", type=float, default=settings.default_batch_risk_share)
    parser.add_argument(
        "--run-date",
        type=parse_run_date,
        help=(
            "Logical run date in YYYY-MM-DD format. "
            "When set, outputs are written to artifacts/batch_runs/<run-date>/."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deprecated: reruns already overwrite existing outputs for the same run date.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    scores_output = build_batch_output_path(Path(args.scores_output), args.run_date)
    high_risk_output = build_batch_output_path(Path(args.high_risk_output), args.run_date)

    raw_df = pd.read_csv(args.input)
    categorical_columns = load_metadata()
    booking_ids, features, summary = prepare_features(raw_df, categorical_columns)
    model = load_pickled_model()

    probabilities = pd.Series(model.predict_proba(features)[:, 1])
    scoring_table = build_scoring_table(booking_ids, probabilities, args.risk_share)
    high_risk_table = scoring_table[scoring_table["is_high_risk"] == 1].reset_index(drop=True)

    scores_output.parent.mkdir(parents=True, exist_ok=True)
    high_risk_output.parent.mkdir(parents=True, exist_ok=True)
    scoring_table.to_csv(scores_output, index=False)
    high_risk_table.to_csv(high_risk_output, index=False)

    if settings.postgres_enabled:
        try:
            save_prediction_batch(
                scoring_table,
                model_name="LightGBM",
                run_date=args.run_date,
            )
            save_monitoring_metrics(
                run_type="batch_predict",
                model_name="LightGBM",
                metrics=build_batch_monitoring_metrics(
                    preprocessing_summary=summary,
                    scoring_table=scoring_table,
                ),
                run_date=args.run_date,
            )
            print("Batch scoring results were saved to Postgres.")
        except SQLAlchemyError as error:
            print(f"Failed to save predictions to Postgres: {error}")
            raise

    if args.run_date:
        upload_batch_outputs(args.run_date, scoring_table, high_risk_table)
        print("Batch outputs and success marker were uploaded to S3.")

    print("Input file:", args.input)
    print("Rows before preprocessing:", summary["original_shape"][0])
    print("Rows after preprocessing:", summary["processed_shape"][0])
    if args.run_date:
        print("Logical run date:", args.run_date)
    print("Full scoring output saved to:", scores_output)
    print("High-risk segment saved to:", high_risk_output)
    print("\nFirst 5 scoring rows:")
    print(scoring_table.head())


if __name__ == "__main__":
    main()
