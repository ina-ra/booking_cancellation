import argparse
import json
import pickle
from sqlalchemy.exc import SQLAlchemyError

import pandas as pd

from src.application.monitoring import build_batch_monitoring_metrics
from src.application.scoring import build_scoring_table, prepare_features
from src.config import settings
from src.infrastructure.db.repositories import save_monitoring_metrics, save_prediction_batch


def load_metadata():
    with open(settings.model_report_path, "r", encoding="utf-8") as file:
        report = json.load(file)
    return report.get("categorical_columns", [])


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch LightGBM scoring for new bookings.")
    parser.add_argument("--input", default=str(settings.new_bookings_data_path))
    parser.add_argument("--scores-output", default=str(settings.risk_scores_path))
    parser.add_argument("--high-risk-output", default=str(settings.high_risk_bookings_path))
    parser.add_argument("--risk-share", type=float, default=settings.default_batch_risk_share)
    return parser.parse_args()


def main():
    args = parse_args()
    raw_df = pd.read_csv(args.input)
    categorical_columns = load_metadata()
    booking_ids, features, summary = prepare_features(raw_df, categorical_columns)

    with open(settings.lightgbm_model_pickle_path, "rb") as file:
        model = pickle.load(file)

    probabilities = pd.Series(model.predict_proba(features)[:, 1])
    scoring_table = build_scoring_table(booking_ids, probabilities, args.risk_share)
    high_risk_table = scoring_table[scoring_table["is_high_risk"] == 1].reset_index(drop=True)

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    scoring_table.to_csv(args.scores_output, index=False)
    high_risk_table.to_csv(args.high_risk_output, index=False)

    if settings.postgres_enabled:
        try:
            save_prediction_batch(scoring_table, model_name="LightGBM")
            save_monitoring_metrics(
                run_type="batch_predict",
                model_name="LightGBM",
                metrics=build_batch_monitoring_metrics(
                    preprocessing_summary=summary,
                    scoring_table=scoring_table,
                ),
            )
            print("Результаты batch scoring сохранены в Postgres.")
        except SQLAlchemyError as error:
            print(f"Не удалось сохранить predictions в Postgres: {error}")

    print("Входной файл:", args.input)
    print("Строк до предобработки:", summary["original_shape"][0])
    print("Строк после предобработки:", summary["processed_shape"][0])
    print("Полный скоринг сохранён в:", args.scores_output)
    print("Top-30% риск-сегмент сохранён в:", args.high_risk_output)
    print("\nПервые 5 строк скоринга:")
    print(scoring_table.head())


if __name__ == "__main__":
    main()
