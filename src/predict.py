import argparse
import json
import os

import lightgbm as lgb
import pandas as pd

from preprocessing import preprocess_booking_data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "new_bookings.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lightgbm_model.txt")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "model_comparison.json")
DEFAULT_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "predictions.csv")

TARGET_COLUMN = "booking status"
ID_COLUMN = "Booking_ID"


def load_metadata():
    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        report = json.load(file)

    categorical_columns = report.get("categorical_columns", [])
    return categorical_columns


def prepare_features(df, categorical_columns):
    processed_df, summary = preprocess_booking_data(df, is_training=False)

    booking_ids = None
    if ID_COLUMN in processed_df.columns:
        booking_ids = processed_df[ID_COLUMN].copy()

    feature_df = processed_df.drop(
        columns=[column for column in [ID_COLUMN, TARGET_COLUMN] if column in processed_df.columns]
    )

    for column in categorical_columns:
        if column in feature_df.columns:
            feature_df[column] = feature_df[column].astype("category")

    return booking_ids, feature_df, summary


def build_predictions_frame(raw_df, booking_ids, probabilities, threshold):
    predictions = pd.DataFrame(
        {
            "cancellation_probability": probabilities.round(4),
            "prediction": (probabilities >= threshold).astype(int),
            "prediction_label": [
                "Canceled" if value >= threshold else "Not_Canceled"
                for value in probabilities
            ],
        }
    )

    raw_without_target = raw_df.drop(columns=[TARGET_COLUMN], errors="ignore").reset_index(drop=True)

    if booking_ids is not None and ID_COLUMN not in raw_without_target.columns:
        predictions.insert(0, ID_COLUMN, booking_ids.values)

    return pd.concat([raw_without_target, predictions], axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LightGBM predictions on new booking data.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to the raw CSV with new bookings.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save predictions CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for class prediction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_df = pd.read_csv(args.input)
    categorical_columns = load_metadata()
    booking_ids, features, summary = prepare_features(raw_df, categorical_columns)

    model = lgb.Booster(model_file=MODEL_PATH)
    probabilities = pd.Series(model.predict(features))
    predictions_df = build_predictions_frame(raw_df, booking_ids, probabilities, args.threshold)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    predictions_df.to_csv(args.output, index=False)

    print("Входной файл:", args.input)
    print("Строк до предобработки:", summary["original_shape"][0])
    print("Строк после предобработки:", summary["processed_shape"][0])
    print("Предсказания сохранены в:", args.output)
    print("\nПервые 5 предсказаний:")
    print(predictions_df.head())


if __name__ == "__main__":
    main()
