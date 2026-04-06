import argparse
import json
import math
import os
import pickle

import pandas as pd

from preprocessing import preprocess_booking_data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "new_bookings.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lightgbm_model.pkl")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "model_comparison.json")
DEFAULT_SCORES_PATH = os.path.join(ARTIFACTS_DIR, "booking_risk_scores.csv")
DEFAULT_HIGH_RISK_PATH = os.path.join(ARTIFACTS_DIR, "high_risk_bookings.csv")

TARGET_COLUMN = "booking status"
ID_COLUMN = "Booking_ID"


def load_metadata():
    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        report = json.load(file)

    return report.get("categorical_columns", [])


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


def build_scoring_table(booking_ids, probabilities, risk_share):
    if booking_ids is None:
        booking_ids = pd.Series(range(1, len(probabilities) + 1), name=ID_COLUMN)

    scores = pd.DataFrame(
        {
            "booking_id": booking_ids.values,
            "probability_of_cancellation": probabilities.round(4),
        }
    )
    scores = scores.sort_values("probability_of_cancellation", ascending=False).reset_index(drop=True)
    scores["rank"] = scores.index + 1
    scores["risk_percentile"] = ((scores["rank"] / len(scores)) * 100).round(2)

    high_risk_count = max(1, math.ceil(len(scores) * risk_share))
    scores["is_high_risk"] = 0
    scores.loc[: high_risk_count - 1, "is_high_risk"] = 1
    scores["risk_segment"] = scores["is_high_risk"].map({1: "top_30_percent", 0: "regular"})

    return scores


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch LightGBM scoring for new bookings.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to the raw CSV with new bookings.",
    )
    parser.add_argument(
        "--scores-output",
        default=DEFAULT_SCORES_PATH,
        help="Path to save the ranked scoring table.",
    )
    parser.add_argument(
        "--high-risk-output",
        default=DEFAULT_HIGH_RISK_PATH,
        help="Path to save only the high-risk segment.",
    )
    parser.add_argument(
        "--risk-share",
        type=float,
        default=0.3,
        help="Share of bookings to mark as high risk, e.g. 0.3 for Top-30%%.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_df = pd.read_csv(args.input)
    categorical_columns = load_metadata()
    booking_ids, features, summary = prepare_features(raw_df, categorical_columns)

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    probabilities = pd.Series(model.predict_proba(features)[:, 1])

    scoring_table = build_scoring_table(booking_ids, probabilities, args.risk_share)
    high_risk_table = scoring_table[scoring_table["is_high_risk"] == 1].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.scores_output), exist_ok=True)
    scoring_table.to_csv(args.scores_output, index=False)
    high_risk_table.to_csv(args.high_risk_output, index=False)

    print("Входной файл:", args.input)
    print("Строк до предобработки:", summary["original_shape"][0])
    print("Строк после предобработки:", summary["processed_shape"][0])
    print("Полный скоринг сохранён в:", args.scores_output)
    print("Top-30% риск-сегмент сохранён в:", args.high_risk_output)
    print("\nПервые 5 строк скоринга:")
    print(scoring_table.head())


if __name__ == "__main__":
    main()
