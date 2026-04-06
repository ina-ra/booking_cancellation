import os

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "booking.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "booking_clean.csv")

TARGET_COLUMN = "booking status"
ID_COLUMN = "Booking_ID"
DATE_COLUMN = "date of reservation"


def preprocess_booking_data(df, is_training=True):
    processed_df = df.copy()
    original_columns = processed_df.columns.tolist()

    booking_ids = None
    if ID_COLUMN in processed_df.columns:
        booking_ids = processed_df[ID_COLUMN].copy()
        processed_df = processed_df.drop(columns=[ID_COLUMN])

    num_cols = processed_df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].median())

    cat_cols = processed_df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])

    duplicates_count = int(processed_df.duplicated().sum())
    if is_training:
        processed_df = processed_df.drop_duplicates()

    if TARGET_COLUMN in processed_df.columns:
        processed_df[TARGET_COLUMN] = processed_df[TARGET_COLUMN].map(
            {
                "Canceled": 1,
                "Not_Canceled": 0,
            }
        )

    processed_df[DATE_COLUMN] = pd.to_datetime(
        processed_df[DATE_COLUMN],
        format="mixed",
        errors="coerce",
    )

    invalid_dates_count = int(processed_df[DATE_COLUMN].isna().sum())
    processed_df = processed_df.dropna(subset=[DATE_COLUMN])

    processed_df["reservation_month"] = processed_df[DATE_COLUMN].dt.month
    processed_df["reservation_dayofweek"] = processed_df[DATE_COLUMN].dt.dayofweek
    processed_df = processed_df.drop(columns=[DATE_COLUMN])

    processed_df["total_nights"] = (
        processed_df["number of weekend nights"] + processed_df["number of week nights"]
    )
    processed_df["price_per_night"] = processed_df["average price"] / (
        processed_df["total_nights"] + 1
    )
    processed_df["is_family"] = (processed_df["number of children"] > 0).astype(int)
    processed_df["total_guests"] = (
        processed_df["number of adults"] + processed_df["number of children"]
    )
    processed_df["weekend_share"] = processed_df["number of weekend nights"] / (
        processed_df["total_nights"] + 1
    )
    processed_df["previous_bookings"] = processed_df["P-C"] + processed_df["P-not-C"]
    processed_df["previous_cancel_ratio"] = processed_df["P-C"] / (
        processed_df["previous_bookings"] + 1
    )
    processed_df["has_special_requests"] = (processed_df["special requests"] > 0).astype(int)

    if booking_ids is not None:
        processed_df[ID_COLUMN] = booking_ids.loc[processed_df.index]

    added_features = list(set(processed_df.columns.tolist()) - set(original_columns))
    removed_features = list(set(original_columns) - set(processed_df.columns.tolist()))

    summary = {
        "original_shape": df.shape,
        "processed_shape": processed_df.shape,
        "duplicates_count": duplicates_count,
        "invalid_dates_count": invalid_dates_count,
        "added_features": added_features,
        "removed_features": removed_features,
    }

    return processed_df, summary


def main():
    df = pd.read_csv(INPUT_PATH)
    processed_df, summary = preprocess_booking_data(df, is_training=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    processed_df.to_csv(OUTPUT_PATH, index=False)

    print("Размер датасета до обработки:", summary["original_shape"])
    print("Дубликаты:", summary["duplicates_count"])
    print("Некорректные даты:", summary["invalid_dates_count"])
    print("Размер датасета:", summary["processed_shape"])

    print("\nСписок признаков:")
    print(processed_df.columns.tolist())

    print("\nДобавленные признаки:")
    print(summary["added_features"])

    print("\nУдалённые признаки:")
    print(summary["removed_features"])

    print("\nПервые 5 строк:")
    print(processed_df.head())


if __name__ == "__main__":
    main()
