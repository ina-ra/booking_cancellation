import pandas as pd

from src.config import settings


def preprocess_booking_data(df: pd.DataFrame, is_training: bool = True):
    processed_df = df.copy()
    original_columns = processed_df.columns.tolist()

    booking_ids = None
    if settings.id_column in processed_df.columns:
        booking_ids = processed_df[settings.id_column].copy()
        processed_df = processed_df.drop(columns=[settings.id_column])

    num_cols = processed_df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].median())

    cat_cols = processed_df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])

    duplicates_count = int(processed_df.duplicated().sum())
    if is_training:
        processed_df = processed_df.drop_duplicates()

    if settings.target_column in processed_df.columns:
        processed_df[settings.target_column] = processed_df[settings.target_column].map(
            {"Canceled": 1, "Not_Canceled": 0}
        )

    processed_df[settings.date_column] = pd.to_datetime(
        processed_df[settings.date_column], format="mixed", errors="coerce"
    )
    invalid_dates_count = int(processed_df[settings.date_column].isna().sum())
    processed_df = processed_df.dropna(subset=[settings.date_column])

    processed_df["reservation_month"] = processed_df[settings.date_column].dt.month
    processed_df["reservation_dayofweek"] = processed_df[settings.date_column].dt.dayofweek
    processed_df = processed_df.drop(columns=[settings.date_column])

    processed_df["total_nights"] = (
        processed_df["number of weekend nights"] + processed_df["number of week nights"]
    )
    processed_df["price_per_night"] = (
        processed_df["average price"] / (processed_df["total_nights"] + 1)
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
        processed_df[settings.id_column] = booking_ids.loc[processed_df.index]

    summary = {
        "original_shape": df.shape,
        "processed_shape": processed_df.shape,
        "duplicates_count": duplicates_count,
        "invalid_dates_count": invalid_dates_count,
        "added_features": list(set(processed_df.columns.tolist()) - set(original_columns)),
        "removed_features": list(set(original_columns) - set(processed_df.columns.tolist())),
    }
    return processed_df, summary


def save_processed_training_dataset():
    raw_df = pd.read_csv(settings.raw_booking_data_path)
    processed_df, summary = preprocess_booking_data(raw_df, is_training=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(settings.processed_booking_data_path, index=False)
    return processed_df, summary

