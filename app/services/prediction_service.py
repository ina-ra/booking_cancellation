import math
from typing import Any

import pandas as pd

from app.core.config import DEFAULT_BATCH_RISK_SHARE, DEFAULT_HIGH_RISK_THRESHOLD
from app.core.model_loader import model_registry
from src.preprocessing import preprocess_booking_data

TARGET_COLUMN = "booking status"
ID_COLUMN = "Booking_ID"


def prepare_features(df: pd.DataFrame):
    processed_df, summary = preprocess_booking_data(df, is_training=False)

    if processed_df.empty:
        raise ValueError("No valid rows left after preprocessing")

    booking_ids = None
    if ID_COLUMN in processed_df.columns:
        booking_ids = processed_df[ID_COLUMN].copy()

    feature_df = processed_df.drop(
        columns=[col for col in [ID_COLUMN, TARGET_COLUMN] if col in processed_df.columns],
        errors="ignore",
    )

    for column in model_registry.categorical_columns:
        if column in feature_df.columns:
            feature_df[column] = feature_df[column].astype("category")

    return booking_ids, feature_df, summary


def predict_one(payload: dict[str, Any]) -> dict[str, Any]:
    df = pd.DataFrame([payload])
    booking_ids, features, _ = prepare_features(df)

    probabilities = model_registry.model.predict_proba(features)[:, 1]
    probability = float(probabilities[0])

    booking_id = None
    if booking_ids is not None and len(booking_ids) > 0:
        booking_id = booking_ids.iloc[0]

    is_high_risk = probability >= DEFAULT_HIGH_RISK_THRESHOLD

    return {
        "booking_id": booking_id,
        "probability_of_cancellation": round(probability, 4),
        "is_high_risk": is_high_risk,
        "risk_segment": "high_risk" if is_high_risk else "regular",
    }


def build_scoring_table(
    booking_ids: pd.Series | None,
    probabilities: pd.Series,
    risk_share: float = DEFAULT_BATCH_RISK_SHARE,
) -> pd.DataFrame:
    if booking_ids is None:
        booking_ids = pd.Series(range(1, len(probabilities) + 1), name=ID_COLUMN)

    scores = pd.DataFrame(
        {
            "booking_id": booking_ids.values,
            "probability_of_cancellation": probabilities.round(4),
        }
    )

    scores = scores.sort_values("probability_of_cancellation", ascending=False).reset_index(drop=True)

    high_risk_count = max(1, math.ceil(len(scores) * risk_share))
    scores["is_high_risk"] = False
    scores.loc[: high_risk_count - 1, "is_high_risk"] = True
    scores["risk_segment"] = scores["is_high_risk"].map(
        {
            True: "top_risk_segment",
            False: "regular",
        }
    )

    return scores


def predict_batch(payloads: list[dict[str, Any]], risk_share: float = DEFAULT_BATCH_RISK_SHARE):
    df = pd.DataFrame(payloads)
    booking_ids, features, _ = prepare_features(df)

    probabilities = pd.Series(model_registry.model.predict_proba(features)[:, 1])
    scores = build_scoring_table(booking_ids, probabilities, risk_share)

    return scores.to_dict(orient="records")