import pandas as pd

from src.config import settings
from src.domain.entities.scoring import BookingRiskScore
from src.domain.rules.risk_rules import (
    batch_segment_name,
    high_risk_count,
    is_high_risk_by_threshold,
    risk_segment_name,
)
from src.infrastructure.data.preprocessing import preprocess_booking_data


def prepare_features(df: pd.DataFrame, categorical_columns: list[str]):
    processed_df, summary = preprocess_booking_data(df, is_training=False)

    booking_ids = None
    if settings.id_column in processed_df.columns:
        booking_ids = processed_df[settings.id_column].copy()

    feature_df = processed_df.drop(
        columns=[
            column
            for column in [settings.id_column, settings.target_column]
            if column in processed_df.columns
        ]
    )

    for column in categorical_columns:
        if column in feature_df.columns:
            feature_df[column] = feature_df[column].astype("category")

    return booking_ids, feature_df, summary


def build_scoring_table(
    booking_ids: pd.Series | None,
    probabilities: pd.Series,
    risk_share: float = settings.default_batch_risk_share,
) -> pd.DataFrame:
    if booking_ids is None:
        booking_ids = pd.Series(range(1, len(probabilities) + 1), name=settings.id_column)

    scores = pd.DataFrame(
        {
            "booking_id": booking_ids.values,
            "probability_of_cancellation": probabilities.round(4),
        }
    )
    scores = scores.sort_values("probability_of_cancellation", ascending=False).reset_index(drop=True)
    scores["rank"] = scores.index + 1
    scores["risk_percentile"] = ((scores["rank"] / len(scores)) * 100).round(2)

    top_count = high_risk_count(len(scores), risk_share)
    scores["is_high_risk"] = 0
    scores.loc[: top_count - 1, "is_high_risk"] = 1
    scores["risk_segment"] = scores["is_high_risk"].map({1: batch_segment_name(True), 0: batch_segment_name(False)})
    return scores


def build_single_prediction(payload: dict, model, categorical_columns: list[str]) -> dict:
    booking_ids, features, _ = prepare_features(pd.DataFrame([payload]), categorical_columns)
    probability = float(model.predict_proba(features)[:, 1][0])

    booking_id = None
    if booking_ids is not None and len(booking_ids) > 0:
        booking_id = booking_ids.iloc[0]

    high_risk = is_high_risk_by_threshold(probability, settings.default_high_risk_threshold)
    score = BookingRiskScore(
        booking_id=booking_id,
        probability_of_cancellation=round(probability, 4),
        is_high_risk=int(high_risk),
        risk_segment=risk_segment_name(high_risk),
    )
    return score.to_dict()


def build_batch_predictions(payloads: list[dict], model, categorical_columns: list[str], risk_share: float):
    booking_ids, features, _ = prepare_features(pd.DataFrame(payloads), categorical_columns)
    probabilities = pd.Series(model.predict_proba(features)[:, 1])
    scores = build_scoring_table(booking_ids, probabilities, risk_share)
    return scores.to_dict(orient="records")


def predict_one_use_case(payload: dict, model_registry) -> dict:
    return build_single_prediction(
        payload=payload,
        model=model_registry.model,
        categorical_columns=model_registry.categorical_columns,
    )


def predict_batch_use_case(payloads: list[dict], risk_share: float, model_registry):
    return build_batch_predictions(
        payloads=payloads,
        model=model_registry.model,
        categorical_columns=model_registry.categorical_columns,
        risk_share=risk_share,
    )
