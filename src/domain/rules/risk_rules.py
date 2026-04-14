import math


def is_high_risk_by_threshold(probability: float, threshold: float) -> bool:
    return probability >= threshold


def high_risk_count(total_rows: int, risk_share: float) -> int:
    return max(1, math.ceil(total_rows * risk_share))


def risk_segment_name(is_high_risk: bool) -> str:
    return "high_risk" if is_high_risk else "regular"


def batch_segment_name(is_high_risk: bool) -> str:
    return "top_30_percent" if is_high_risk else "regular"

