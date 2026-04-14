from src.domain.rules.risk_rules import (
    batch_segment_name,
    high_risk_count,
    is_high_risk_by_threshold,
    risk_segment_name,
)


def test_is_high_risk_by_threshold_true():
    assert is_high_risk_by_threshold(0.8, 0.7) is True


def test_is_high_risk_by_threshold_false():
    assert is_high_risk_by_threshold(0.4, 0.7) is False


def test_high_risk_count_rounds_up():
    assert high_risk_count(10, 0.3) == 3
    assert high_risk_count(7, 0.3) == 3


def test_high_risk_count_has_minimum_one():
    assert high_risk_count(1, 0.01) == 1


def test_risk_segment_name():
    assert risk_segment_name(True) == "high_risk"
    assert risk_segment_name(False) == "regular"


def test_batch_segment_name():
    assert batch_segment_name(True) == "top_30_percent"
    assert batch_segment_name(False) == "regular"