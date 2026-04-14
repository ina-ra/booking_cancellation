from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BookingRiskScore:
    booking_id: str | int | None
    probability_of_cancellation: float
    rank: int | None = None
    risk_percentile: float | None = None
    is_high_risk: int | None = None
    risk_segment: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

