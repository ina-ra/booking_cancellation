from dataclasses import dataclass

from src.domain.entities.scoring import BookingRiskScore


@dataclass(frozen=True)
class BatchScoringResult:
    scores: list[BookingRiskScore]

    @property
    def total_bookings(self) -> int:
        return len(self.scores)

    @property
    def high_risk_count(self) -> int:
        return sum(1 for score in self.scores if score.is_high_risk == 1)

    def to_dict_list(self) -> list[dict]:
        return [score.to_dict() for score in self.scores]
