from typing import List

from pydantic import BaseModel, field_validator


class FrontendBookingRequest(BaseModel):
    bookingId: str = ""
    reservationDate: str
    adults: int
    children: int
    weekendNights: int
    weekNights: int
    meal: str
    parking: int | str
    roomType: str
    leadTime: int
    marketSegment: str
    repeated: int | str
    previousCanceled: int
    previousNotCanceled: int
    averagePrice: float
    specialRequests: int

    @field_validator(
        "adults",
        "children",
        "weekendNights",
        "weekNights",
        "leadTime",
        "previousCanceled",
        "previousNotCanceled",
        "specialRequests",
        mode="before",
    )
    @classmethod
    def validate_non_negative_ints(cls, value: int) -> int:
        normalized = int(value)
        if normalized < 0:
            raise ValueError("Value must be non-negative")
        return normalized

    @field_validator("parking", "repeated", mode="before")
    @classmethod
    def validate_binary_flags(cls, value: int | str) -> int:
        normalized = int(value)
        if normalized not in (0, 1):
            raise ValueError("Value must be 0 or 1")
        return normalized

    @field_validator("averagePrice", mode="before")
    @classmethod
    def validate_average_price(cls, value: float) -> float:
        normalized = float(value)
        if normalized < 0:
            raise ValueError("Average price must be non-negative")
        return normalized


class FrontendBatchBookingRequest(BaseModel):
    bookings: List[FrontendBookingRequest]
    riskShare: float = 0.3

    @field_validator("riskShare")
    @classmethod
    def validate_risk_share(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("riskShare must be in the range (0, 1]")
        return value


class FrontendHealthResponse(BaseModel):
    status: str
    modelLoaded: bool
    modelName: str


class FrontendPredictionResponse(BaseModel):
    bookingId: str
    reservationDate: str
    marketSegment: str
    roomType: str
    leadTime: int
    averagePrice: float
    probabilityOfCancellation: float
    risk: int
    isHighRisk: bool
    riskSegment: str


class FrontendBatchSummaryResponse(BaseModel):
    total: int
    highRiskCount: int
    averageProbability: int


class FrontendBatchPredictionResponse(BaseModel):
    summary: FrontendBatchSummaryResponse
    predictions: List[FrontendPredictionResponse]
