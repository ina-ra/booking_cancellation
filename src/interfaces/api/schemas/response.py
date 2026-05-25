from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


class ReadinessResponse(BaseModel):
    status: str
    ready: bool
    model_loaded: bool
    model_name: str
    postgres_configured: bool
    s3_configured: bool
    missing_dependencies: List[str]


class PredictionResponse(BaseModel):
    booking_id: Optional[str]
    probability_of_cancellation: float
    is_high_risk: int
    risk_segment: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

