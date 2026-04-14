from fastapi import APIRouter, HTTPException

from src.application.scoring import predict_batch_use_case, predict_one_use_case
from src.infrastructure.ml.model_loader import model_registry
from src.interfaces.api.schemas.request import BatchBookingRequest, BookingRequest
from src.interfaces.api.schemas.response import (
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model_registry.is_ready(),
        model_name=model_registry.model_name,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(booking: BookingRequest):
    if not model_registry.is_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    result = predict_one_use_case(booking.model_dump(by_alias=True), model_registry)
    return PredictionResponse(**result)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_route(request: BatchBookingRequest):
    if not model_registry.is_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    predictions = predict_batch_use_case(
        payloads=[item.model_dump(by_alias=True) for item in request.bookings],
        risk_share=request.risk_share,
        model_registry=model_registry,
    )
    return BatchPredictionResponse(predictions=predictions)
