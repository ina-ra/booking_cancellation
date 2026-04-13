from fastapi import APIRouter, HTTPException

from app.core.model_loader import model_registry
from app.schemas.request import BatchBookingRequest, BookingRequest
from app.schemas.response import BatchPredictionResponse, HealthResponse, PredictionResponse
from app.services.prediction_service import predict_batch, predict_one

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

    result = predict_one(booking.model_dump(by_alias=True))
    return PredictionResponse(**result)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_route(request: BatchBookingRequest):
    if not model_registry.is_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    predictions = predict_batch(
        payloads=[item.model_dump(by_alias=True) for item in request.bookings],
        risk_share=request.risk_share,
    )
    return BatchPredictionResponse(predictions=predictions)