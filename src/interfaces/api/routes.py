from fastapi import APIRouter, HTTPException

from src.application.scoring import predict_batch_use_case, predict_one_use_case
from src.infrastructure.ml.model_loader import model_registry
from src.interfaces.api.schemas.frontend import (
    FrontendBatchBookingRequest,
    FrontendBatchPredictionResponse,
    FrontendBatchSummaryResponse,
    FrontendBookingRequest,
    FrontendHealthResponse,
    FrontendPredictionResponse,
)
from src.interfaces.api.schemas.request import BatchBookingRequest, BookingRequest
from src.interfaces.api.schemas.response import (
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
)

router = APIRouter()
frontend_router = APIRouter(prefix="/frontend-api")


def _build_frontend_booking_payload(
    booking: FrontendBookingRequest,
    fallback_booking_id: str | None = None,
) -> dict:
    booking_id = booking.bookingId.strip() if booking.bookingId else ""
    if not booking_id and fallback_booking_id:
        booking_id = fallback_booking_id

    return {
        "Booking_ID": booking_id or None,
        "number of adults": booking.adults,
        "number of children": booking.children,
        "number of weekend nights": booking.weekendNights,
        "number of week nights": booking.weekNights,
        "type of meal": booking.meal,
        "car parking space": int(booking.parking),
        "room type": booking.roomType,
        "lead time": booking.leadTime,
        "market segment type": booking.marketSegment,
        "repeated": int(booking.repeated),
        "P-C": booking.previousCanceled,
        "P-not-C": booking.previousNotCanceled,
        "average price": booking.averagePrice,
        "special requests": booking.specialRequests,
        "date of reservation": booking.reservationDate,
    }


def _build_frontend_prediction(
    booking: FrontendBookingRequest,
    prediction: dict,
) -> FrontendPredictionResponse:
    probability = float(prediction["probability_of_cancellation"])

    return FrontendPredictionResponse(
        bookingId=str(prediction.get("booking_id") or booking.bookingId or ""),
        reservationDate=booking.reservationDate,
        marketSegment=booking.marketSegment,
        roomType=booking.roomType,
        leadTime=booking.leadTime,
        averagePrice=booking.averagePrice,
        probabilityOfCancellation=probability,
        risk=round(probability * 100),
        isHighRisk=bool(prediction.get("is_high_risk")),
        riskSegment=str(prediction.get("risk_segment") or ""),
    )


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


@frontend_router.get("/health", response_model=FrontendHealthResponse)
def frontend_health_check():
    return FrontendHealthResponse(
        status="ok",
        modelLoaded=model_registry.is_ready(),
        modelName=model_registry.model_name,
    )


@frontend_router.post("/predict", response_model=FrontendPredictionResponse)
def frontend_predict(booking: FrontendBookingRequest):
    if not model_registry.is_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    prediction = predict_one_use_case(
        _build_frontend_booking_payload(booking),
        model_registry,
    )
    return _build_frontend_prediction(booking, prediction)


@frontend_router.post("/predict/batch", response_model=FrontendBatchPredictionResponse)
def frontend_predict_batch(request: FrontendBatchBookingRequest):
    if not model_registry.is_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    prepared_bookings: list[tuple[FrontendBookingRequest, dict]] = []
    booking_lookup: dict[str, FrontendBookingRequest] = {}

    for index, booking in enumerate(request.bookings, start=1):
        fallback_booking_id = f"ROW-{index:04d}"
        payload = _build_frontend_booking_payload(booking, fallback_booking_id=fallback_booking_id)
        prepared_bookings.append((booking, payload))
        booking_lookup[str(payload["Booking_ID"])] = booking

    predictions = predict_batch_use_case(
        payloads=[payload for _, payload in prepared_bookings],
        risk_share=request.riskShare,
        model_registry=model_registry,
    )

    frontend_predictions = [
        _build_frontend_prediction(
            booking_lookup[str(prediction.get("booking_id"))],
            prediction,
        )
        for prediction in predictions
    ]

    average_probability = 0
    if frontend_predictions:
        average_probability = round(
            sum(item.risk for item in frontend_predictions) / len(frontend_predictions)
        )

    summary = FrontendBatchSummaryResponse(
        total=len(frontend_predictions),
        highRiskCount=sum(1 for item in frontend_predictions if item.isHighRisk),
        averageProbability=average_probability,
    )

    return FrontendBatchPredictionResponse(
        summary=summary,
        predictions=frontend_predictions,
    )
