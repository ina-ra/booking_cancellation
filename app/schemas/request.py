from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BookingRequest(BaseModel):
    booking_id: Optional[str] = Field(default=None, alias="Booking_ID")

    number_of_adults: int = Field(alias="number of adults")
    number_of_children: int = Field(alias="number of children")
    number_of_weekend_nights: int = Field(alias="number of weekend nights")
    number_of_week_nights: int = Field(alias="number of week nights")
    type_of_meal: str = Field(alias="type of meal")
    car_parking_space: int = Field(alias="car parking space")
    room_type: str = Field(alias="room type")
    lead_time: int = Field(alias="lead time")
    market_segment_type: str = Field(alias="market segment type")
    repeated: int
    previous_cancellations: int = Field(alias="P-C")
    previous_not_canceled: int = Field(alias="P-not-C")
    average_price: float = Field(alias="average price")
    special_requests: int = Field(alias="special requests")
    date_of_reservation: str = Field(alias="date of reservation")

    @field_validator(
        "number_of_adults",
        "number_of_children",
        "number_of_weekend_nights",
        "number_of_week_nights",
        "car_parking_space",
        "lead_time",
        "repeated",
        "previous_cancellations",
        "previous_not_canceled",
        "special_requests",
    )
    @classmethod
    def validate_non_negative_ints(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Value must be non-negative")
        return value

    @field_validator("average_price")
    @classmethod
    def validate_average_price(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Average price must be non-negative")
        return value

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Booking_ID": "INN00001",
                "number of adults": 2,
                "number of children": 1,
                "number of weekend nights": 1,
                "number of week nights": 3,
                "type of meal": "Meal Plan 1",
                "car parking space": 0,
                "room type": "Room_Type 1",
                "lead time": 45,
                "market segment type": "Online",
                "repeated": 0,
                "P-C": 0,
                "P-not-C": 1,
                "average price": 120.5,
                "special requests": 2,
                "date of reservation": "2026-04-10",
            }
        },
    }


class BatchBookingRequest(BaseModel):
    bookings: List[BookingRequest]
    risk_share: float = 0.3

    @field_validator("risk_share")
    @classmethod
    def validate_risk_share(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("risk_share must be in the range (0, 1]")
        return value