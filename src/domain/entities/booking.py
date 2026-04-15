from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Booking:
    booking_id: str | None
    number_of_adults: int
    number_of_children: int
    number_of_weekend_nights: int
    number_of_week_nights: int
    type_of_meal: str
    car_parking_space: int
    room_type: str
    lead_time: int
    market_segment_type: str
    repeated: int
    previous_cancellations: int
    previous_not_canceled: int
    average_price: float
    special_requests: int
    date_of_reservation: str

    @classmethod
    def from_payload(cls, payload: dict) -> "Booking":
        return cls(
            booking_id=payload.get("Booking_ID"),
            number_of_adults=payload["number of adults"],
            number_of_children=payload["number of children"],
            number_of_weekend_nights=payload["number of weekend nights"],
            number_of_week_nights=payload["number of week nights"],
            type_of_meal=payload["type of meal"],
            car_parking_space=payload["car parking space"],
            room_type=payload["room type"],
            lead_time=payload["lead time"],
            market_segment_type=payload["market segment type"],
            repeated=payload["repeated"],
            previous_cancellations=payload["P-C"],
            previous_not_canceled=payload["P-not-C"],
            average_price=payload["average price"],
            special_requests=payload["special requests"],
            date_of_reservation=payload["date of reservation"],
        )

    def to_payload(self) -> dict:
        data = asdict(self)
        return {
            "Booking_ID": data["booking_id"],
            "number of adults": data["number_of_adults"],
            "number of children": data["number_of_children"],
            "number of weekend nights": data["number_of_weekend_nights"],
            "number of week nights": data["number_of_week_nights"],
            "type of meal": data["type_of_meal"],
            "car parking space": data["car_parking_space"],
            "room type": data["room_type"],
            "lead time": data["lead_time"],
            "market segment type": data["market_segment_type"],
            "repeated": data["repeated"],
            "P-C": data["previous_cancellations"],
            "P-not-C": data["previous_not_canceled"],
            "average price": data["average_price"],
            "special requests": data["special_requests"],
            "date of reservation": data["date_of_reservation"],
        }
