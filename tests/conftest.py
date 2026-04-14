import pandas as pd
import pytest


@pytest.fixture
def sample_booking_payload():
    return {
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


@pytest.fixture
def sample_batch_payload(sample_booking_payload):
    risky_booking = {
        "Booking_ID": "INN00099",
        "number of adults": 2,
        "number of children": 0,
        "number of weekend nights": 2,
        "number of week nights": 3,
        "type of meal": "Meal Plan 1",
        "car parking space": 0,
        "room type": "Room_Type 1",
        "lead time": 180,
        "market segment type": "Online",
        "repeated": 0,
        "P-C": 2,
        "P-not-C": 0,
        "average price": 220.0,
        "special requests": 0,
        "date of reservation": "2026-10-01",
    }
    return [sample_booking_payload, risky_booking]


@pytest.fixture
def sample_probabilities():
    return pd.Series([0.2, 0.9, 0.5])