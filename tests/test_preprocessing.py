import pandas as pd

from src.infrastructure.data.preprocessing import preprocess_booking_data


def test_preprocess_booking_data_returns_dataframe_and_summary():
    df = pd.DataFrame(
        [
            {
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
                "booking status": "Not_Canceled",
            }
        ]
    )

    processed_df, summary = preprocess_booking_data(df, is_training=True)

    assert isinstance(processed_df, pd.DataFrame)
    assert isinstance(summary, dict)
    assert len(processed_df) == 1


def test_preprocess_booking_data_without_target_for_inference():
    df = pd.DataFrame(
        [
            {
                "Booking_ID": "INN00002",
                "number of adults": 1,
                "number of children": 0,
                "number of weekend nights": 0,
                "number of week nights": 2,
                "type of meal": "Meal Plan 2",
                "car parking space": 1,
                "room type": "Room_Type 2",
                "lead time": 10,
                "market segment type": "Offline",
                "repeated": 1,
                "P-C": 0,
                "P-not-C": 3,
                "average price": 80.0,
                "special requests": 0,
                "date of reservation": "2026-04-11",
            }
        ]
    )

    processed_df, summary = preprocess_booking_data(df, is_training=False)

    assert isinstance(processed_df, pd.DataFrame)
    assert isinstance(summary, dict)
    assert len(processed_df) == 1