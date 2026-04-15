import numpy as np
import pandas as pd

from src.application.training import evaluate_model, train_lightgbm_pipeline


class DummyEvalModel:
    def predict(self, x_test):
        return [1, 0, 1, 0]

    def predict_proba(self, x_test):
        return np.array([
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.7, 0.3],
        ])


def test_evaluate_model_returns_all_metrics():
    x_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_test = pd.Series([1, 0, 1, 0])

    result = evaluate_model(DummyEvalModel(), x_test, y_test)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}

    for key in result:
        assert isinstance(result[key], float)
        assert 0.0 <= result[key] <= 1.0


class FakeBooster:
    def __init__(self):
        self.saved_path = None

    def save_model(self, path):
        self.saved_path = path


class FakeLGBMClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.booster_ = FakeBooster()
        self.fit_called = False

    def fit(self, x_train, y_train, categorical_feature=None):
        self.fit_called = True
        self.x_train = x_train
        self.y_train = y_train
        self.categorical_feature = categorical_feature

    def predict(self, x_test):
        return [0 for _ in range(len(x_test))]

    def predict_proba(self, x_test):
        return [[0.8, 0.2] for _ in range(len(x_test))]


def test_train_lightgbm_pipeline(monkeypatch):
    raw_df = pd.DataFrame(
        [
            {
                "Booking_ID": "INN00001",
                "feature_num": 10,
                "feature_cat": "A",
                "booking status": 1,
            },
            {
                "Booking_ID": "INN00002",
                "feature_num": 20,
                "feature_cat": "B",
                "booking status": 0,
            },
            {
                "Booking_ID": "INN00003",
                "feature_num": 30,
                "feature_cat": "A",
                "booking status": 1,
            },
            {
                "Booking_ID": "INN00004",
                "feature_num": 40,
                "feature_cat": "B",
                "booking status": 0,
            },
        ]
    )

    processed_df = raw_df.copy()
    preprocessing_summary = {"rows": 4}

    def fake_read_csv(path):
        return raw_df

    def fake_preprocess_booking_data(df, is_training=True):
        return processed_df, preprocessing_summary

    def fake_train_test_split(x, y, test_size, random_state, stratify):
        x_train = x.iloc[:2].copy()
        x_test = x.iloc[2:].copy()
        y_train = y.iloc[:2].copy()
        y_test = y.iloc[2:].copy()
        return x_train, x_test, y_train, y_test

    def fake_evaluate_model(model, x_test, y_test):
        return {
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.7,
            "f1": 0.72,
            "roc_auc": 0.81,
        }

    class FakeSettings:
        raw_booking_data_path = "dummy.csv"
        target_column = "booking status"
        id_column = "Booking_ID"
        test_size = 0.2
        random_state = 42
        postgres_enabled = False

    upload_calls = {}

    monkeypatch.setattr("src.application.training.settings", FakeSettings)
    monkeypatch.setattr("src.application.training.pd.read_csv", fake_read_csv)
    monkeypatch.setattr(
        "src.application.training.preprocess_booking_data",
        fake_preprocess_booking_data,
    )
    monkeypatch.setattr("src.application.training.train_test_split", fake_train_test_split)
    monkeypatch.setattr("src.application.training.evaluate_model", fake_evaluate_model)
    monkeypatch.setattr("src.application.training.LGBMClassifier", FakeLGBMClassifier)
    monkeypatch.setattr(
        "src.application.training.upload_training_artifacts",
        lambda model, report: upload_calls.update({"model": model, "report": report}),
    )

    result = train_lightgbm_pipeline()

    assert result.metrics["accuracy"] == 0.8
    assert result.parameters["n_estimators"] == 700
    assert result.report["best_model"]["name"] == "LightGBM"
    assert result.report["categorical_columns"] == ["feature_cat"]
    assert isinstance(upload_calls["model"], FakeLGBMClassifier)
    assert upload_calls["report"]["best_model"]["name"] == "LightGBM"
