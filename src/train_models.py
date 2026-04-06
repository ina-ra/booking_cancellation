import json
import os
import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_booking_data


RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "booking.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lightgbm_model.txt")
MODEL_PICKLE_PATH = os.path.join(ARTIFACTS_DIR, "lightgbm_model.pkl")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "model_comparison.json")

TARGET_COLUMN = "booking status"
ID_COLUMN = "Booking_ID"

# CatBoost, XGBoost и Logistic Regression в этой итерации отключены.
# Оставляем только обучение и сохранение LightGBM.


def evaluate_model(model, x_test, y_test):
    y_pred = pd.Series(model.predict(x_test)).astype(int)
    y_proba = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def main():
    raw_df = pd.read_csv(DATA_PATH)
    df, preprocessing_summary = preprocess_booking_data(raw_df, is_training=True)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Column '{TARGET_COLUMN}' not found in dataset.")

    drop_columns = [TARGET_COLUMN]
    if ID_COLUMN in df.columns:
        drop_columns.append(ID_COLUMN)

    x = df.drop(columns=drop_columns)
    y = df[TARGET_COLUMN]

    categorical_columns = x.select_dtypes(include=["object", "string"]).columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    for column in categorical_columns:
        x_train[column] = x_train[column].astype("category")
        x_test[column] = pd.Categorical(x_test[column], categories=x_train[column].cat.categories)

    lightgbm_model = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        min_split_gain=0.0,
        random_state=RANDOM_STATE,
        objective="binary",
        n_jobs=1,
        verbosity=-1,
    )
    lightgbm_model.fit(
        x_train,
        y_train,
        categorical_feature=categorical_columns,
    )

    metrics = evaluate_model(lightgbm_model, x_test, y_test)
    parameters = {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "min_split_gain": 0.0,
        "objective": "binary",
    }
    report = {
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
        "preprocessing": preprocessing_summary,
        "categorical_columns": categorical_columns,
        "parameters": {
            "LightGBM": parameters,
        },
        "metrics": {
            "LightGBM": metrics,
        },
        "best_model": {
            "name": "LightGBM",
            "reason": "only active model in the current training script",
        },
    }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    lightgbm_model.booster_.save_model(MODEL_PATH)
    with open(MODEL_PICKLE_PATH, "wb") as file:
        pickle.dump(lightgbm_model, file)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\nLightGBM")
    print("--------")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")

    print("\nПараметры LightGBM:")
    print(json.dumps(parameters, ensure_ascii=False, indent=2))
    print("\nОтчёт сохранён в:", REPORT_PATH)


if __name__ == "__main__":
    main()
