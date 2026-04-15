import json
import pickle
from datetime import datetime, timezone

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.application.monitoring import build_training_monitoring_metrics
from src.config import settings
from src.infrastructure.data.preprocessing import preprocess_booking_data
from src.infrastructure.db.repositories import save_model_run, save_monitoring_metrics
from src.infrastructure.ml.artifacts import upload_training_artifacts


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


def train_lightgbm_pipeline():
    raw_df = pd.read_csv(settings.raw_booking_data_path)
    df, preprocessing_summary = preprocess_booking_data(raw_df, is_training=True)

    drop_columns = [settings.target_column]
    if settings.id_column in df.columns:
        drop_columns.append(settings.id_column)

    x = df.drop(columns=drop_columns)
    y = df[settings.target_column]
    categorical_columns = x.select_dtypes(include=["object", "string"]).columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )

    for column in categorical_columns:
        x_train[column] = x_train[column].astype("category")
        x_test[column] = pd.Categorical(x_test[column], categories=x_train[column].cat.categories)

    model = LGBMClassifier(
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
        random_state=settings.random_state,
        objective="binary",
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(x_train, y_train, categorical_feature=categorical_columns)

    metrics = evaluate_model(model, x_test, y_test)
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
            "test_size": settings.test_size,
            "random_state": settings.random_state,
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

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(settings.lightgbm_model_text_path)
    with open(settings.lightgbm_model_pickle_path, "wb") as file:
        pickle.dump(model, file)
    with open(settings.model_report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    upload_training_artifacts()

    if settings.postgres_enabled:
        model_version = f"lightgbm_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        save_model_run(
            model_name="LightGBM",
            model_version=model_version,
            metrics=metrics,
            parameters=parameters,
        )
        save_monitoring_metrics(
            run_type="train",
            model_name="LightGBM",
            metrics=build_training_monitoring_metrics(
                preprocessing_summary=preprocessing_summary,
                evaluation_metrics=metrics,
            ),
        )
    return metrics, parameters, report

