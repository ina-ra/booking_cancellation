import json

from src.application.training import train_lightgbm_pipeline
from src.config import settings


def main():
    metrics, parameters, _ = train_lightgbm_pipeline()
    print("\nLightGBM")
    print("--------")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")
    print("\nПараметры LightGBM:")
    print(json.dumps(parameters, ensure_ascii=False, indent=2))
    print("\nОтчёт сохранён в:", settings.model_report_path)


if __name__ == "__main__":
    main()
