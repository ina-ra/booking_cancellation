import json

from src.application.training import train_lightgbm_pipeline
from src.config import settings


def main():
    result = train_lightgbm_pipeline()
    print("\nLightGBM")
    print("--------")
    for metric_name, value in result.metrics.items():
        print(f"{metric_name}: {value}")
    print("\nLightGBM parameters:")
    print(json.dumps(result.parameters, ensure_ascii=False, indent=2))
    print("\nModel artifacts uploaded to S3:", settings.model_report_object_name)


if __name__ == "__main__":
    main()
