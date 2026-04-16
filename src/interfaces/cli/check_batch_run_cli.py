import argparse

from src.infrastructure.storage.batch_runs import (
    batch_run_exists,
    build_high_risk_object_name,
    build_scores_object_name,
    build_success_marker_object_name,
)
from src.infrastructure.storage.s3 import artifact_storage


def parse_args():
    parser = argparse.ArgumentParser(description="Check batch run state in S3.")
    parser.add_argument("--run-date", required=True)
    parser.add_argument(
        "--mode",
        choices=["precheck", "verify-outputs"],
        required=True,
    )
    return parser.parse_args()


def _ensure_object_exists(object_name: str) -> None:
    if not artifact_storage.object_exists(object_name):
        raise FileNotFoundError(f"Expected S3 object is missing: {object_name}")


def main():
    args = parse_args()

    if args.mode == "precheck":
        if batch_run_exists(args.run_date):
            print(f"Batch run for {args.run_date} is already marked as successful.")
            raise SystemExit(99)

        print(f"Batch run for {args.run_date} has not been processed yet.")
        return

    _ensure_object_exists(build_scores_object_name(args.run_date))
    _ensure_object_exists(build_high_risk_object_name(args.run_date))
    _ensure_object_exists(build_success_marker_object_name(args.run_date))
    print(f"All batch outputs are present for {args.run_date}.")


if __name__ == "__main__":
    main()
