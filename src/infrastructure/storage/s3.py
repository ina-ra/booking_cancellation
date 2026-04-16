from pathlib import Path
from typing import Any, cast

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from src.config import settings


class S3ArtifactStorage:
    def __init__(self, app_settings=settings):
        self.settings = app_settings
        self._client = None

    def is_enabled(self) -> bool:
        return bool(self.settings.s3_enabled)

    def _build_client(self):
        if not self.is_enabled():
            raise RuntimeError("S3 storage is not configured.")

        return boto3.client(
            "s3",
            endpoint_url=self.settings.s3_endpoint_url,
            aws_access_key_id=self.settings.s3_access_key,
            aws_secret_access_key=self.settings.s3_secret_key,
            region_name=self.settings.s3_region,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path" if self.settings.s3_use_path_style else "auto"},
            ),
        )

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def ensure_bucket_exists(self):
        if not self.is_enabled():
            return

        try:
            self.client.head_bucket(Bucket=self.settings.s3_bucket)
        except ClientError as error:
            error_code = error.response.get("Error", {}).get("Code", "")
            if error_code not in {"404", "NoSuchBucket"}:
                raise

            if not self.settings.s3_auto_create_bucket:
                raise

            self.client.create_bucket(Bucket=self.settings.s3_bucket)

    def upload_file(self, local_path: Path, object_name: str):
        self.ensure_bucket_exists()
        self.client.upload_file(str(local_path), self.settings.s3_bucket, object_name)

    def upload_bytes(
        self,
        payload: bytes,
        object_name: str,
        content_type: str = "application/octet-stream",
    ):
        self.ensure_bucket_exists()
        self.client.put_object(
            Bucket=self.settings.s3_bucket,
            Key=object_name,
            Body=payload,
            ContentType=content_type,
        )

    def upload_text(
        self,
        payload: str,
        object_name: str,
        content_type: str = "text/plain; charset=utf-8",
    ):
        self.upload_bytes(
            payload=payload.encode("utf-8"),
            object_name=object_name,
            content_type=content_type,
        )

    def download_bytes(self, object_name: str) -> bytes:
        response = self.client.get_object(Bucket=self.settings.s3_bucket, Key=object_name)
        body = cast(Any, response["Body"])
        return cast(bytes, body.read())

    def object_exists(self, object_name: str) -> bool:
        try:
            self.client.head_object(Bucket=self.settings.s3_bucket, Key=object_name)
            return True
        except ClientError as error:
            error_code = error.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise


artifact_storage = S3ArtifactStorage()
