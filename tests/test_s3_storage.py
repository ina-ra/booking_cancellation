from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from src.infrastructure.storage.s3 import S3ArtifactStorage


class DummyBody:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self) -> bytes:
        return self.payload


class FakeClient:
    def __init__(self):
        self.calls = []
        self.head_bucket_error = None

    def head_bucket(self, Bucket):
        self.calls.append(("head_bucket", Bucket))
        if self.head_bucket_error is not None:
            raise self.head_bucket_error

    def create_bucket(self, Bucket):
        self.calls.append(("create_bucket", Bucket))

    def upload_file(self, local_path, bucket, object_name):
        self.calls.append(("upload_file", local_path, bucket, object_name))

    def put_object(self, **kwargs):
        self.calls.append(("put_object", kwargs))

    def get_object(self, Bucket, Key):
        self.calls.append(("get_object", Bucket, Key))
        return {"Body": DummyBody(b"payload")}


def build_settings(**overrides):
    values = {
        "s3_enabled": True,
        "s3_endpoint_url": "http://localhost:9000",
        "s3_access_key": "minio-user",
        "s3_secret_key": "minio-pass",
        "s3_region": "us-east-1",
        "s3_use_path_style": True,
        "s3_bucket": "booking-cancellation-artifacts",
        "s3_auto_create_bucket": True,
    }
    values.update(overrides)
    return type("FakeSettings", (), values)()


def build_client_error(code: str):
    return ClientError({"Error": {"Code": code, "Message": "boom"}}, "HeadBucket")


def test_build_client_raises_when_s3_is_not_configured():
    storage = S3ArtifactStorage(build_settings(s3_enabled=False))

    with pytest.raises(RuntimeError, match="S3 storage is not configured"):
        storage._build_client()


def test_client_property_builds_client_once(monkeypatch):
    built_clients = []

    def fake_boto3_client(*args, **kwargs):
        client = FakeClient()
        built_clients.append((client, args, kwargs))
        return client

    monkeypatch.setattr("src.infrastructure.storage.s3.boto3.client", fake_boto3_client)

    storage = S3ArtifactStorage(build_settings())

    first_client = storage.client
    second_client = storage.client

    assert first_client is second_client
    assert len(built_clients) == 1
    assert built_clients[0][1] == ("s3",)


def test_ensure_bucket_exists_returns_early_when_s3_disabled():
    storage = S3ArtifactStorage(build_settings(s3_enabled=False))

    storage.ensure_bucket_exists()


def test_ensure_bucket_exists_creates_missing_bucket():
    fake_client = FakeClient()
    fake_client.head_bucket_error = build_client_error("404")
    storage = S3ArtifactStorage(build_settings())
    storage._client = fake_client

    storage.ensure_bucket_exists()

    assert fake_client.calls == [
        ("head_bucket", "booking-cancellation-artifacts"),
        ("create_bucket", "booking-cancellation-artifacts"),
    ]


def test_ensure_bucket_exists_raises_when_auto_create_disabled():
    fake_client = FakeClient()
    fake_client.head_bucket_error = build_client_error("NoSuchBucket")
    storage = S3ArtifactStorage(build_settings(s3_auto_create_bucket=False))
    storage._client = fake_client

    with pytest.raises(ClientError):
        storage.ensure_bucket_exists()


def test_ensure_bucket_exists_reraises_unexpected_client_error():
    fake_client = FakeClient()
    fake_client.head_bucket_error = build_client_error("403")
    storage = S3ArtifactStorage(build_settings())
    storage._client = fake_client

    with pytest.raises(ClientError):
        storage.ensure_bucket_exists()


def test_upload_file_uses_bucket_and_object_name():
    fake_client = FakeClient()
    storage = S3ArtifactStorage(build_settings())
    storage._client = fake_client

    storage.upload_file(Path("artifacts/model.pkl"), "artifacts/model.pkl")

    assert fake_client.calls == [
        ("head_bucket", "booking-cancellation-artifacts"),
        (
            "upload_file",
            str(Path("artifacts/model.pkl")),
            "booking-cancellation-artifacts",
            "artifacts/model.pkl",
        ),
    ]


def test_download_bytes_reads_payload():
    fake_client = FakeClient()
    storage = S3ArtifactStorage(build_settings())
    storage._client = fake_client

    result = storage.download_bytes("artifacts/model.pkl")

    assert result == b"payload"
    assert fake_client.calls == [
        ("get_object", "booking-cancellation-artifacts", "artifacts/model.pkl")
    ]


def test_upload_bytes_uses_put_object():
    fake_client = FakeClient()
    storage = S3ArtifactStorage(build_settings())
    storage._client = fake_client

    storage.upload_bytes(b"payload", "artifacts/model.pkl", content_type="application/octet-stream")

    assert fake_client.calls == [
        ("head_bucket", "booking-cancellation-artifacts"),
        (
            "put_object",
            {
                "Bucket": "booking-cancellation-artifacts",
                "Key": "artifacts/model.pkl",
                "Body": b"payload",
                "ContentType": "application/octet-stream",
            },
        ),
    ]


def test_upload_text_encodes_payload():
    fake_client = FakeClient()
    storage = S3ArtifactStorage(build_settings())
    storage._client = fake_client

    storage.upload_text("payload", "artifacts/report.json", content_type="application/json")

    assert fake_client.calls == [
        ("head_bucket", "booking-cancellation-artifacts"),
        (
            "put_object",
            {
                "Bucket": "booking-cancellation-artifacts",
                "Key": "artifacts/report.json",
                "Body": b"payload",
                "ContentType": "application/json",
            },
        ),
    ]
