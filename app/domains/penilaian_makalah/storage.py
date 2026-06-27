"""MinIO helpers for penilaian_makalah."""

from __future__ import annotations

from typing import List, Optional

from app.core.logger import log
from app.domains.penilaian_makalah.core.config import settings as domain_settings


class MinioStorage:
    @staticmethod
    def get_minio_client():
        try:
            import boto3
            from botocore.client import Config

            return boto3.client(
                "s3",
                endpoint_url=domain_settings.minio_endpoint,
                aws_access_key_id=domain_settings.minio_access_key,
                aws_secret_access_key=domain_settings.minio_secret_key,
                config=Config(signature_version="s3v4"),
            )
        except Exception as exc:
            log.error(f"MinIO connection failed: {exc}")
            return None

    @staticmethod
    def ensure_bucket(client, bucket_name: str) -> None:
        try:
            existing = [item["Name"] for item in client.list_buckets().get("Buckets", [])]
            if bucket_name not in existing:
                client.create_bucket(Bucket=bucket_name)
        except Exception as exc:
            log.warning(f"Could not ensure bucket '{bucket_name}': {exc}")

    @classmethod
    def list_minio_files(cls, bucket: str) -> List[str]:
        client = cls.get_minio_client()
        if not client:
            return []
        try:
            response = client.list_objects_v2(Bucket=bucket)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception:
            return []

    @classmethod
    def list_minio_files_detailed(cls, bucket: str) -> List[dict]:
        client = cls.get_minio_client()
        if not client:
            return []
        try:
            response = client.list_objects_v2(Bucket=bucket)
            results = []
            for obj in response.get("Contents", []):
                results.append(
                    {
                        "Nama File": obj["Key"],
                        "Ukuran": f"{obj['Size']:,} bytes",
                        "Terakhir Diubah": obj["LastModified"].strftime("%Y-%m-%d %H:%M")
                        if obj.get("LastModified")
                        else "-",
                    }
                )
            return results
        except Exception:
            return []

    @classmethod
    def upload_to_minio(cls, bucket: str, key: str, data: bytes) -> bool:
        client = cls.get_minio_client()
        if not client:
            return False
        try:
            cls.ensure_bucket(client, bucket)
            client.put_object(Bucket=bucket, Key=key, Body=data)
            return True
        except Exception as exc:
            log.error(f"MinIO upload failed: {exc}")
            return False

    @classmethod
    def download_from_minio(cls, bucket: str, key: str):
        client = cls.get_minio_client()
        if not client:
            return None
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()
        except Exception as exc:
            log.error(f"MinIO download failed: {exc}")
            return None


minio_storage = MinioStorage()
