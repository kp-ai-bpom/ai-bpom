import io

from minio import Minio
from minio.error import S3Error

from app.core.config import settings
from app.core.logger import log


class MinioClient:
    def __init__(self):
        self._client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self._bucket = settings.MINIO_BUCKET
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
            log.info(f"🪣 Created MinIO bucket: {self._bucket}")

    def list_documents(self, prefix: str = "") -> list[str]:
        objects = self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
        return [
            obj.object_name
            for obj in objects
            if obj.object_name.endswith(".xlsx")
        ]

    def download_file(self, object_name: str) -> bytes:
        response = self._client.get_object(self._bucket, object_name)
        data = response.read()
        response.close()
        response.release_conn()
        return data

    def upload_file(self, object_name: str, data: bytes, content_type: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
        self._client.put_object(
            self._bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        log.info(f"📤 Uploaded '{object_name}' to MinIO bucket '{self._bucket}'")

    def file_exists(self, object_name: str) -> bool:
        try:
            self._client.stat_object(self._bucket, object_name)
            return True
        except S3Error:
            return False