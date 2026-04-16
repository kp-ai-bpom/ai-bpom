from pydantic import BaseModel, Field, field_validator


def _normalize_table_name(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip()
    if not normalized:
        return None

    # Swagger often sends "string" as a placeholder value for optional fields.
    if normalized.lower() == "string":
        return None

    return normalized


class SendMessageRequest(BaseModel):
    """Schema request untuk mengirim pesan ke chatbot."""

    user_id: str
    message: str
    session_id: str | None = None


class GetSessionMessagesRequest(BaseModel):
    """Schema request untuk mengambil histori pesan berdasarkan session."""

    user_id: str
    session_id: str


class ListSessionsRequest(BaseModel):
    """Schema request untuk mengambil daftar session milik user."""

    user_id: str


class DeleteSessionRequest(BaseModel):
    """Schema request untuk menghapus session user."""

    user_id: str
    session_id: str


class ChatRequest(BaseModel):
    """Schema untuk request chat"""

    input: str


class EmbeddingRequest(BaseModel):
    """Schema untuk request embedding"""

    input: str


class ImportBaseKnowledgeCsvRequest(BaseModel):
    """Schema request untuk import + embedding base knowledge dari CSV."""

    csv_path: str = "data/base_knowledge.csv"
    table_name: str | None = None
    batch_size: int = Field(default=50, ge=1, le=500)
    truncate_before_insert: bool = True

    @field_validator("table_name", mode="before")
    @classmethod
    def _validate_table_name(cls, value: str | None) -> str | None:
        return _normalize_table_name(value)
