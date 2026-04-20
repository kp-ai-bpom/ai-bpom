from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class ChatErrorResponse(BaseModel):
    """Schema response error untuk endpoint chatbot baru."""

    status: int
    error: str


class SendMessageDataResponse(BaseModel):
    """Data response saat mengirim pesan."""

    query: str
    explanation: str
    user_id: str
    session_id: str


class SendMessageResponse(BaseModel):
    """Schema response untuk endpoint kirim pesan."""

    status: int
    data: SendMessageDataResponse


class ConversationItemResponse(BaseModel):
    """Representasi satu item percakapan dalam session."""

    question: str
    query: str
    explanation: str | None = None


class SessionMessagesDataResponse(BaseModel):
    """Data response untuk detail percakapan per session."""

    user_id: str
    session_id: str
    created_at: datetime
    updated_at: datetime
    conversations: List[ConversationItemResponse]


class SessionMessagesResponse(BaseModel):
    """Schema response untuk endpoint detail session."""

    status: int
    data: SessionMessagesDataResponse


class SessionListItemResponse(BaseModel):
    """Representasi ringkas metadata session."""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SessionListDataResponse(BaseModel):
    """Data response untuk daftar session user."""

    user_id: str
    sessions: List[SessionListItemResponse]


class SessionListResponse(BaseModel):
    """Schema response untuk endpoint list session."""

    status: int
    data: SessionListDataResponse


class DeleteSessionDataResponse(BaseModel):
    """Data response untuk penghapusan session."""

    message: str


class DeleteSessionResponse(BaseModel):
    """Schema response untuk endpoint hapus session."""

    status: int
    data: DeleteSessionDataResponse


class ChatDataResponse(BaseModel):
    """Schema untuk data response"""

    response: str


class ChatResponse(BaseModel):
    """Schema untuk response chat"""

    message: str
    data: ChatDataResponse


class EmbeddingDataResponse(BaseModel):
    """Schema untuk data response embedding"""

    embedding: List[float] | None = None


class EmbeddingResponse(BaseModel):
    """Schema untuk response embedding"""

    message: str
    data: EmbeddingDataResponse


class ImportBaseKnowledgeCsvDataResponse(BaseModel):
    """Data response untuk import + embedding base knowledge CSV."""

    table_name: str
    csv_path: str
    embedding_model: str
    total_rows: int
    targeted_rows: int
    processed_rows: int
    inserted_rows: int
    failed_rows: int
    truncated: bool
    expected_dimensions: int | None = None
    sample_errors: List[str] = Field(default_factory=list)


class ImportBaseKnowledgeCsvResponse(BaseModel):
    """Schema response untuk endpoint import CSV base knowledge."""

    status: int
    data: ImportBaseKnowledgeCsvDataResponse
