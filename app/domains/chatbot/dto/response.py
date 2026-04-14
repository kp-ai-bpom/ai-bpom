from typing import List

from pydantic import BaseModel


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
