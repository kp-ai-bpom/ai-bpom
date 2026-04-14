from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Schema untuk request chat"""

    input: str


class EmbeddingRequest(BaseModel):
    """Schema untuk request embedding"""

    input: str
