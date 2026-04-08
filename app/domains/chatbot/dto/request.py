from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Schema untuk request chat"""

    input: str
