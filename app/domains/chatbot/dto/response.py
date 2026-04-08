from pydantic import BaseModel


class DataResponse(BaseModel):
    """Schema untuk data response"""

    response: str


class ChatResponse(BaseModel):
    """Schema untuk response chat"""

    message: str
    data: DataResponse
