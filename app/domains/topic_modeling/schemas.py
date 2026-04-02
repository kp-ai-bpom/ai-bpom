from pydantic import BaseModel, Field


# Request Model
class AnalyzeTopicRequest(BaseModel):
    """Request model for topic analysis"""

    project_id: str = Field(
        ...,
        description="Project ID for data persistence",
        json_schema_extra={"example": "proj_123456"},
    )
    keyword: str = Field(
        ...,
        description="Keyword to search for",
        json_schema_extra={"example": "byond bsi"},
    )
    start_date: str = Field(
        ...,
        description="Start date (YYYY-MM-DD)",
        json_schema_extra={"example": "2025-02-08"},
    )
    end_date: str = Field(
        ...,
        description="End date (YYYY-MM-DD)",
        json_schema_extra={"example": "2025-02-08"},
    )

    class Config:
        json_schema_extra = {
            "example": {
                "project_id": "proj_123456",
                "keyword": "byond bsi",
                "start_date": "2025-02-08",
                "end_date": "2025-02-08",
            }
        }


# Response Model
class AnalyzeTopicResponse(BaseModel):
    """Response model for topic analysis"""

    status: str = Field(
        ...,
        description="Status of the topic analysis",
        json_schema_extra={"example": "Success"},
    )
    message: str = Field(
        ...,
        description="Message of the topic analysis",
        json_schema_extra={"example": "Topic analysis completed successfully"},
    )
    data: dict = Field(
        ...,
        description="Data of the topic analysis",
        json_schema_extra={
            "example": {
                "project_id": "proj_123456",
                "status": "completed",
                "topic_modelling": True,
                "topics": ["Topic 1", "Topic 2"],
                "keyword": "byond bsi",
                "start_date": "2025-02-08",
                "end_date": "2025-02-08",
                "timestamp": "2025-02-08 12:00:00",
            }
        },
    )


class GetTopicsByProjectIdResponse(BaseModel):
    """Response model for getting topics by project ID"""

    status: str = Field(
        ...,
        description="Status of the topic analysis",
        json_schema_extra={"example": "Success"},
    )
    message: str = Field(
        ...,
        description="Message of the topic analysis",
        json_schema_extra={"example": "Topics retrieved successfully"},
    )
    data: list = Field(
        ...,
        description="Data of the topic analysis",
        json_schema_extra={
            "example": [
                {
                    "topicId": 1,
                    "projectId": "proj_123456",
                    "context": "Logo BYOND by BSI",
                    "words": ["Logo", "BYOND", "BSI"],
                    "keyword": "byond bsi",
                }
            ]
        },
    )


class GetDocumentsByProjectIdResponse(BaseModel):
    """Response model for getting documents by project ID"""

    status: str = Field(
        ...,
        description="Status of the topic analysis",
        json_schema_extra={"example": "Success"},
    )
    message: str = Field(
        ...,
        description="Message of the topic analysis",
        json_schema_extra={"example": "Documents retrieved successfully"},
    )
    data: list = Field(
        ...,
        description="Data of the topic analysis",
        json_schema_extra={
            "example": [
                {
                    "documentId": 1,
                    "projectId": "proj_123456",
                    "full_text": "Logo BYOND by BSI",
                    "raw_text": "Logo BYOND by BSI",
                    "username": "brugmansia_",
                    "tweet_url": "https://x.com/brugmansia_/status/1888336969257889924",
                    "topic": "0",
                    "probability": "1.0",
                }
            ]
        },
    )
