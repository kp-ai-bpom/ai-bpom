from pydantic import BaseModel, Field


# Request Model
class ClassifySentimentRequest(BaseModel):
    """Request model for sentiment classification"""

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


# Response Models
class SentimentPercentage(BaseModel):
    """Model for sentiment percentage"""

    positive: float = Field(
        ...,
        description="Positive sentiment percentage",
        json_schema_extra={"example": 65.5},
    )
    negative: float = Field(
        ...,
        description="Negative sentiment percentage",
        json_schema_extra={"example": 34.5},
    )


class SentimentByTopic(BaseModel):
    """Model for sentiment percentage by topic"""

    positive: float = Field(
        ...,
        description="Positive sentiment percentage",
        json_schema_extra={"example": 70.0},
    )
    negative: float = Field(
        ...,
        description="Negative sentiment percentage",
        json_schema_extra={"example": 30.0},
    )
    total: int = Field(
        ...,
        description="Total documents in topic",
        json_schema_extra={"example": 20},
    )


class ClassifySentimentData(BaseModel):
    """Data model for sentiment classification response"""

    project_id: str = Field(
        ...,
        description="Project ID",
        json_schema_extra={"example": "proj_123456"},
    )
    total: int = Field(
        ...,
        description="Total documents classified",
        json_schema_extra={"example": 100},
    )
    documents: list = Field(
        ...,
        description="List of classified documents",
        json_schema_extra={
            "example": [
                {
                    "_id": "6950d625d8d001249fcee3f1",
                    "raw_text": "@bankbsi_id 8 kali muncul Logo BYOND",
                    "preprocessed_text": "muncul logo byond bsi",
                    "sentiment_cnn": "Positif",
                    "sentiment_cnn_probability": 0.85,
                    "sentiment_cnn_lstm": "Positif",
                    "sentiment_cnn_lstm_probability": 0.92,
                    "project_id": "proj_123456",
                }
            ]
        },
    )
    sentiment_percentage_cnn: SentimentPercentage = Field(
        ..., description="Overall CNN sentiment percentage"
    )
    sentiment_percentage_cnn_lstm: SentimentPercentage = Field(
        ..., description="Overall CNN-LSTM sentiment percentage"
    )
    sentiment_by_topic_cnn: dict = Field(
        ...,
        description="CNN sentiment percentage by topic",
        json_schema_extra={
            "example": {"0": {"positive": 70.0, "negative": 30.0, "total": 20}}
        },
    )
    sentiment_by_topic_cnn_lstm: dict = Field(
        ...,
        description="CNN-LSTM sentiment percentage by topic",
        json_schema_extra={
            "example": {"0": {"positive": 75.0, "negative": 25.0, "total": 20}}
        },
    )


class ClassifySentimentResponse(BaseModel):
    """Response model for sentiment classification"""

    status: str = Field(
        ...,
        description="Status of the classification",
        json_schema_extra={"example": "success"},
    )
    message: str = Field(
        ...,
        description="Message of the classification",
        json_schema_extra={"example": "Sentiment classification completed"},
    )
    data: ClassifySentimentData = Field(..., description="Classification result data")


class GetSentimentsByProjectIdResponse(BaseModel):
    """Response model for getting sentiments by project ID"""

    status: str = Field(
        ...,
        description="Status of the request",
        json_schema_extra={"example": "success"},
    )
    message: str = Field(
        ...,
        description="Message of the request",
        json_schema_extra={"example": "Sentiments retrieved successfully"},
    )
    data: list = Field(
        ...,
        description="List of sentiments",
        json_schema_extra={
            "example": [
                {
                    "_id": "6950d625d8d001249fcee3f1",
                    "raw_text": "@bankbsi_id 8 kali muncul Logo BYOND",
                    "preprocessed_text": "muncul logo byond bsi",
                    "sentiment_cnn": "Positif",
                    "sentiment_cnn_probability": 0.85,
                    "sentiment_cnn_lstm": "Positif",
                    "sentiment_cnn_lstm_probability": 0.92,
                    "project_id": "proj_123456",
                }
            ]
        },
    )
