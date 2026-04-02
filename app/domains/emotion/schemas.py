# app/domains/emotion/schemas.py
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


# ==========================================
# REQUEST MODELS
# ==========================================
class ClassifyEmotionRequest(BaseModel):
    """Request model for emotion classification"""

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


# ==========================================
# COMPONENT MODELS (For Nested Responses)
# ==========================================
class EmotionPercentage(BaseModel):
    """Model for emotion percentage (6 emotion categories)"""

    Neutral: float = Field(
        ...,
        description="Neutral emotion percentage",
        json_schema_extra={"example": 15.5},
    )
    Anger: float = Field(
        ...,
        description="Anger emotion percentage",
        json_schema_extra={"example": 12.0},
    )
    Joy: float = Field(
        ...,
        description="Joy emotion percentage",
        json_schema_extra={"example": 45.5},
    )
    Love: float = Field(
        ...,
        description="Love emotion percentage",
        json_schema_extra={"example": 18.0},
    )
    Sad: float = Field(
        ...,
        description="Sad emotion percentage",
        json_schema_extra={"example": 7.0},
    )
    Fear: float = Field(
        ...,
        description="Fear emotion percentage",
        json_schema_extra={"example": 2.0},
    )


class EmotionDocumentDetail(BaseModel):
    """Detail of a single document's emotion classification"""

    raw_text: str = Field(
        ..., json_schema_extra={"example": "@bankbsi_id 8 kali muncul Logo BYOND"}
    )
    preprocessed_text: str = Field(
        ..., json_schema_extra={"example": "muncul logo byond bsi"}
    )
    topic: int | str | None = Field(None, json_schema_extra={"example": 0})

    emotion_cnn: str = Field(..., json_schema_extra={"example": "Joy"})
    emotion_cnn_probability: Dict[str, float] = Field(
        ...,
        json_schema_extra={
            "example": {
                "Neutral": 0.05,
                "Anger": 0.10,
                "Joy": 0.65,
                "Love": 0.15,
                "Sad": 0.03,
                "Fear": 0.02,
            }
        },
    )

    emotion_bilstm: str = Field(..., json_schema_extra={"example": "Joy"})
    emotion_bilstm_probability: Dict[str, float] = Field(
        ...,
        json_schema_extra={
            "example": {
                "Neutral": 0.04,
                "Anger": 0.08,
                "Joy": 0.70,
                "Love": 0.12,
                "Sad": 0.04,
                "Fear": 0.02,
            }
        },
    )


class ClassifyEmotionData(BaseModel):
    """Data payload for emotion classification response"""

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
    documents: List[EmotionDocumentDetail] = Field(
        ..., description="List of classified documents"
    )

    emotion_percentage_cnn: EmotionPercentage = Field(
        ..., description="Overall CNN emotion percentage"
    )
    emotion_percentage_bilstm: EmotionPercentage = Field(
        ..., description="Overall BiLSTM emotion percentage"
    )

    emotion_by_topic_cnn: Dict[str, EmotionPercentage] = Field(
        ..., description="CNN emotion percentage grouped by topic"
    )
    emotion_by_topic_bilstm: Dict[str, EmotionPercentage] = Field(
        ..., description="BiLSTM emotion percentage grouped by topic"
    )


# ==========================================
# RESPONSE MODELS
# ==========================================
class ClassifyEmotionResponse(BaseModel):
    """Main response model for /predict endpoint"""

    status: str = Field(
        ...,
        description="Status of the classification",
        json_schema_extra={"example": "success"},
    )
    message: str = Field(
        ...,
        description="Message of the classification",
        json_schema_extra={"example": "Emotion classification completed"},
    )
    # Menggunakan Union/| dict untuk fleksibilitas jika format data bervariasi
    data: Union[ClassifyEmotionData, Dict[str, Any]] = Field(
        ..., description="Classification result data"
    )


class GetEmotionsByProjectIdResponse(BaseModel):
    """Response model for getting raw emotions array by project ID"""

    status: str = Field(
        ...,
        description="Status of the request",
        json_schema_extra={"example": "success"},
    )
    message: str = Field(
        ...,
        description="Message of the request",
        json_schema_extra={"example": "Emotions retrieved successfully"},
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Wrapper containing documents list and total count",
        json_schema_extra={
            "example": {
                "total": 1,
                "documents": [
                    {
                        "raw_text": "@bankbsi_id 8 kali muncul Logo BYOND",
                        "preprocessed_text": "muncul logo byond bsi",
                        "emotion_cnn": "Joy",
                        "emotion_cnn_probability": {
                            "Neutral": 0.05,
                            "Anger": 0.10,
                            "Joy": 0.65,
                            "Love": 0.15,
                            "Sad": 0.03,
                            "Fear": 0.02,
                        },
                        "emotion_bilstm": "Joy",
                        "emotion_bilstm_probability": {
                            "Neutral": 0.04,
                            "Anger": 0.08,
                            "Joy": 0.70,
                            "Love": 0.12,
                            "Sad": 0.04,
                            "Fear": 0.02,
                        },
                    }
                ],
            }
        },
    )
