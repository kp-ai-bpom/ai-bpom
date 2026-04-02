from datetime import datetime, timezone
from typing import Dict, Optional

from beanie import Document
from pydantic import Field


class EmotionModel(Document):
    """Model untuk menyimpan hasil analisis emosi per dokumen"""

    projectId: str = Field(..., description="Project ID for data persistence")
    raw_text: str = Field(..., description="Original raw text")
    preprocessed_text: str = Field(..., description="Preprocessed text")
    topic: Optional[int] = Field(None, description="Topic ID from Topic Modeling")

    emotion_cnn: str = Field(..., description="CNN model prediction")
    emotion_cnn_probability: Dict[str, float] = Field(
        ..., description="CNN probabilities for all emotions"
    )

    emotion_bilstm: str = Field(..., description="BiLSTM model prediction")
    emotion_bilstm_probability: Dict[str, float] = Field(
        ..., description="BiLSTM probabilities for all emotions"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "emotions"  # Nama collection di MongoDB
        indexes = ["projectId", "topic"]
