from datetime import datetime, timezone
from typing import Optional

from beanie import Document
from pydantic import Field


class SentimentModel(Document):
    """Model untuk menyimpan hasil analisis sentimen per dokumen"""

    projectId: str = Field(..., description="Project ID for data persistence")
    raw_text: str = Field(..., description="Original raw text")
    preprocessed_text: str = Field(..., description="Preprocessed text")
    topic: Optional[int] = Field(None, description="Topic ID from Topic Modeling")

    sentiment_cnn: str = Field(
        ..., description="CNN model prediction (Positif/Negatif)"
    )
    sentiment_cnn_probability: float = Field(
        ..., description="CNN model prediction probability"
    )

    sentiment_cnn_lstm: str = Field(..., description="CNN-LSTM model prediction")
    sentiment_cnn_lstm_probability: float = Field(
        ..., description="CNN-LSTM model prediction probability"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "sentiments"  # Nama collection di MongoDB
        indexes = ["projectId", "topic"]
