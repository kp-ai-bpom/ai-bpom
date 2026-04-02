import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    # Environment
    ENV: str = os.getenv("ENV", "development")

    # MongoDB Configuration
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "socialabs_ai_db")

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL", None)

    MODEL_CNN_SENTIMENT_FILENAME: str = "latest_model-cnn-sentiment.h5"
    TOKENIZER_CNN_SENTIMENT_FILENAME: str = "latest_tokenizer-cnn-sentiment.pickle"
    MODEL_CNN_LSTM_SENTIMENT_FILENAME: str = "latest_model-cnn-lstm-sentiment.h5"
    TOKENIZER_CNN_LSTM_SENTIMENT_FILENAME: str = (
        "latest_tokenizer-cnn-lstm-sentiment.pickle"
    )

    MODEL_CNN_EMOTION_FILENAME: str = "latest_model-cnn-emotion.h5"
    TOKENIZER_CNN_EMOTION_FILENAME: str = "latest_tokenizer-cnn-emotion.pickle"
    MODEL_BILSTM_EMOTION_FILENAME: str = "latest_model-bilstm-emotion.h5"
    TOKENIZER_BILSTM_EMOTION_FILENAME: str = "latest_tokenizer-bilstm-emotion.pickle"

    # Path Configuration
    MODELS_BASE_PATH: str = os.getenv("MODELS_BASE_PATH", "models/raw")

    # Computed Paths (using @property for dynamic path resolution)
    @property
    def models_base_dir(self) -> Path:
        """Absolute path to models base directory"""
        return PROJECT_ROOT / self.MODELS_BASE_PATH

    @property
    def topic_modeling_path(self) -> Path:
        """Path to topic modeling models and preprocessing"""
        return self.models_base_dir / "topic_modeling"

    @property
    def topic_modeling_utils_path(self) -> Path:
        """Path to topic modeling utils (KBBA dictionary, etc.)"""
        return self.topic_modeling_path / "utils"

    @property
    def topic_modeling_preprocessing_path(self) -> Path:
        """Path to topic modeling preprocessing artifacts"""
        return self.topic_modeling_path / "preprocessing"

    @property
    def sentiment_models_path(self) -> Path:
        """Path to sentiment analysis models"""
        return self.models_base_dir / "sentiment"

    @property
    def sentiment_utils_path(self) -> Path:
        """Path to sentiment analysis utils (kamus, stopwords)"""
        return self.sentiment_models_path / "utils"

    @property
    def emotion_models_path(self) -> Path:
        """Path to emotion classification models"""
        return self.models_base_dir / "emotion"

    @property
    def emotion_utils_path(self) -> Path:
        """Path to emotion classification utils"""
        return self.emotion_models_path / "utils"


settings = Settings()
