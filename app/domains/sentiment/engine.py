import os
import pickle

from keras.models import load_model

from app.core.config import settings
from app.core.logger import log


class SentimentModelManager:
    """
    Singleton Engine untuk memastikan model Keras (.h5) dan tokenizer (.pickle)
    hanya dimuat SATU KALI ke dalam RAM saat aplikasi berjalan.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentModelManager, cls).__new__(cls)
            cls._instance._is_loaded = False
        return cls._instance

    def load_models(self):
        # Lazy Loading: Hanya load jika belum pernah di-load
        if not self._is_loaded:
            log.info("🧠 Loading models...")

            # Setup Paths dari Config
            models_path = str(settings.sentiment_models_path)

            # Load CNN
            self.cnn_model, self.cnn_tokenizer = self._load_keras_model(
                models_path,
                settings.MODEL_CNN_SENTIMENT_FILENAME,
                settings.TOKENIZER_CNN_SENTIMENT_FILENAME,
            )

            # Load CNN-LSTM
            self.cnn_lstm_model, self.cnn_lstm_tokenizer = self._load_keras_model(
                models_path,
                settings.MODEL_CNN_LSTM_SENTIMENT_FILENAME,
                settings.TOKENIZER_CNN_LSTM_SENTIMENT_FILENAME,
            )

            self._is_loaded = True
            log.info("✅ All Models loaded successfully.")

    def reload(self):
        """
        Force reload semua models. Berguna ketika ada update model baru.
        """
        log.info("🔄 Reloading sentiment models...")
        self._is_loaded = False
        self.load_models()
        log.info("✅ Sentiment models reloaded")

    def _load_keras_model(self, base_path, model_file, tokenizer_file):
        try:
            m_path = os.path.join(base_path, model_file)
            t_path = os.path.join(base_path, tokenizer_file)

            if not os.path.exists(m_path) or not os.path.exists(t_path):
                log.error(f"Model or tokenizer not found: {m_path}")
                return None, None

            model = load_model(m_path)
            with open(t_path, "rb") as f:
                tokenizer = pickle.load(f)
            log.info(f"Model Loaded Successfully: {model_file}")
            return model, tokenizer
        except Exception as e:
            log.error(f"Failed Load Model {model_file}: {e}")
            return None, None


def get_sentiment_models() -> SentimentModelManager:
    """
    Dependency Factory untuk menginjeksi instance SentimentModelManager.
    """
    manager = SentimentModelManager()
    manager.load_models()
    return manager
