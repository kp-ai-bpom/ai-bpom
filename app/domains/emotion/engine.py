import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
from keras.models import load_model

from app.core.config import settings
from app.core.logger import log


class EmotionModelManager:
    """Singleton Engine untuk model Emotion (CNN & BiLSTM)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._is_loaded = False
        return cls._instance

    def load_models(self):
        if not self._is_loaded:
            log.info("🧠 Memuat model Emotion CNN & BiLSTM ke Memory (Singleton)...")

            # Path dari Config
            models_path = str(settings.emotion_models_path)

            # Load CNN (Sesuaikan nama file asli Anda jika berbeda)
            self.cnn_model, self.cnn_tokenizer = self._load_keras_model(
                models_path,
                settings.MODEL_CNN_EMOTION_FILENAME,
                settings.TOKENIZER_CNN_EMOTION_FILENAME,
            )

            # Load BiLSTM (Sesuaikan nama file asli Anda jika berbeda)
            self.bilstm_model, self.bilstm_tokenizer = self._load_keras_model(
                models_path,
                settings.MODEL_BILSTM_EMOTION_FILENAME,
                settings.TOKENIZER_BILSTM_EMOTION_FILENAME,
            )

            # Label Map (Sesuaikan urutan array output model Anda)
            self.emotion_labels = ["Anger", "Fear", "Joy", "Love", "Sad", "Neutral"]

            self._is_loaded = True
            log.info("✅ Semua model ML Emotion berhasil dimuat.")

    def reload(self):
        """
        Force reload semua models. Berguna ketika ada update model baru.
        """
        log.info("🔄 Reloading emotion models...")
        self._is_loaded = False
        self.load_models()
        log.info("✅ Emotion models reloaded")

    def _load_keras_model(self, base_path, model_file, tokenizer_file):
        try:
            m_path = os.path.join(base_path, model_file)
            t_path = os.path.join(base_path, tokenizer_file)

            if not os.path.exists(m_path) or not os.path.exists(t_path):
                log.error(f"File model tidak ditemukan di: {m_path}")
                return None, None

            model = load_model(m_path)
            with open(t_path, "rb") as f:
                tokenizer = pickle.load(f)
            return model, tokenizer
        except Exception as e:
            log.error(f"Gagal memuat model Emotion {model_file}: {e}")
            return None, None


def get_emotion_models() -> EmotionModelManager:
    manager = EmotionModelManager()
    manager.load_models()
    return manager
