"""Centralized Model Engine Manager for hot-reloading ML models."""

from app.core.logger import log

class ModelEngineManager:
    """Central manager untuk semua ML model engines dalam aplikasi."""
    
    @staticmethod
    def reload_all_models() -> dict:
        """
        Reload semua ML models (Sentiment & Emotion) tanpa restart server.
        
        Returns:
            dict: Status reload untuk setiap model engine
        """
        log.info("🔄 Starting manual model reload...")
        results = {}
        
        # Reload Sentiment Models
        try:
            from app.domains.sentiment.engine import SentimentModelManager
            sentiment_manager = SentimentModelManager()
            sentiment_manager.reload()
            results["sentiment"] = "success"
            log.info("✅ Sentiment models reloaded successfully")
        except Exception as e:
            log.error(f"❌ Failed to reload sentiment models: {e}")
            results["sentiment"] = f"failed: {str(e)}"
        
        # Reload Emotion Models
        try:
            from app.domains.emotion.engine import EmotionModelManager
            emotion_manager = EmotionModelManager()
            emotion_manager.reload()
            results["emotion"] = "success"
            log.info("✅ Emotion models reloaded successfully")
        except Exception as e:
            log.error(f"❌ Failed to reload emotion models: {e}")
            results["emotion"] = f"failed: {str(e)}"
        
        log.info("🎯 Model reload completed")
        return results
    
    @staticmethod
    def get_models_status() -> dict:
        """
        Mendapatkan status semua loaded models.
        
        Returns:
            dict: Status untuk setiap model engine
        """
        status = {}
        
        try:
            from app.domains.sentiment.engine import SentimentModelManager
            sentiment_manager = SentimentModelManager()
            status["sentiment"] = {
                "loaded": sentiment_manager._is_loaded,
                "models": {
                    "cnn": sentiment_manager.cnn_model is not None if sentiment_manager._is_loaded else False,
                    "cnn_lstm": sentiment_manager.cnn_lstm_model is not None if sentiment_manager._is_loaded else False
                }
            }
        except Exception as e:
            status["sentiment"] = {"error": str(e)}
        
        try:
            from app.domains.emotion.engine import EmotionModelManager
            emotion_manager = EmotionModelManager()
            status["emotion"] = {
                "loaded": emotion_manager._is_loaded,
                "models": {
                    "cnn": emotion_manager.cnn_model is not None if emotion_manager._is_loaded else False,
                    "bilstm": emotion_manager.bilstm_model is not None if emotion_manager._is_loaded else False
                }
            }
        except Exception as e:
            status["emotion"] = {"error": str(e)}
        
        return status