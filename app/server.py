from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.logger import log
from app.db.database import init_db
from app.domains.emotion.engine import get_emotion_models
from app.domains.sentiment.engine import get_sentiment_models
from app.shared.llm import init_llm


def create_app() -> FastAPI:
    """
    Application Factory: Merakit dan mengembalikan instance FastAPI.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan Manager untuk startup dan shutdown events."""

        log.info("🚀 Starting up server...")

        try:
            # 1. Inisialisasi Database
            await init_db()

            # 2. Inisialisasi & Pre-warm LLM Singleton
            init_llm()

            # 3. Inisialisasi & Pre-warm ML Models Singleton
            get_sentiment_models().load_models()
            get_emotion_models().load_models()
        except Exception as e:
            log.error(f"❌ Startup failed: {e}")
            raise e  # Hentikan server jika DB/LLM gagal connect

        yield  # Aplikasi berjalan di titik ini

        # -----------------------------------------------------
        # SHUTDOWN EVENT
        # -----------------------------------------------------
        log.info("🛑 Shutting down server...")
        # Lakukan pembersihan (cleanup) memori model AI atau koneksi DB di sini

    # Inisialisasi instance FastAPI
    app = FastAPI(
        title="AI Services API",
        description="API Modular untuk Topic Modeling, Sentiment, SNA, dan Chatbot",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_api_route(
        "/", lambda: {"message": "Welcome to AI Services API!"}, methods=["GET"]
    )
    app.include_router(api_router, prefix="/api")

    return app
