from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.llm import init_llm
from app.core.logger import log
from app.db.database import init_db


def create_app() -> FastAPI:
    """
    Application Factory: Merakit dan mengembalikan instance FastAPI.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan Manager untuk startup dan shutdown events."""

        log.info("🚀 Starting up server...")

        if settings.ENV == "production":
            try:
                await init_db()
                init_llm()
            except Exception as e:
                log.error(f"❌ Startup failed: {e}")
                raise e

        yield

        log.info("🛑 Shutting down server...")
        # Lakukan pembersihan (cleanup) memori model AI atau koneksi DB di sini

    # Inisialisasi instance FastAPI
    app = FastAPI(
        title="AI Service API BPOM",
        description="AI Service API for Chatbot, Pemetaan Suksesor, and Penilaian Suksesor",
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
