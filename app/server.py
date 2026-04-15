from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.llm import init_llm
from app.core.logger import log


def create_app() -> FastAPI:
    """
    Application Factory: Merakit dan mengembalikan instance FastAPI.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan Manager untuk startup dan shutdown events."""

        log.info("🚀 Starting up server...")

        # Initialize LLM
        init_llm()

        yield

        log.info("🛑 Shutting down server...")

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
