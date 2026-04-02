from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.core.engine import ModelEngineManager
from app.core.logger import log

router = APIRouter()


class ReloadResponse(BaseModel):
    """Response model untuk reload models endpoint."""

    status: str = Field(..., description="Status keseluruhan reload operation")
    message: str = Field(..., description="Pesan deskriptif")
    results: dict = Field(
        ..., description="Detail hasil reload untuk setiap model engine"
    )


class ModelsStatusResponse(BaseModel):
    """Response model untuk status models endpoint."""

    status: str = Field(..., description="Status keseluruhan")
    models: dict = Field(..., description="Status detail untuk setiap model engine")


@router.post(
    "/reload-models",
    response_model=ReloadResponse,
    status_code=status.HTTP_200_OK,
    summary="Reload All ML Models",
    description="""
    Reload semua ML models (Sentiment & Emotion) tanpa perlu restart server.

    **Use Case:**
    - Setelah update/training model baru
    - Troubleshooting jika model corrupt
    - Hot-reload untuk production deployment

    **Warning:**
    - Proses reload bisa memakan waktu 5-15 detik
    - Model lama akan di-unload dari memory
    - Request inference yang sedang berjalan bisa gagal
    """,
)
async def reload_models():
    """
    Endpoint untuk manual reload semua ML models.

    Returns:
        ReloadResponse: Status hasil reload untuk setiap model engine
    """
    log.info("📞 API Call: /system/reload-models")

    try:
        results = ModelEngineManager.reload_all_models()

        # Check if all reloads successful
        all_success = all(v == "success" for v in results.values())

        if all_success:
            return ReloadResponse(
                status="success",
                message="All models reloaded successfully",
                results=results,
            )
        else:
            return ReloadResponse(
                status="partial",
                message="Some models failed to reload. Check results for details.",
                results=results,
            )

    except Exception as e:
        log.exception(f"Error during model reload: {e}")
        return ReloadResponse(
            status="error", message=f"Failed to reload models: {str(e)}", results={}
        )


@router.get(
    "/models-status",
    response_model=ModelsStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Models Loading Status",
    description="""
    Mendapatkan status loading untuk semua ML models dalam aplikasi.

    **Response Info:**
    - `loaded`: True jika model sudah di-load ke memory
    - `models`: Detail status untuk setiap model type (CNN, LSTM, etc.)
    """,
)
async def get_models_status():
    """
    Endpoint untuk mengecek status semua ML models.

    Returns:
        ModelsStatusResponse: Status detail untuk setiap model engine
    """
    log.info("📞 API Call: /system/models-status")

    try:
        models_status = ModelEngineManager.get_models_status()

        return ModelsStatusResponse(status="success", models=models_status)

    except Exception as e:
        log.exception(f"Error getting models status: {e}")
        return ModelsStatusResponse(status="error", models={"error": str(e)})


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Simple health check endpoint untuk monitoring",
)
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "service": "SociaLabs AI Backend",
        "message": "Service is running",
    }
