from fastapi import APIRouter, Depends, HTTPException, status

from app.core.logger import log

from .repositories import SentimentRepository, get_sentiment_repository
from .schemas import (
    ClassifySentimentRequest,
    ClassifySentimentResponse,
    GetSentimentsByProjectIdResponse,
)
from .services import SentimentService, get_sentiment_service

router = APIRouter()


@router.post("/predict", response_model=ClassifySentimentResponse)
async def predict_sentiment(
    request: ClassifySentimentRequest,
    service: SentimentService = Depends(get_sentiment_service),
):
    """
    Classify sentiment for documents by project_id.
    Mengambil data dari Topic Modeling, memproses teks, dan menjalankan inferensi CNN/LSTM.
    """
    try:
        results = await service.process_sentiment(request.project_id)

        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tidak ada dokumen yang ditemukan untuk Project ID: {request.project_id}. Pastikan Topic Modeling sudah dijalankan.",
            )

        return ClassifySentimentResponse(
            status="success",
            message="Sentiment classification completed",
            data=results,  # type: ignore[arg-type]
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Error in predict_sentiment endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal melakukan prediksi sentimen: {str(e)}",
        )


@router.get("/by-project/{project_id}", response_model=GetSentimentsByProjectIdResponse)
async def get_sentiments_by_project_id(
    project_id: str,
    repository: SentimentRepository = Depends(get_sentiment_repository),
):
    """Get all computed sentiments by project ID"""
    try:
        sentiments = await repository.get_sentiments_by_project_id(project_id)

        if not sentiments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Sentiments not found for the given project ID",
            )

        return GetSentimentsByProjectIdResponse(
            status="success",
            message="Sentiments retrieved successfully",
            data=sentiments,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Error retrieving sentiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Terjadi kesalahan pada server saat mengambil data.",
        )
