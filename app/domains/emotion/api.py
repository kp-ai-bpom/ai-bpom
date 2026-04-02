from fastapi import APIRouter, Depends, HTTPException, status

from app.core.logger import log

from .repositories import EmotionRepository, get_emotion_repository
from .schemas import (
    ClassifyEmotionRequest,
    ClassifyEmotionResponse,
    GetEmotionsByProjectIdResponse,
)
from .services import EmotionService, get_emotion_service

router = APIRouter()


@router.post("/predict", response_model=ClassifyEmotionResponse)
async def predict_emotion(
    request: ClassifyEmotionRequest,
    service: EmotionService = Depends(get_emotion_service),
):
    """
    Classify emotions for documents by project_id.
    Menggunakan model CNN dan BiLSTM yang dimuat via Singleton Engine.
    """
    try:
        result = await service.process_emotion(request.project_id)

        if not result or result["total"] == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tidak ada dokumen yang ditemukan untuk project_id: {request.project_id}. Pastikan Topic Modeling sudah berjalan.",
            )

        return ClassifyEmotionResponse(
            status="success",
            message="Emotion classification completed successfully",
            data=result,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Error in predict_emotion endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal melakukan klasifikasi emosi: {str(e)}",
        )


@router.get("/by-project/{project_id}", response_model=GetEmotionsByProjectIdResponse)
async def get_emotion_by_project(
    project_id: str, repository: EmotionRepository = Depends(get_emotion_repository)
):
    """
    Retrieve saved emotion classification results from the database.
    """
    try:
        result = await repository.get_emotions_by_project_id(project_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data emosi tidak ditemukan untuk project_id: {project_id}",
            )

        # Beanie document list (result) akan otomatis di-serialize oleh Pydantic
        return GetEmotionsByProjectIdResponse(
            status="success",
            message="Emotion data retrieved successfully",
            data={"total": len(result), "documents": result},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Error in get_emotion_by_project endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal mengambil data emosi: {str(e)}",
        )
