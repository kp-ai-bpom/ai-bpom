from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.logger import log

from .repositories import TopicRepository, get_topic_repository
from .schemas import (
    AnalyzeTopicRequest,
    AnalyzeTopicResponse,
    GetDocumentsByProjectIdResponse,
    GetTopicsByProjectIdResponse,
)
from .services import TopicModelingService, get_topic_modeling_service

router = APIRouter()


@router.post("/topic-modelling", response_model=AnalyzeTopicResponse)
async def topic_modelling(
    request: AnalyzeTopicRequest,
    service: TopicModelingService = Depends(get_topic_modeling_service),
):
    """
    Run complete topic modelling pipeline.

    Proses ini mencakup:
    1. Pengambilan data tweet dari database NestJS.
    2. Augmentasi data (rephrase/terjemahan) via LLM.
    3. 10-Step NLP Preprocessing.
    4. Model Training ETM & Evaluasi Coherence.
    5. Penyimpanan hasil (Topics & Documents) ke MongoDB Beanie.
    """
    try:
        topic_data = {
            "project_id": request.project_id,
            "keyword": request.keyword,
            "start_date": request.start_date,
            "end_date": request.end_date,
        }

        # Panggil service yang sudah di-inject
        result = await service.process_topic_modeling(topic_data)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found for the given parameters. Pastikan tweet ada di rentang tanggal tersebut.",
            )

        return AnalyzeTopicResponse(
            status="success",
            message="Topic modelling completed successfully",
            data=result,
        )

    except HTTPException:
        raise  # Rethrow HTTPException agar status code tidak berubah jadi 500
    except Exception as e:
        log.exception(f"Error in topic_modelling endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan saat melakukan topic analysis: {str(e)}",
        )


@router.get(
    "/topics-by-project/{project_id}", response_model=GetTopicsByProjectIdResponse
)
async def get_topics_by_project(
    project_id: str,
    # Karena endpoint ini murni HANYA mengambil data tanpa logika AI (LLM/ETM),
    # kita bisa langsung menginjeksi Repository untuk menghindari overhead Service.
    repository: TopicRepository = Depends(get_topic_repository),
):
    """
    Retrieve generated topic contexts by project ID.
    """
    try:
        topics = await repository.get_topics_by_project_id(project_id)

        if not topics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Topics not found for project ID: {project_id}",
            )

        return GetTopicsByProjectIdResponse(
            status="success",
            message="Topics retrieved successfully",
            data=topics,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Error in get_topics_by_project endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving topics: {str(e)}",
        )


@router.get(
    "/documents-by-project/{project_id}", response_model=GetDocumentsByProjectIdResponse
)
async def get_documents_by_project(
    project_id: str,
    topic_id: Optional[
        int
    ] = None,  # Contoh fitur tambahan: Filter opsional berdasarkan ID topik
    repository: TopicRepository = Depends(get_topic_repository),
):
    """
    Retrieve document data (tweet yang sudah di-assign topik) by project ID.
    Opsional: Filter berdasarkan topic_id tertentu.
    """
    try:
        if topic_id is not None:
            # Panggil fungsi repo jika user ingin filter by topic_id (Sudah ada di repo Anda)
            documents = await repository.get_documents_by_topic(project_id, topic_id)
        else:
            # Jika Anda belum punya fungsi ini di Repo, tambahkan:
            # return await DocumentsModel.find(DocumentsModel.projectId == project_id).to_list()
            # Asumsi: Anda akan menambahkan `get_documents_by_project_id` ke repository.py

            # WORKAROUND SEMENTARA: Akses model langsung jika fungsi repo belum dibuat
            from .models import DocumentsModel

            documents = await DocumentsModel.find(
                DocumentsModel.projectId == project_id
            ).to_list()

        if not documents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Documents not found for project ID: {project_id}",
            )

        return GetDocumentsByProjectIdResponse(
            status="success",
            message="Documents retrieved successfully",
            data=documents,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Error in get_documents_by_project endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving documents: {str(e)}",
        )
