"""FastAPI router for penilaian_makalah."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.domains.penilaian_makalah.constant import DEFAULT_M_SAMPLES, DEFAULT_QUERY_MODE, DEFAULT_TEMPERATURE, SUPPORTED_QUERY_MODE
from app.domains.penilaian_makalah.core.config import settings as domain_settings
from app.domains.penilaian_makalah.schemas import EvaluateRequest, EvaluateResponse, ExportResponse, HistoryResponse, UploadKnowledgeResponse, UploadPaperResponse
from app.domains.penilaian_makalah.service import PenilaianMakalahService, get_penilaian_makalah_service

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "bucket_makalah": domain_settings.bucket_makalah,
        "bucket_tema": domain_settings.bucket_tema,
        "bucket_knowledge": domain_settings.bucket_knowledge,
    }


@router.get("/modes")
async def supported_modes(service: PenilaianMakalahService = Depends(get_penilaian_makalah_service)) -> dict:
    return {"data": service.get_supported_query_modes()}


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(100, ge=1, le=500),
    service: PenilaianMakalahService = Depends(get_penilaian_makalah_service),
) -> HistoryResponse:
    data = await service.list_history(limit=limit)
    return HistoryResponse(total=len(data), data=data)


@router.get("/history/{history_id}")
async def get_history_item(
    history_id: int,
    service: PenilaianMakalahService = Depends(get_penilaian_makalah_service),
) -> dict:
    return await service.get_history_item(history_id)


@router.get("/tema")
async def list_tema_files(service: PenilaianMakalahService = Depends(get_penilaian_makalah_service)) -> dict:
    return {"data": await service.get_minio_files(domain_settings.bucket_tema)}


@router.get("/papers")
async def list_papers(service: PenilaianMakalahService = Depends(get_penilaian_makalah_service)) -> dict:
    return {"data": await service.get_minio_files(domain_settings.bucket_makalah, detailed=True)}


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_from_minio(
    request: EvaluateRequest,
    service: PenilaianMakalahService = Depends(get_penilaian_makalah_service),
) -> EvaluateResponse:
    result = await service.evaluate_from_minio(
        jabatan=request.jabatan,
        paper_filename=request.paper_filename,
        tema_filename=request.tema_filename,
        query_mode=request.query_mode,
        m_samples=request.m_samples,
        temperature=request.temperature,
        save_result=True,
    )
    return EvaluateResponse(
        ringkasan=result.get("Ringkasan", ""),
        final_score=result["final_score"],
        scores=result["scores"],
        justification=result["justification"],
        evidence=result["evidence"],
        uncertainty=result.get("uncertainty_metrics", {}),
        valid_samples=result.get("valid_samples", 0),
        total_samples=result.get("total_samples", 0),
    )


@router.post("/evaluate/upload", response_model=EvaluateResponse)
async def evaluate_uploaded_file(
    jabatan: str = Form(...),
    tema_filename: str = Form(...),
    query_mode: str = Form(DEFAULT_QUERY_MODE),
    m_samples: int = Form(DEFAULT_M_SAMPLES),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    paper: UploadFile = File(...),
    service: PenilaianMakalahService = Depends(get_penilaian_makalah_service),
) -> EvaluateResponse:
    uploaded_data = await paper.read()
    result = await service.evaluate_uploaded_file(
        jabatan=jabatan,
        tema_filename=tema_filename,
        uploaded_filename=paper.filename,
        uploaded_data=uploaded_data,
        query_mode=query_mode,
        m_samples=m_samples,
        temperature=temperature,
        save_result=True,
    )
    return EvaluateResponse(
        ringkasan=result.get("Ringkasan", ""),
        final_score=result["final_score"],
        scores=result["scores"],
        justification=result["justification"],
        evidence=result["evidence"],
        uncertainty=result.get("uncertainty_metrics", {}),
        valid_samples=result.get("valid_samples", 0),
        total_samples=result.get("total_samples", 0),
    )


@router.post("/evaluate/raw")
async def evaluate_raw(
    request: EvaluateRequest,
    makalah_text: str = Form(...),
    tema_text: str = Form(...),
    service: PenilaianMakalahService = Depends(get_penilaian_makalah_service),
) -> dict:
    return await service.evaluate_from_text(
        paper_filename=request.paper_filename,
        jabatan=request.jabatan,
        makalah_text=makalah_text,
        tema_text=tema_text,
        query_mode=request.query_mode,
        m_samples=request.m_samples,
        temperature=request.temperature,
        save_result=True,
    )


@router.post("/upload/paper", response_model=UploadPaperResponse)
async def upload_paper(
    file: UploadFile = File(...),
    service: PenilaianMakalahService = Depends(get_penilaian_makalah_service),
) -> UploadPaperResponse:
    data = await file.read()
    uploaded = await service.upload_document(domain_settings.bucket_makalah, file.filename, data)
    return UploadPaperResponse(filename=file.filename, uploaded=uploaded, message="Paper uploaded" if uploaded else "Upload failed")


@router.post("/upload/knowledge", response_model=UploadKnowledgeResponse)
async def upload_knowledge(
    file: UploadFile = File(...),
    service: PenilaianMakalahService = Depends(get_penilaian_makalah_service),
) -> UploadKnowledgeResponse:
    data = await file.read()
    uploaded = await service.upload_document(domain_settings.bucket_knowledge, file.filename, data)
    return UploadKnowledgeResponse(filename=file.filename, ingested=uploaded, message="Knowledge uploaded" if uploaded else "Upload failed")
