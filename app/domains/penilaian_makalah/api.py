from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import os
import threading
import asyncio

from app.db.database import get_db
from app.core.config import Settings

settings = Settings()


from .schemas import PenilaianRequest, PenilaianResponse, IngestResponse
from .services import PenilaianService
from .repositories import MinioRepository, EvaluationRepository, IngestionRepository

SQLALCHEMY_DATABASE_URL = f"postgresql://{settings.PG_USER}:{settings.PG_PASS}@{settings.PG_HOST}:{settings.PG_PORT}/{settings.PG_DB}"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── 2. Konfigurasi LightRAG Standalone ───────────────────────────────────────
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import wrap_embedding_func_with_attrs

_rag_instance = None

async def _initialize_rag():
    WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "/rag_storage")
    
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
        return await openai_complete_if_cache(
            os.getenv("LLM_MODEL"),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=os.getenv("LLM_BINDING_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST"),
            **kwargs,
        )
        
    embedding_dim = int(os.getenv("EMBEDDING_DIM", 1536))
    token_limit   = int(os.getenv("EMBEDDING_TOKEN_LIMIT", 8192))
    model_name    = os.getenv("EMBEDDING_MODEL")

    async def raw_embedding_func(texts):
        return await openai_embed.func(
            texts,
            api_key=os.getenv("LLM_BINDING_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST"),
            model=model_name,
        )

    embedding_func = wrap_embedding_func_with_attrs(
        embedding_dim=embedding_dim,
        max_token_size=token_limit,
        model_name=model_name,
    )(raw_embedding_func)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="Neo4JStorage",
        enable_llm_cache=False, 
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()
    return rag

async def get_rag():
    """Dependencies RAG yang diinisialisasi secara singleton menggunakan FastAPI loop."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = await _initialize_rag()
    return _rag_instance


# ── 3. API Router & Endpoints ────────────────────────────────────────────────
router = APIRouter()

def get_penilaian_service(db: Session = Depends(get_db), rag = Depends(get_rag)) -> PenilaianService:
    minio_repo = MinioRepository()
    db_repo = EvaluationRepository(db)
    ingest_repo = IngestionRepository(db)
    return PenilaianService(db_repo, minio_repo, rag, ingest_repo)

@router.get("/jabatan", response_model=List[str])
async def get_jabatan_list(service: PenilaianService = Depends(get_penilaian_service)):
    """Mengembalikan daftar jabatan yang tersedia berdasarkan file SKJ di MinIO"""
    return service.get_available_jabatan()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_skj_documents(service: PenilaianService = Depends(get_penilaian_service)):
    """Menelan (ingest) dokumen SKJ dari MinIO ke dalam LightRAG"""
    try:
        response = await service.ingest_skj_documents()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tema", response_model=List[str])
async def get_tema_list(service: PenilaianService = Depends(get_penilaian_service)):
    """Mengembalikan daftar file ketentuan tema dari MinIO"""
    return service.minio_repo.list_files(service.minio_repo.BUCKET_TEMA)

@router.get("/makalah", response_model=List[dict])
async def get_makalah_list(service: PenilaianService = Depends(get_penilaian_service)):
    """Mengembalikan daftar file makalah dari MinIO secara detail"""
    return service.minio_repo.list_files_detailed(service.minio_repo.BUCKET_MAKALAH)

@router.get("/makalah/{filename}/text")
async def get_makalah_text(filename: str, service: PenilaianService = Depends(get_penilaian_service)):
    """Mengambil dan mengekstrak teks dari makalah di MinIO"""
    data = service.minio_repo.download_file(service.minio_repo.BUCKET_MAKALAH, filename)
    if not data:
        raise HTTPException(status_code=404, detail="File makalah tidak ditemukan di MinIO")
    from .services import DocumentExtractor
    text = DocumentExtractor.extract_from_bytes(data, filename)
    return {"text": text}

@router.post("/evaluate", response_model=PenilaianResponse)
async def evaluate_makalah(request: PenilaianRequest, service: PenilaianService = Depends(get_penilaian_service)):
    """Menjalankan proses evaluasi makalah dengan LLM"""
    try:
        response = await service.process_evaluation(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history(
    limit: int = 100,
    service: PenilaianService = Depends(get_penilaian_service)
):
    history = await service.db_repo.get_history(limit=limit)
    return history

@router.post("/upload/{kategori}")
async def upload_document(kategori: str, file: UploadFile = File(...), service: PenilaianService = Depends(get_penilaian_service)):
    """
    Mengunggah dokumen ke MinIO.
    Kategori yang didukung: 'makalah', 'tema', 'skj'
    """
    if kategori == "makalah":
        bucket = service.minio_repo.BUCKET_MAKALAH
    elif kategori == "tema":
        bucket = service.minio_repo.BUCKET_TEMA
    elif kategori == "skj":
        bucket = service.minio_repo.BUCKET_SKJ
    else:
        raise HTTPException(status_code=400, detail=f"Kategori '{kategori}' tidak valid. Gunakan: makalah, tema, atau skj.")
    
    content = await file.read()
    success = service.upload_document(bucket, file.filename, content)
    
    if not success:
        raise HTTPException(status_code=500, detail="Gagal mengunggah file ke MinIO")
        
    return {"message": "Berhasil", "filename": file.filename, "bucket": bucket}
