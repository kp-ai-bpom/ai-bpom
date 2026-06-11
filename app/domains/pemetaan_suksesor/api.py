import asyncio
import logging
import os
import shutil
import tempfile
from enum import Enum
from typing import Callable, List, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from .dto.pipeline import AgentEvaluateRequest, AgentName, AgentRunRequest
from .dto.request import (
    IngestRequest,
    KandidatSIASN,
    SaveMatchingRequest,
    SimulasiRequest,
    SuksesorCreateRequest,
    SuksesorUpdateRequest,
)
from .dto.response import (
    AgentResponse,
    IngestLogDetailResponse,
    IngestLogListResponse,
    IngestResponse,
    JobAcceptedResponse,
    JobStatusResponse,
    KandidatListResponse,
    MatchingHistoryDetailResponse,
    MatchingHistoryListResponse,
    MatchingHistorySaveResponse,
    NineBoxResponse,
    SimulasiResponse,
    SuksesorDeleteResponse,
    SuksesorListResponse,
    SuksesorResponse,
    UploadResponse,
)
from .services import (
    AgentPipelineService,
    IngestionService,
    JobService,
    MatchingHistoryService,
    PipelineService,
    SimulationService,
    SuksesorService,
    get_agent_pipeline_service,
    get_ingestion_service,
    get_job_service,
    get_matching_history_service,
    get_pipeline_service,
    get_simulation_service,
    get_suksesor_service,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class KnowledgeGraphType(str, Enum):
    jabatan = "jabatan"
    regulasi = "regulasi"


# Allowed file extensions per pipeline type
_KNOWLEDGE_GRAPH_JABATAN_EXTS = {".xlsx"}
_KNOWLEDGE_GRAPH_REGULASI_EXTS = {".pdf"}
_VECTORRAG_JABATAN_EXTS = {".xlsx"}
_PROFIL_JABATAN_EXTS = {".xlsx"}


def _validate_file_extensions(files: List[UploadFile], allowed_exts: set) -> None:
    """Validate that all uploaded files have allowed extensions."""
    for f in files:
        if not f.filename:
            raise HTTPException(status_code=400, detail="All files must have a filename")
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}' has unsupported extension '{ext}'. "
                       f"Allowed: {', '.join(sorted(allowed_exts))}",
            )


async def _run_and_cleanup(
    func: Callable, job_id: str, file_paths: List[str], temp_dir: str, *args
):
    try:
        await func(job_id, file_paths, *args)
    finally:
        await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)


async def _save_uploads_to_temp(files: List[UploadFile]) -> tuple[List[str], str]:
    """Save UploadFile contents to a temp directory synchronously.

    Returns (file_paths, temp_dir). Caller is responsible for cleaning up temp_dir.
    """
    temp_dir = tempfile.mkdtemp()
    file_paths: List[str] = []
    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            file_paths.append(file_path)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return file_paths, temp_dir


async def _start_pipeline_job(
    background_tasks: BackgroundTasks,
    job_service: JobService,
    job_name: str,
    pipeline_func: Callable,
    files: List[UploadFile],
    allowed_exts: set,
) -> JobAcceptedResponse:
    """Helper to validate, save, create job, and schedule background pipeline task."""
    _validate_file_extensions(files, allowed_exts)

    file_paths, temp_dir = await _save_uploads_to_temp(files)

    job_id = job_service.create_job(job_name)
    background_tasks.add_task(
        _run_and_cleanup, pipeline_func, job_id, file_paths, temp_dir
    )
    return JobAcceptedResponse(
        job_id=job_id,
        message=f"{job_name.replace('_', ' ').title()} pipeline started",
    )


# ── CRUD Endpoints ────────────────────────────────────────────────


@router.post(
    "/",
    response_model=SuksesorResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new Suksesor",
    description="Create a new Suksesor (calon penerus jabatan) entry.",
)
async def create_suksesor(
    data: SuksesorCreateRequest,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorResponse:
    """Create a new Suksesor."""
    return await service.create(data)


@router.get(
    "/",
    response_model=SuksesorListResponse,
    summary="Get list of Suksesor",
    description="Get paginated list of Suksesor with optional filters.",
)
async def list_suksesor(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by nama or nip"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorListResponse:
    """Get paginated list of Suksesor."""
    return await service.get_list(
        page=page, page_size=page_size, search=search, is_active=is_active
    )


@router.get(
    "/nip/{nip}",
    response_model=SuksesorResponse,
    summary="Get Suksesor by NIP",
    description="Get a specific Suksesor by their NIP (Nomor Induk Pegawai).",
)
async def get_suksesor_by_nip(
    nip: str,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorResponse:
    """Get a Suksesor by NIP."""
    return await service.get_by_nip(nip)


@router.get(
    "/{suksesor_id}",
    response_model=SuksesorResponse,
    summary="Get Suksesor by ID",
    description="Get a specific Suksesor by their UUID.",
)
async def get_suksesor_by_id(
    suksesor_id: UUID,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorResponse:
    """Get a Suksesor by ID."""
    return await service.get_by_id(suksesor_id)


@router.put(
    "/{suksesor_id}",
    response_model=SuksesorResponse,
    summary="Update a Suksesor",
    description="Update an existing Suksesor by ID.",
)
async def update_suksesor(
    suksesor_id: UUID,
    data: SuksesorUpdateRequest,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorResponse:
    """Update a Suksesor."""
    return await service.update(suksesor_id, data)


@router.delete(
    "/{suksesor_id}",
    response_model=SuksesorDeleteResponse,
    summary="Delete a Suksesor",
    description="Delete a Suksesor by ID (hard delete).",
)
async def delete_suksesor(
    suksesor_id: UUID,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorDeleteResponse:
    """Delete a Suksesor."""
    return await service.delete(suksesor_id)


@router.patch(
    "/{suksesor_id}/deactivate",
    response_model=SuksesorResponse,
    summary="Soft delete a Suksesor",
    description="Deactivate a Suksesor by setting is_active to False.",
)
async def deactivate_suksesor(
    suksesor_id: UUID,
    service: SuksesorService = Depends(get_suksesor_service),
) -> SuksesorResponse:
    """Soft delete (deactivate) a Suksesor."""
    return await service.soft_delete(suksesor_id)


# ── Match (Simulation) Endpoints ─────────────────────────────────


@router.get(
    "/match/jabatan",
    summary="Daftar Jabatan Target Tersedia",
    description=(
        "Mengembalikan daftar jabatan target yang tersedia untuk simulasi "
        "pemetaan suksesor. Gunakan nama jabatan dari daftar ini sebagai "
        "parameter target_jabatan di endpoint match."
    ),
)
async def list_jabatan_target(
    service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Get all available target positions for simulation."""
    jabatan_list = service.list_available_jabatan()
    return {
        "message": "Daftar jabatan target tersedia",
        "data": jabatan_list,
    }


@router.get(
    "/match/nine-box",
    response_model=NineBoxResponse,
    summary="Data Nine-Box Talenta Grid",
    description=(
        "Mengembalikan data grid nine-box talenta lengkap dengan jumlah "
        "kandidat per box dan daftar nama kandidat. Frontend menggunakan "
        "data ini untuk merender grid 3x3 dan tooltip hover."
    ),
)
async def get_nine_box_data(
    service: SimulationService = Depends(get_simulation_service),
) -> NineBoxResponse:
    """Get nine-box talenta grid data with candidate counts."""
    return service.get_nine_box_data()


@router.get(
    "/match/kandidat",
    response_model=KandidatListResponse,
    summary="Daftar Kandidat Berdasarkan Box Talenta",
    description=(
        "Mengembalikan daftar kandidat yang berada di box-box talenta terpilih. "
        "Gunakan parameter boxes untuk memfilter (misal: boxes=7,8,9). "
        "Data dikembalikan lengkap dengan ringkasan untuk kartu kandidat UI."
    ),
)
async def get_kandidat_by_boxes(
    boxes: str = Query(
        "7,8,9",
        description="Kotak talenta yang dipilih, pisahkan dengan koma (e.g. 7,8,9)",
    ),
    service: SimulationService = Depends(get_simulation_service),
) -> KandidatListResponse:
    """Get candidates filtered by selected nine-box positions."""
    box_numbers = [int(b.strip()) for b in boxes.split(",") if b.strip().isdigit()]
    return service.get_kandidat_by_boxes(box_numbers)


@router.post(
    "/match",
    response_model=SimulasiResponse,
    summary="Simulasi Pemetaan Suksesor",
    description=(
        "Menjalankan simulasi pemetaan suksesor menggunakan multi-agent pipeline. "
        "Menerima daftar kandidat (atau gunakan data sampel) dan mengembalikan "
        "top 5 kandidat paling cocok berdasarkan evaluasi multi-tahap: "
        "Decomposition → Retrieval & Extraction → Validation (L-Eval + C-Eval) → Scoring."
    ),
)
async def simulasi_pemetaan_suksesor(
    request: SimulasiRequest,
    service: SimulationService = Depends(get_simulation_service),
) -> SimulasiResponse:
    """Run multi-agent simulation for succession mapping."""
    return await service.run(
        target_jabatan=request.target_jabatan,
        kandidat_list=request.kandidat,
        top_n=5,
    )


@router.post(
    "/match/sampel",
    response_model=SimulasiResponse,
    summary="Simulasi Pemetaan Suksesor (Data Sampel)",
    description=(
        "Menjalankan simulasi pemetaan suksesor menggunakan 10 data kandidat sampel "
        "dari candidates.json. Tidak perlu mengirim data kandidat — cukup sebutkan "
        "jabatan target. Mengembalikan top 5 kandidat paling cocok."
    ),
)
async def simulasi_pemetaan_suksesor_sampel(
    target_jabatan: str = Query(
        "Inspektur I",
        description="Jabatan target suksesi",
    ),
    service: SimulationService = Depends(get_simulation_service),
) -> SimulasiResponse:
    """Run simulation using built-in sample candidate data (10 candidates)."""
    raw_candidates = service.load_candidates()
    kandidat_list = [KandidatSIASN.model_validate(c) for c in raw_candidates]

    return await service.run(
        target_jabatan=target_jabatan,
        kandidat_list=kandidat_list,
        top_n=5,
    )


# ── Matching History Endpoints ─────────────────────────────────────


@router.post(
    "/match/history",
    response_model=MatchingHistorySaveResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Simpan Hasil Matching ke Riwayat",
    description=(
        "Menyimpan hasil simulasi matching ke database untuk referensi di kemudian hari. "
        "Kirim data hasil dari endpoint /match atau /match/sampel."
    ),
)
async def save_matching_history(
    data: SaveMatchingRequest,
    service: MatchingHistoryService = Depends(get_matching_history_service),
) -> MatchingHistorySaveResponse:
    """Save a matching simulation result to history."""
    return await service.save(data)


@router.get(
    "/match/history",
    response_model=MatchingHistoryListResponse,
    summary="Daftar Riwayat Matching",
    description=(
        "Mengembalikan daftar riwayat matching yang tersimpan (data ringkas). "
        "Diurutkan dari yang terbaru."
    ),
)
async def list_matching_history(
    page: int = Query(1, ge=1, description="Nomor halaman"),
    page_size: int = Query(10, ge=1, le=100, description="Jumlah item per halaman"),
    service: MatchingHistoryService = Depends(get_matching_history_service),
) -> MatchingHistoryListResponse:
    """Get paginated list of matching history (summary only)."""
    return await service.get_list(page=page, page_size=page_size)


@router.get(
    "/match/history/{history_id}",
    response_model=MatchingHistoryDetailResponse,
    summary="Detail Riwayat Matching",
    description="Mengembalikan detail lengkap satu riwayat matching berdasarkan ID.",
)
async def get_matching_history_detail(
    history_id: UUID,
    service: MatchingHistoryService = Depends(get_matching_history_service),
) -> MatchingHistoryDetailResponse:
    """Get full detail of a matching history record by ID."""
    return await service.get_by_id(history_id)


# ── Ingestion Endpoints ─────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trigger smart ingestion from MinIO",
)
async def trigger_ingestion(
    request: IngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
):
    """Trigger smart ingestion of documents from MinIO bucket."""
    return await service.ingest(
        document_names=request.document_names,
        force_reingest=request.force_reingest,
    )


@router.get(
    "/ingest/status",
    response_model=IngestLogListResponse,
    summary="List ingestion logs",
)
async def list_ingestion_logs(
    offset: int = 0,
    limit: int = 50,
    service: IngestionService = Depends(get_ingestion_service),
):
    """List all ingestion log entries."""
    return await service.list_logs(offset=offset, limit=limit)


@router.get(
    "/ingest/{log_id}",
    response_model=IngestLogDetailResponse,
    summary="Get ingestion log detail",
)
async def get_ingestion_log(
    log_id: int, service: IngestionService = Depends(get_ingestion_service)
):
    """Get details of a specific ingestion log."""
    result = await service.get_log(log_id)
    if not result:
        raise HTTPException(status_code=404, detail="Ingestion log not found")
    return result


@router.post(
    "/ingest/upload",
    response_model=UploadResponse,
    summary="Upload document and auto-ingest",
)
async def upload_and_ingest(
    file: UploadFile = File(...),
    force_reingest: bool = False,
    service: IngestionService = Depends(get_ingestion_service),
):
    """Upload an XLSX document to MinIO and automatically ingest it."""
    if not file.filename or not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are accepted")

    data = await file.read()
    return await service.upload_and_ingest(
        filename=file.filename,
        data=data,
        force_reingest=force_reingest,
    )


# ── Job Endpoints ──────────────────────────────────────────────────


@router.get(
    "/jobs/{job_id}", response_model=JobStatusResponse, summary="Get Job Status"
)
async def get_job_status(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
):
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ── Pipeline Endpoints ─────────────────────────────────────────────


@router.post(
    "/vectorrag/jabatan",
    response_model=JobAcceptedResponse,
    summary="Run VectorRAG Jabatan Pipeline",
    description="Upload XLSX files to chunk, embed, and ingest into pgvector.",
)
async def vectorrag_jabatan(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="XLSX files with jabatan data"),
    job_service: JobService = Depends(get_job_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    return await _start_pipeline_job(
        background_tasks,
        job_service,
        "vectorrag_jabatan",
        pipeline_service.run_vectorrag_jabatan,
        files,
        _VECTORRAG_JABATAN_EXTS,
    )


@router.post(
    "/knowledge-graph",
    response_model=JobAcceptedResponse,
    summary="Ingest Knowledge Graph",
    description=(
        "Upload dokumen ke Neo4j Knowledge Graph. "
        "Gunakan type=jabatan untuk file XLSX, type=regulasi untuk file PDF."
    ),
)
async def ingest_knowledge_graph(
    background_tasks: BackgroundTasks,
    type: KnowledgeGraphType = Query(..., description="Tipe dokumen: jabatan (XLSX) atau regulasi (PDF)"),
    files: List[UploadFile] = File(..., description="Dokumen yang akan diingest"),
    job_service: JobService = Depends(get_job_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    if type == KnowledgeGraphType.jabatan:
        return await _start_pipeline_job(
            background_tasks,
            job_service,
            "knowledge_graph_jabatan",
            pipeline_service.run_knowledge_graph_jabatan,
            files,
            _KNOWLEDGE_GRAPH_JABATAN_EXTS,
        )
    return await _start_pipeline_job(
        background_tasks,
        job_service,
        "knowledge_graph_regulasi",
        pipeline_service.run_knowledge_graph_regulasi,
        files,
        _KNOWLEDGE_GRAPH_REGULASI_EXTS,
    )


@router.post(
    "/profil-jabatan",
    response_model=JobAcceptedResponse,
    summary="Ingest Profil Jabatan",
    description=(
        "Upload file XLSX profil jabatan untuk diingest ke tabel jabatan_rules PostgreSQL "
        "dengan kategorisasi 14 field persyaratan (deterministik, tanpa LLM)."
    ),
)
async def ingest_profil_jabatan(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="File XLSX profil jabatan"),
    job_service: JobService = Depends(get_job_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    return await _start_pipeline_job(
        background_tasks,
        job_service,
        "profil_jabatan",
        pipeline_service.run_profil_jabatan,
        files,
        _PROFIL_JABATAN_EXTS,
    )


# ── Agent Endpoints ────────────────────────────────────────────────


@router.post(
    "/agents/{agent_name}", response_model=AgentResponse, summary="Run Single Agent"
)
async def run_single_agent(
    agent_name: AgentName,
    payload: AgentRunRequest,
    agent_service: AgentPipelineService = Depends(get_agent_pipeline_service),
):
    """Run a named agent with input_text and/or input_json."""
    try:
        result = await agent_service.run_agent(
            agent_name=agent_name,
            input_text=payload.input_text,
            input_json=payload.input_json,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Agent {agent_name.value} failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")
    return result


@router.post(
    "/agents/{agent_name}/evaluate",
    response_model=AgentResponse,
    summary="Evaluate Single Agent",
)
async def evaluate_single_agent(
    agent_name: AgentName,
    payload: AgentEvaluateRequest,
    agent_service: AgentPipelineService = Depends(get_agent_pipeline_service),
):
    """Evaluate an agent's output quality. Currently a placeholder."""
    # TODO: Implement evaluator agent integration
    return {
        "agent_name": f"{agent_name.value}_evaluator",
        "message": f"Evaluator for {agent_name.value} is not yet implemented",
        "output": None,
        "usage": None,
    }
