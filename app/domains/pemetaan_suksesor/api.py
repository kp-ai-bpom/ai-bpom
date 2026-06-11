import asyncio
import logging
import os
import shutil
import tempfile
from enum import Enum
from typing import Annotated, Callable, List
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

from .dto.pipeline import AgentEvaluateRequest, AgentName, PlannerRequest
from .dto.request import (
    KandidatSIASN,
    SaveMatchingRequest,
    SimulasiRequest,
)
from .dto.response import (
    AgentResponse,
    JobAcceptedResponse,
    JobStatusResponse,
    KandidatListResponse,
    MatchingHistoryDetailResponse,
    MatchingHistoryListResponse,
    MatchingHistorySaveResponse,
    NineBoxResponse,
    SimulasiResponse,
)
from .services import (
    AgentPipelineService,
    JobService,
    MatchingHistoryService,
    PipelineService,
    SimulationService,
    get_agent_pipeline_service,
    get_job_service,
    get_matching_history_service,
    get_pipeline_service,
    get_simulation_service,
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
            raise HTTPException(
                status_code=400, detail="All files must have a filename"
            )
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
    type: KnowledgeGraphType = Query(
        ..., description="Tipe dokumen: jabatan (XLSX) atau regulasi (PDF)"
    ),
    files: Annotated[
        list[UploadFile], File(description="Dokumen yang akan diingest")
    ] = ...,
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
    "/agents/planner",
    response_model=AgentResponse,
    summary="Run Planner Agent",
    description=(
        "Menjalankan Planner Agent XAI-MENTARI. "
        "Menerima jabatan target dan daftar kandidat, "
        "lalu menghasilkan XAI Blueprint berisi profil jabatan, "
        "aturan penilaian dari Knowledge Graph, dan paket data per-kandidat "
        "untuk diproses oleh Analysis Agent."
    ),
)
async def run_planner_agent(
    payload: PlannerRequest,
    agent_service: AgentPipelineService = Depends(get_agent_pipeline_service),
) -> AgentResponse:
    """Run the Planner Agent with target jabatan and candidate list."""
    try:
        result = await agent_service.run_planner(
            input_json=payload.model_dump(),
        )
    except Exception as e:
        logger.error("Planner agent failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Planner agent failed: {e}")
    return result


@router.post(
    "/agents/{agent_name}/evaluate",
    response_model=AgentResponse,
    summary="Evaluate Agent Output",
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
