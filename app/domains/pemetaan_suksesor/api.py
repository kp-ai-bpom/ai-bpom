from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from .dto.request import (
    KandidatSuksesi,
    SimulasiRequest,
    SuksesorCreateRequest,
    SuksesorUpdateRequest,
)
from .dto.response import (
    KandidatListResponse,
    NineBoxResponse,
    SimulasiResponse,
    SuksesorDeleteResponse,
    SuksesorListResponse,
    SuksesorResponse,
)
from .services import (
    SimulationService,
    SuksesorService,
    get_simulation_service,
    get_suksesor_service,
    _load_candidates,
)

router = APIRouter()


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
    raw_candidates = _load_candidates()
    kandidat_list = [KandidatSuksesi.model_validate(c) for c in raw_candidates]

    return await service.run(
        target_jabatan=target_jabatan,
        kandidat_list=kandidat_list,
        top_n=5,
    )
