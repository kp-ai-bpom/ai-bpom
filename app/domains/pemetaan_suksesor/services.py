import asyncio
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import log
from app.db.database import get_db

from .core.agent import AgentAdapter, init_agents
from .dto.request import (
    KandidatSuksesi,
    SaveMatchingRequest,
    SuksesorCreateRequest,
    SuksesorUpdateRequest,
)
from .dto.response import (
    DetailEvaluasi,
    KandidatCard,
    KandidatListData,
    KandidatListResponse,
    KandidatResult,
    MatchingHistoryDetail,
    MatchingHistoryDetailResponse,
    MatchingHistoryListData,
    MatchingHistoryListResponse,
    MatchingHistorySaveResponse,
    MatchingHistorySummary,
    NineBoxData,
    NineBoxItem,
    NineBoxResponse,
    RekamJejakItem,
    SertifikasiItem,
    SimulasiDataResponse,
    SimulasiResponse,
    SkpTahunItem,
    SuksesorDataResponse,
    SuksesorDeleteResponse,
    SuksesorListDataResponse,
    SuksesorListResponse,
    SuksesorResponse,
)
from .repositories import MatchingHistoryRepository, SuksesorRepository

# ── Module-level Constants & Caches ────────────────────────────────

MAX_CONCURRENT_EVALUATIONS = 5
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EVALUATIONS + 1)

_JABATAN_RULES_PATH = Path(__file__).parent / "dto" / "jabatan_rules.json"
_CANDIDATES_JSON_PATH = Path(__file__).parent / "dto" / "candidates.json"

_jabatan_rules_cache: List[Dict[str, Any]] | None = None
_candidates_cache: List[Dict[str, Any]] | None = None

NINE_BOX_DEFINITIONS = {
    1: {
        "label": "Kinerja dibawah ekspektasi dan potensi rendah",
        "kinerja": "Dibawah Ekspektasi",
        "potensi": "Rendah",
        "selectable": False,
    },
    2: {
        "label": "Kinerja sesuai ekspektasi dan potensi rendah",
        "kinerja": "Sesuai Ekspektasi",
        "potensi": "Rendah",
        "selectable": False,
    },
    3: {
        "label": "Kinerja dibawah ekspektasi dan potensi menengah",
        "kinerja": "Dibawah Ekspektasi",
        "potensi": "Menengah",
        "selectable": False,
    },
    4: {
        "label": "Kinerja diatas ekspektasi dan potensi rendah",
        "kinerja": "Diatas Ekspektasi",
        "potensi": "Rendah",
        "selectable": False,
    },
    5: {
        "label": "Kinerja sesuai ekspektasi dan potensi menengah",
        "kinerja": "Sesuai Ekspektasi",
        "potensi": "Menengah",
        "selectable": False,
    },
    6: {
        "label": "Kinerja dibawah ekspektasi dan potensi tinggi",
        "kinerja": "Dibawah Ekspektasi",
        "potensi": "Tinggi",
        "selectable": False,
    },
    7: {
        "label": "Kinerja diatas ekspektasi dan potensi menengah",
        "kinerja": "Diatas Ekspektasi",
        "potensi": "Menengah",
        "selectable": True,
    },
    8: {
        "label": "Kinerja sesuai ekspektasi dan potensi tinggi",
        "kinerja": "Sesuai Ekspektasi",
        "potensi": "Tinggi",
        "selectable": True,
    },
    9: {
        "label": "Kinerja diatas ekspektasi dan potensi tinggi",
        "kinerja": "Diatas Ekspektasi",
        "potensi": "Tinggi",
        "selectable": True,
    },
}


# ── Helpers ───────────────────────────────────────────────────────


def _parse_box_number(posisi: str) -> int | None:
    """Parse box number from posisi_nine_box_talenta string, e.g. 'Kotak 9 (...)' → 9."""
    if not posisi:
        return None
    match = re.search(r"Kotak\s+(\d+)", posisi)
    return int(match.group(1)) if match else None


def _load_candidates() -> List[Dict[str, Any]]:
    """Load candidates.json with caching."""
    global _candidates_cache
    if _candidates_cache is None:
        try:
            with open(_CANDIDATES_JSON_PATH, encoding="utf-8") as f:
                _candidates_cache = json.load(f)
            assert _candidates_cache is not None
            log.info(f"📋 Candidates loaded — {len(_candidates_cache)} kandidat")
        except FileNotFoundError:
            log.warning("⚠️ candidates.json tidak ditemukan")
            _candidates_cache = []
        except json.JSONDecodeError:
            log.warning("⚠️ candidates.json format tidak valid")
            _candidates_cache = []
    assert _candidates_cache is not None
    return _candidates_cache


def _call_agent(agent, prompt: str) -> str:
    """Call a strands Agent synchronously and return its text output."""
    try:
        result = agent(prompt)
        # strands Agent may return a Result object or a plain string
        if hasattr(result, "result"):
            return str(result.result)
        return str(result)
    except Exception as exc:
        log.exception(f"❌ Agent call failed: {exc}")
        return ""


def _extract_json(text: str) -> Any:
    """Extract the first valid JSON object/array from LLM text output."""
    # Try the whole text first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    text = text.strip()

    # Find first { or [ and match to end
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


async def _run_agent_async(agent, prompt: str) -> str:
    """Run a blocking Agent call in a thread-pool so the event loop stays free."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _call_agent, agent, prompt)


def _safe_get(d: Any, *keys, default=None):
    """Safely traverse nested dicts/lists."""
    current = d
    for key in keys:
        try:
            current = current[key]
        except (KeyError, TypeError, IndexError):
            return default
    return current


# ── CRUD Service ──────────────────────────────────────────────────


class SuksesorService:
    """Service for Suksesor business logic."""

    def __init__(self, repository: SuksesorRepository):
        self._repo = repository

    async def create(self, data: SuksesorCreateRequest) -> SuksesorResponse:
        """Create a new Suksesor."""
        # Check if NIP already exists
        if await self._repo.exists_by_nip(data.nip):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Suksesor with NIP '{data.nip}' already exists",
            )

        suksesor = await self._repo.create(data)
        return SuksesorResponse(
            message="Suksesor created successfully",
            data=SuksesorDataResponse.model_validate(suksesor),
        )

    async def get_by_id(self, suksesor_id: UUID) -> SuksesorResponse:
        """Get a Suksesor by ID."""
        suksesor = await self._repo.get_by_id(suksesor_id)
        if not suksesor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suksesor with ID '{suksesor_id}' not found",
            )
        return SuksesorResponse(
            message="Suksesor retrieved successfully",
            data=SuksesorDataResponse.model_validate(suksesor),
        )

    async def get_by_nip(self, nip: str) -> SuksesorResponse:
        """Get a Suksesor by NIP."""
        suksesor = await self._repo.get_by_nip(nip)
        if not suksesor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suksesor with NIP '{nip}' not found",
            )
        return SuksesorResponse(
            message="Suksesor retrieved successfully",
            data=SuksesorDataResponse.model_validate(suksesor),
        )

    async def get_list(
        self,
        page: int = 1,
        page_size: int = 10,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> SuksesorListResponse:
        """Get paginated list of Suksesor."""
        items, total = await self._repo.get_list(
            page=page, page_size=page_size, search=search, is_active=is_active
        )

        total_pages = math.ceil(total / page_size) if total > 0 else 1

        return SuksesorListResponse(
            message="Suksesor list retrieved successfully",
            data=SuksesorListDataResponse(
                items=[SuksesorDataResponse.model_validate(item) for item in items],
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
            ),
        )

    async def update(
        self, suksesor_id: UUID, data: SuksesorUpdateRequest
    ) -> SuksesorResponse:
        """Update a Suksesor."""
        # Check if NIP is being updated and already exists
        if data.nip and await self._repo.exists_by_nip(
            data.nip, exclude_id=suksesor_id
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Suksesor with NIP '{data.nip}' already exists",
            )

        suksesor = await self._repo.update(suksesor_id, data)
        if not suksesor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suksesor with ID '{suksesor_id}' not found",
            )
        return SuksesorResponse(
            message="Suksesor updated successfully",
            data=SuksesorDataResponse.model_validate(suksesor),
        )

    async def delete(self, suksesor_id: UUID) -> SuksesorDeleteResponse:
        """Delete a Suksesor (hard delete)."""
        if not await self._repo.delete(suksesor_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suksesor with ID '{suksesor_id}' not found",
            )
        return SuksesorDeleteResponse(
            message=f"Suksesor '{suksesor_id}' deleted successfully",
            data={"id": str(suksesor_id)},
        )

    async def soft_delete(self, suksesor_id: UUID) -> SuksesorResponse:
        """Soft delete a Suksesor (set is_active to False)."""
        suksesor = await self._repo.soft_delete(suksesor_id)
        if not suksesor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suksesor with ID '{suksesor_id}' not found",
            )
        return SuksesorResponse(
            message="Suksesor deactivated successfully",
            data=SuksesorDataResponse.model_validate(suksesor),
        )


def get_suksesor_service(db: AsyncSession = Depends(get_db)) -> SuksesorService:
    """Dependency injection for SuksesorService."""
    repository = SuksesorRepository(db)
    return SuksesorService(repository)


# ── Matching History Service ─────────────────────────────────────────


class MatchingHistoryService:
    """Service for matching history business logic."""

    def __init__(self, repository: MatchingHistoryRepository):
        self._repo = repository

    async def save(self, data: SaveMatchingRequest) -> MatchingHistorySaveResponse:
        """Save a matching result to history."""
        history = await self._repo.create(data)
        return MatchingHistorySaveResponse(
            message="Riwayat matching berhasil disimpan",
            data=MatchingHistoryDetail.model_validate(history),
        )

    async def get_by_id(self, history_id: UUID) -> MatchingHistoryDetailResponse:
        """Get a matching history by ID."""
        history = await self._repo.get_by_id(history_id)
        if not history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Matching history with ID '{history_id}' not found",
            )
        return MatchingHistoryDetailResponse(
            message="Riwayat matching berhasil diambil",
            data=MatchingHistoryDetail.model_validate(history),
        )

    async def get_list(
        self, page: int = 1, page_size: int = 10
    ) -> MatchingHistoryListResponse:
        """Get paginated list of matching history."""
        items, total = await self._repo.get_list(page=page, page_size=page_size)
        total_pages = math.ceil(total / page_size) if total > 0 else 1
        return MatchingHistoryListResponse(
            message="Daftar riwayat matching berhasil diambil",
            data=MatchingHistoryListData(
                items=[MatchingHistorySummary.model_validate(i) for i in items],
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
            ),
        )


def get_matching_history_service(
    db: AsyncSession = Depends(get_db),
) -> MatchingHistoryService:
    """Dependency injection for MatchingHistoryService."""
    repository = MatchingHistoryRepository(db)
    return MatchingHistoryService(repository)


# ── Simulation Service ────────────────────────────────────────────


class SimulationService:
    """
    Service yang mengorkestrasi pipeline multi-agent untuk simulasi
    pemetaan suksesor JPT BPOM.

    Alur:
      1. Orchestrator → dekomposisi syarat jabatan target
      2. Search (per kandidat) → ekstraksi data kandidat
      3. Analysis (per kandidat) → evaluasi L-Eval + C-Eval
      4. Synthesis → scoring & ranking
      5. Reviewer → validasi output
    """

    def __init__(self, agent_adapter: AgentAdapter):
        self._agents = agent_adapter

    # ── Public API ────────────────────────────────────────────────

    async def run(
        self,
        target_jabatan: str,
        kandidat_list: List[KandidatSuksesi],
        top_n: int = 5,
    ) -> SimulasiResponse:
        """
        Menjalankan simulasi pemetaan suksesor secara lengkap.

        Args:
            target_jabatan: Nama jabatan target (e.g. "Inspektur I")
            kandidat_list: Daftar kandidat suksesi
            top_n: Jumlah kandidat teratas yang dikembalikan

        Returns:
            SimulasiResponse dengan top_n kandidat terbaik
        """
        log.info(
            f"🚀 Simulasi dimulai — target: {target_jabatan}, kandidat: {len(kandidat_list)}"
        )

        # Validasi: target jabatan harus ada di rules
        rules = self._load_jabatan_rules(target_jabatan)
        if rules is None:
            available = self.list_available_jabatan()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Target jabatan '{target_jabatan}' tidak ditemukan. "
                    f"Jabatan tersedia: {available}"
                ),
            )

        # ── Tahap 1: DECOMPOSITION ────────────────────────────────
        sub_tasks = await self._decompose(target_jabatan)
        log.info(f"📋 Tahap 1 selesai — {len(sub_tasks)} sub-tugas")

        # ── Tahap 2+3: RETRIEVAL & VALIDATION (per kandidat, paralel) ──
        # Each concurrent evaluation needs its own Agent instance (Strands doesn't
        # allow concurrent calls on the same Agent). We use an asyncio.Queue as a
        # pool — workers borrow an agent, evaluate, then return it.
        agent_queue: asyncio.Queue = asyncio.Queue()
        for a in self._agents.analysis_pool:
            agent_queue.put_nowait(a)

        async def _eval_one(kandidat: KandidatSuksesi) -> Dict:
            kandidat_id = kandidat.kandidat_suksesi.id
            kandidat_nama = kandidat.kandidat_suksesi.nama
            agent = await agent_queue.get()
            try:
                log.info(
                    f"🔍 [paralel] Mengevaluasi {kandidat_nama} ({kandidat_id})..."
                )
                evaluation = await self._evaluate_candidate(
                    kandidat, target_jabatan, sub_tasks, agent=agent
                )
                evaluation.setdefault(
                    "jabatan_saat_ini", kandidat.kandidat_suksesi.jabatan_saat_ini
                )
                log.info(f"✅ [paralel] {kandidat_nama} ({kandidat_id}) selesai")
                return evaluation
            finally:
                agent_queue.put_nowait(agent)

        evaluation_results: List[Dict] = list(
            await asyncio.gather(*[_eval_one(k) for k in kandidat_list])
        )
        eval_by_id: Dict[str, Dict] = {
            r["id_kandidat"]: r for r in evaluation_results if "id_kandidat" in r
        }

        log.info(
            f"✅ Tahap 2+3 selesai — {len(evaluation_results)} kandidat dievaluasi "
            f"(pool={len(self._agents.analysis_pool)})"
        )

        # ── Tahap 4: SCORING & RANKING ────────────────────────────
        ranking = await self._score_and_rank(target_jabatan, evaluation_results, top_n)
        log.info(f"📊 Tahap 4 selesai — ranking top {top_n}")

        # ── Review ────────────────────────────────────────────────
        review_note = await self._review(ranking)
        log.info("📝 Review selesai")

        # ── Build response (merge evaluation details back) ────────
        top_kandidat = self._build_kandidat_results(ranking, eval_by_id)

        return SimulasiResponse(
            message="Simulasi pemetaan suksesor berhasil",
            data=SimulasiDataResponse(
                target_jabatan=target_jabatan,
                total_kandidat=len(kandidat_list),
                top_kandidat=top_kandidat[:top_n],
                sub_tugas=sub_tasks,
                catatan_reviewer=review_note,
            ),
        )

    # ── Nine-Box & Kandidat Data ────────────────────────────────────

    @staticmethod
    def get_nine_box_data() -> NineBoxResponse:
        """
        Mengembalikan data nine-box talenta grid.
        Setiap box berisi: label, kinerja, potensi, selectable flag,
        jumlah kandidat, dan daftar nama kandidat (untuk tooltip).
        """
        candidates = _load_candidates()

        # Group candidates by box number
        box_candidates: Dict[int, List[str]] = {i: [] for i in range(1, 10)}
        for c in candidates:
            posisi = c.get("posisi_nine_box_talenta", "")
            box_num = _parse_box_number(posisi)
            if box_num and 1 <= box_num <= 9:
                nama = c.get("kandidat_suksesi", {}).get("nama", "")
                if nama:
                    box_candidates[box_num].append(nama)

        boxes = []
        for i in range(1, 10):
            defn = NINE_BOX_DEFINITIONS[i]
            boxes.append(
                NineBoxItem(
                    box_number=i,
                    label=defn["label"],
                    kinerja=defn["kinerja"],
                    potensi=defn["potensi"],
                    selectable=defn["selectable"],
                    count=len(box_candidates[i]),
                    candidates=box_candidates[i],
                )
            )

        return NineBoxResponse(
            message="Data nine-box talenta berhasil dimuat",
            data=NineBoxData(boxes=boxes),
        )

    @staticmethod
    def get_kandidat_by_boxes(boxes: List[int]) -> KandidatListResponse:
        """
        Mengembalikan kandidat yang berada di box-box terpilih,
        lengkap dengan ringkasan untuk kartu UI.
        """
        candidates = _load_candidates()

        # Validate box numbers
        valid_boxes = [b for b in boxes if 1 <= b <= 9]
        if not valid_boxes:
            return KandidatListResponse(
                message="Tidak ada box valid yang dipilih",
                data=KandidatListData(total=0, filtered_boxes=valid_boxes, kandidat=[]),
            )

        filtered = []
        for c in candidates:
            posisi = c.get("posisi_nine_box_talenta", "")
            box_num = _parse_box_number(posisi)
            if box_num in valid_boxes:
                profil = c.get("kandidat_suksesi", {})

                rekam_jejak = [
                    RekamJejakItem(
                        periode=r.get("periode", ""),
                        jabatan=r.get("jabatan", ""),
                        durasi_tahun=r.get("durasi_tahun", 0),
                        deskripsi_tugas_dan_fungsi=r.get(
                            "deskripsi_tugas_dan_fungsi", ""
                        ),
                    )
                    for r in c.get("rekam_jejak", [])
                ]

                sertifikasi = [
                    SertifikasiItem(
                        nama_sertifikasi=s.get("nama_sertifikasi", ""),
                        tahun=s.get("tahun", 0),
                        keterangan=s.get("keterangan", ""),
                    )
                    for s in c.get("sertifikasi", [])
                ]

                skp_raw = c.get("skp", {})
                skp = {
                    k: SkpTahunItem(
                        rating_hasil_kerja=v.get("rating_hasil_kerja", ""),
                        rating_perilaku_kerja=v.get("rating_perilaku_kerja", ""),
                        keterangan=v.get("keterangan", ""),
                    )
                    for k, v in skp_raw.items()
                }

                filtered.append(
                    KandidatCard(
                        id=profil.get("id", ""),
                        nama=profil.get("nama", ""),
                        jabatan_saat_ini=profil.get("jabatan_saat_ini", ""),
                        unit_kerja=profil.get("unit_kerja", ""),
                        box_number=box_num,
                        rekam_jejak=rekam_jejak,
                        sertifikasi=sertifikasi,
                        skp=skp,
                        posisi_nine_box_talenta=posisi,
                    )
                )

        return KandidatListResponse(
            message=f"{len(filtered)} kandidat ditemukan dari box terpilih",
            data=KandidatListData(
                total=len(filtered),
                filtered_boxes=valid_boxes,
                kandidat=filtered,
            ),
        )

    # ── Rules Loader (hardcoded — nanti diganti Hybrid RAG) ─────────

    @staticmethod
    def _load_jabatan_rules(target_jabatan: str) -> Dict | None:
        """
        Load aturan jabatan dari jabatan_rules.json.
        Mengembalikan seluruh data (deskripsi + persyaratan) jika ditemukan.
        Nanti diganti dengan Hybrid RAG (GraphRAG + VectorRAG).
        """
        global _jabatan_rules_cache
        if _jabatan_rules_cache is None:
            try:
                with open(_JABATAN_RULES_PATH, encoding="utf-8") as f:
                    raw = json.load(f)
                # Normalisasi: simpan sebagai list of rules
                if "deskripsi_jabatan" in raw:
                    # Format single-jabatan → bungkus jadi list
                    _jabatan_rules_cache = [raw]
                elif isinstance(raw, list):
                    # Format multi-jabatan
                    _jabatan_rules_cache = raw
                else:
                    _jabatan_rules_cache = []
                log.info(
                    f"📋 Jabatan rules loaded — {len(_jabatan_rules_cache)} posisi"
                )
            except FileNotFoundError:
                log.warning("⚠️ jabatan_rules.json tidak ditemukan")
                _jabatan_rules_cache = []
                return None
            except json.JSONDecodeError:
                log.warning("⚠️ jabatan_rules.json format tidak valid")
                _jabatan_rules_cache = []
                return None

        # Assert to help type checker understand _jabatan_rules_cache is not None
        assert _jabatan_rules_cache is not None

        # Search by nama_jabatan (case-insensitive)
        normalized = target_jabatan.lower().strip()
        for entry in _jabatan_rules_cache:
            nama = entry.get("deskripsi_jabatan", {}).get("nama_jabatan", "")
            if nama.lower().strip() == normalized:
                return entry

        return None

    @staticmethod
    def list_available_jabatan() -> List[str]:
        """Return daftar nama jabatan yang tersedia di jabatan_rules.json."""
        global _jabatan_rules_cache
        if _jabatan_rules_cache is None:
            # Trigger load
            SimulationService._load_jabatan_rules("")

        assert _jabatan_rules_cache is not None
        return [
            entry.get("deskripsi_jabatan", {}).get("nama_jabatan", "")
            for entry in _jabatan_rules_cache
            if entry.get("deskripsi_jabatan", {}).get("nama_jabatan")
        ]

    # ── Tahap 1: Decomposition ────────────────────────────────────

    async def _decompose(self, target_jabatan: str) -> List[Dict]:
        """
        Orchestrator agent menerima aturan jabatan dan menghasilkan sub-tugas evaluasi.
        Sub-tugas di-generate oleh agent, bukan hardcoded.
        Source: (1) jabatan_rules.json → (2) LLM knowledge jika tidak ada rules
        """
        # Load rules jika ada
        rules = self._load_jabatan_rules(target_jabatan)

        if rules:
            # Berikan rules lengkap ke orchestrator untuk di-decompose
            prompt = (
                f"Berikut adalah aturan jabatan target suksesi:\n\n"
                f"```json\n{json.dumps(rules, ensure_ascii=False, indent=2)}\n```\n\n"
                "Berdasarkan data di atas, dekomposisikan persyaratan menjadi sub-tugas evaluasi.\n"
                "Setiap sub-tugas harus memetakan ke persyaratan spesifik di data.\n"
                "Tentukan apakah setiap sub-tugas bersifat mutlak (syarat wajib) atau tambahan (diutamakan).\n"
                "Tentukan bobot (0-100) berdasarkan prioritas persyaratan.\n"
                "Untuk sub-tugas pencocokan semantik, identifikasi kata kunci pencocokan.\n"
                "Output WAJIB JSON sesuai format yang ditentukan di system prompt."
            )
            log.info(
                f"📋 Tahap 1 (agent+rules): mendekomposisi '{target_jabatan}' dari jabatan_rules.json"
            )
        else:
            # Tidak ada rules — agent gunakan pengetahuan LLM
            prompt = (
                f"Jabatan target suksesi: {target_jabatan}\n\n"
                "Tidak ada aturan spesifik yang tersedia di database. "
                "Berdasarkan PerBPOM No. 21 Tahun 2020 dan KepKabadan 322/2023, "
                "dekomposisikan persyaratan jabatan ini menjadi sub-tugas evaluasi.\n"
                "Output WAJIB JSON sesuai format yang ditentukan di system prompt."
            )
            log.info(
                f"📋 Tahap 1 (agent-only): mendekomposisi '{target_jabatan}' dari pengetahuan LLM"
            )

        raw = await _run_agent_async(self._agents.orchestrator, prompt)
        parsed = _extract_json(raw)

        if parsed and isinstance(parsed, dict) and "sub_tasks" in parsed:
            return parsed["sub_tasks"]

        # Fallback terakhir
        log.warning("⚠️ Fallback sub-tasks digunakan (rules & agent gagal)")
        return [
            {
                "id": 1,
                "nama": "Pengalaman Eselon III/Ahli Madya min 2 tahun",
                "syarat_mutlak": True,
                "bobot": 30,
            },
            {
                "id": 2,
                "nama": "Pengalaman bidang terkait kumulatif min 5 tahun",
                "syarat_mutlak": True,
                "bobot": 25,
            },
            {
                "id": 3,
                "nama": "Pencocokan semantik fungsional",
                "syarat_mutlak": False,
                "bobot": 20,
            },
            {
                "id": 4,
                "nama": "SKP baik 2 tahun + Kotak Talenta prioritas",
                "syarat_mutlak": True,
                "bobot": 15,
            },
            {
                "id": 5,
                "nama": "Diklat PIM + Kemampuan Bahasa Inggris",
                "syarat_mutlak": False,
                "bobot": 10,
            },
        ]

    # ── Tahap 2+3: Evaluate single candidate ──────────────────────

    async def _evaluate_candidate(
        self,
        kandidat: KandidatSuksesi,
        target_jabatan: str,
        sub_tasks: List[Dict],
        agent: Any = None,
    ) -> Dict:
        """Search (extract) + Analysis (L-Eval + C-Eval) untuk satu kandidat."""
        eval_agent = agent or self._agents.analysis
        kandidat_json = kandidat.model_dump(mode="json")

        # Sertakan konteks jabatan dari rules jika ada
        rules = self._load_jabatan_rules(target_jabatan)
        context_extra = ""
        if rules:
            deskripsi = rules.get("deskripsi_jabatan", {})
            fungsi = deskripsi.get("fungsi", [])
            if fungsi:
                context_extra += f"\nFungsi Jabatan {target_jabatan}:\n"
                for i, f in enumerate(fungsi, 1):
                    context_extra += f"  {i}. {f}\n"

            # Kompetensi spesifik dari persyaratan
            pengalaman = rules.get("persyaratan", {}).get("pengalaman_bidang_tugas", {})
            kompetensi = pengalaman.get("kompetensi_spesifik", [])
            if kompetensi:
                context_extra += "\nKompetensi Spesifik:\n"
                for i, k in enumerate(kompetensi, 1):
                    context_extra += f"  {i}. {k}\n"

            # Kumpulkan kata kunci dari sub_tasks yang di-generate orchestrator
            all_keywords = []
            for st in sub_tasks:
                for kw in st.get("kata_kunci_pencocokan", []):
                    if kw not in all_keywords:
                        all_keywords.append(kw)
            if all_keywords:
                context_extra += (
                    f"\nKata Kunci Pencocokan Semantik:\n"
                    f"{json.dumps(all_keywords, ensure_ascii=False)}\n"
                )

        prompt = (
            f"Jabatan Target: {target_jabatan}\n"
            f"{context_extra}\n"
            f"Data Kandidat:\n```json\n{json.dumps(kandidat_json, ensure_ascii=False, indent=2)}\n```\n\n"
            f"Sub-Tugas Evaluasi:\n```json\n{json.dumps(sub_tasks, ensure_ascii=False, indent=2)}\n```\n\n"
            "Tahap 2 — Ekstrak informasi esensial kandidat untuk setiap sub-tugas.\n"
            "Tahap 3 — Lakukan Logical Evaluation (L-Eval) dan Counterfactual Evaluation (C-Eval).\n"
            "Output WAJIB JSON sesuai format yang ditentukan di system prompt Analysis Agent."
        )

        raw = await _run_agent_async(eval_agent, prompt)
        parsed = _extract_json(raw)

        if parsed and isinstance(parsed, dict):
            parsed.setdefault("id_kandidat", kandidat.kandidat_suksesi.id)
            parsed.setdefault("nama", kandidat.kandidat_suksesi.nama)
            return parsed

        # Fallback: basic result if parsing fails
        log.warning(f"⚠️ Fallback evaluasi untuk {kandidat.kandidat_suksesi.id}")
        return {
            "id_kandidat": kandidat.kandidat_suksesi.id,
            "nama": kandidat.kandidat_suksesi.nama,
            "l_eval": {"keputusan": "REJECT", "alasan": "Gagal memproses evaluasi"},
            "c_eval": {
                "keputusan": "REJECT",
                "bukti_kontradiksi": "Data tidak dapat dievaluasi",
            },
            "acceptances": 0,
            "detail_evaluasi": {
                "pengalaman": {
                    "status": "Tidak Dapat Dievaluasi",
                    "keterangan": "Agent output tidak terparse",
                },
                "fungsi_semantik": {
                    "status": "Tidak Dapat Dievaluasi",
                    "keterangan": "Agent output tidak terparse",
                },
                "kinerja_talenta": {
                    "status": "Tidak Dapat Dievaluasi",
                    "keterangan": "Agent output tidak terparse",
                },
                "kualifikasi_tambahan": {
                    "status": "Tidak Dapat Dievaluasi",
                    "keterangan": "Agent output tidak terparse",
                },
            },
        }

    # ── Tahap 4: Scoring & Ranking ────────────────────────────────

    async def _score_and_rank(
        self,
        target_jabatan: str,
        evaluation_results: List[Dict],
        top_n: int = 5,
    ) -> List[Dict]:
        """Synthesis agent menggabungkan hasil evaluasi & memberi skor."""
        prompt = (
            f"Jabatan Target: {target_jabatan}\n\n"
            f"Hasil Evaluasi Semua Kandidat:\n```json\n"
            f"{json.dumps(evaluation_results, ensure_ascii=False, indent=2)}\n```\n\n"
            f"Tahap 4 — Berikan skor kesesuaian (0-100) untuk setiap kandidat, "
            f"tentukan kategori kesiapan, tingkat keyakinan, dan kesimpulan. "
            f"Urutkan dari skor tertinggi ke terendah. Ambil top {top_n}.\n\n"
            "PENTING: Untuk setiap kandidat di peringkat, WAJIB sertakan:\n"
            "- id_kandidat, nama, jabatan_saat_ini\n"
            "- skor_kesesuaian, kategori_kesiapan, confidence_level, kesimpulan\n"
            "- acceptances (jumlah ACCEPT dari L-Eval + C-Eval: 0, 1, atau 2)\n"
            "- detail_evaluasi (salin lengkap dari data evaluasi setiap kandidat)\n\n"
            "Output WAJIB JSON sesuai format yang ditentukan di system prompt Synthesis Agent."
        )

        raw = await _run_agent_async(self._agents.synthesis, prompt)
        parsed = _extract_json(raw)

        if parsed and isinstance(parsed, dict) and "peringkat" in parsed:
            return parsed["peringkat"][:top_n]

        # Fallback: simple scoring from acceptances
        log.warning("⚠️ Fallback scoring digunakan (agent output tidak terparse)")
        ranked = sorted(
            evaluation_results,
            key=lambda x: x.get("acceptances", 0),
            reverse=True,
        )
        results = []
        for i, eval_data in enumerate(ranked[:top_n], 1):
            acc = eval_data.get("acceptances", 0)
            results.append(
                {
                    "rank": i,
                    "id_kandidat": eval_data.get("id_kandidat", f"UNKNOWN-{i}"),
                    "nama": eval_data.get("nama", "Tidak diketahui"),
                    "skor_kesesuaian": acc * 50,  # rough mapping
                    "kategori_kesiapan": "SUKSESOR"
                    if acc == 2
                    else "POTENSIAL"
                    if acc == 1
                    else "BELUM SIAP",
                    "confidence_level": "Tinggi"
                    if acc == 2
                    else "Sedang"
                    if acc == 1
                    else "Rendah",
                    "kesimpulan": "Clear" if acc == 2 else "Review Needed",
                }
            )
        return results

    # ── Review ─────────────────────────────────────────────────────

    async def _review(self, ranking: List[Dict]) -> str:
        """Reviewer agent memvalidasi output akhir."""
        prompt = (
            f"Peringkat Kandidat Top 5:\n```json\n"
            f"{json.dumps(ranking, ensure_ascii=False, indent=2)}\n```\n\n"
            "Validasi output akhir: apakah evaluasi konsisten, skor akurat, "
            "dan pemilihan top 5 dapat dipertanggungjawabkan?\n"
            "Output WAJIB JSON sesuai format Reviewer Agent."
        )

        raw = await _run_agent_async(self._agents.reviewer, prompt)
        parsed = _extract_json(raw)

        if parsed and isinstance(parsed, dict):
            return parsed.get("catatan", parsed.get("rekomendasi", str(parsed)))

        return "Review tidak dapat diproses — output agent tidak terparse."

    # ── Build response objects ─────────────────────────────────────

    @staticmethod
    def _build_kandidat_results(
        ranking: List[Dict], eval_by_id: Dict[str, Dict]
    ) -> List[KandidatResult]:
        """
        Convert raw ranking dicts to KandidatResult models.
        Merges evaluation details from eval_by_id when synthesis agent omits them.
        """
        results: List[KandidatResult] = []
        for entry in ranking:
            kid = entry.get("id_kandidat", "")
            source_eval = eval_by_id.get(kid, {})

            # Merge: prefer synthesis output, fall back to evaluation result
            detail_raw = entry.get("detail_evaluasi") or source_eval.get(
                "detail_evaluasi"
            )
            detail_models = None
            if isinstance(detail_raw, dict):
                detail_models = {
                    k: DetailEvaluasi(**v)
                    if isinstance(v, dict)
                    else DetailEvaluasi(status=str(v), keterangan="")
                    for k, v in detail_raw.items()
                }

            # Derive acceptances from L-Eval + C-Eval if synthesis omitted it
            acceptances = entry.get("acceptances")
            if acceptances is None:
                l_eval = source_eval.get("l_eval", {})
                c_eval = source_eval.get("c_eval", {})
                acc = 0
                if l_eval.get("keputusan", "").upper() == "ACCEPT":
                    acc += 1
                if c_eval.get("keputusan", "").upper() == "ACCEPT":
                    acc += 1
                acceptances = acc

            # Derive confidence_level from acceptances if synthesis omitted it
            confidence = entry.get("confidence_level", "")
            if not confidence:
                acc = int(acceptances)
                confidence = (
                    "Tinggi" if acc == 2 else "Sedang" if acc == 1 else "Rendah"
                )

            # Derive kesimpulan from acceptances if synthesis omitted it
            kesimpulan = entry.get("kesimpulan", "")
            if not kesimpulan:
                kesimpulan = "Clear" if int(acceptances) == 2 else "Review Needed"

            # Alasan penilaian: from synthesis, or build from detail_evaluasi
            alasan = entry.get("alasan_penilaian", "")
            if not alasan and detail_models:
                # Fallback: compose reason from detail_evaluasi
                parts = []
                for aspek, detail in detail_models.items():
                    parts.append(f"{aspek}: {detail.status} — {detail.keterangan}")
                alasan = "; ".join(parts)

            results.append(
                KandidatResult(
                    rank=entry.get("rank", 0),
                    id_kandidat=kid,
                    nama=entry.get("nama", ""),
                    jabatan_saat_ini=entry.get("jabatan_saat_ini", "")
                    or source_eval.get("jabatan_saat_ini", ""),
                    skor_kesesuaian=float(entry.get("skor_kesesuaian", 0)),
                    kategori_kesiapan=entry.get("kategori_kesiapan", "BELUM SIAP"),
                    confidence_level=confidence,
                    acceptances=int(acceptances),
                    kesimpulan=kesimpulan,
                    alasan_penilaian=alasan,
                    detail_evaluasi=detail_models,
                )
            )
        return results


def get_simulation_service() -> SimulationService:
    """Factory untuk mendapatkan instance SimulationService."""
    adapter = init_agents()
    return SimulationService(adapter)
