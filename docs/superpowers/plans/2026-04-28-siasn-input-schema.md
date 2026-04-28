# SIASN Input Schema Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the nested `KandidatSuksesi` schema with a flat SIASN-format `KandidatSIASN` schema across all endpoints, agent prompts, and sample data.

**Architecture:** New flat `KandidatSIASN` model replaces all nested sub-models (`KandidatProfil`, `RekamJejakEntry`, `SertifikasiEntry`, `SKPTahun`, `KandidatSuksesi`). All service code, response DTOs, agent prompts, and sample data are updated to use SIASN fields.

**Tech Stack:** Python, Pydantic, FastAPI, Strands Agents SDK

---

### Task 1: Replace request DTOs with SIASN schema

**Files:**
- Modify: `app/domains/pemetaan_suksesor/dto/request.py`

- [ ] **Step 1: Remove old models and add `KandidatSIASN`**

Remove these classes entirely: `KandidatProfil`, `RekamJejakEntry`, `SertifikasiEntry`, `SKPTahun`, `KandidatSuksesi`.

Add `KandidatSIASN` in their place. The full file should look like this (keeping `SuksesorCreateRequest`, `SuksesorUpdateRequest`, `SuksesorListRequest`, `SaveMatchingRequest` unchanged):

```python
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SuksesorCreateRequest(BaseModel):
    """Request DTO for creating a new Suksesor."""

    nip: str = Field(..., min_length=1, max_length=20, description="NIP pegawai")
    nama: str = Field(..., min_length=1, max_length=255, description="Nama lengkap")
    unit_kerja: Optional[str] = Field(None, max_length=255, description="Unit kerja")
    grade: Optional[str] = Field(None, max_length=5, description="Grade jabatan")
    kompetensi: Optional[str] = Field(None, description="Kompetensi (JSON format)")
    potensi: Optional[str] = Field(
        None, max_length=20, description="Potensi (High/Medium/Low)"
    )
    readiness: Optional[int] = Field(
        None, ge=0, le=100, description="Readiness level (0-100)"
    )
    is_active: bool = Field(True, description="Status aktif")


class SuksesorUpdateRequest(BaseModel):
    """Request DTO for updating an existing Suksesor."""

    nip: Optional[str] = Field(None, min_length=1, max_length=20)
    nama: Optional[str] = Field(None, min_length=1, max_length=255)
    unit_kerja: Optional[str] = Field(None, max_length=255)
    grade: Optional[str] = Field(None, max_length=5)
    kompetensi: Optional[str] = None
    potensi: Optional[str] = Field(None, max_length=20)
    readiness: Optional[int] = Field(None, ge=0, le=100)
    is_active: Optional[bool] = None


class SuksesorListRequest(BaseModel):
    """Request DTO for listing Suksesor with pagination."""

    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(10, ge=1, le=100, description="Items per page")
    search: Optional[str] = Field(None, description="Search by nama or nip")
    is_active: Optional[bool] = Field(None, description="Filter by active status")


# ── SIASN Candidate Schema ────────────────────────────────────────


class KandidatSIASN(BaseModel):
    """Data kandidat dalam format SIASN (Sistem Informasi Aparatur Sipil Negara)."""

    nip: str = Field(..., description="NIP pegawai")
    nama: str = Field(..., description="Nama singkat pegawai")
    nama_lengkap: str = Field(..., description="Nama lengkap dengan gelar")
    jabatan_nama: str = Field(..., description="Nama jabatan saat ini")
    foto_url: Optional[str] = Field(None, description="URL foto pegawai")
    pool: Optional[int] = Field(None, description="Nomor pool")
    pool_id: Optional[int] = Field(None, description="ID pool")
    fungsi_jabatan: List[str] = Field(default_factory=list, description="Daftar fungsi jabatan")
    riwayat_jabatan: List[str] = Field(default_factory=list, description="Daftar riwayat jabatan")
    jabatan_terakhir: str = Field(..., description="Jabatan terakhir")
    riwayat_pendidikan: List[str] = Field(default_factory=list, description="Daftar riwayat pendidikan")
    nilai_potensi: Optional[float] = Field(None, description="Nilai potensi")
    nilai_mansoskul: Optional[int] = Field(None, description="Nilai manajemen sosial kultural")
    nilai_kinerja: Optional[float] = Field(None, description="Nilai kinerja")
    nilai_kinerja_label: Optional[str] = Field(None, description="Label nilai kinerja")
    masa_kerja: Optional[int] = Field(None, description="Masa kerja dalam tahun")
    masa_kerja_total_tahun: Optional[int] = Field(None, description="Total masa kerja dalam tahun")
    diklat_pim_level: Optional[str] = Field(None, description="Level diklat kepemimpinan")
    jenjang_pendidikan_id: Optional[str] = Field(None, description="ID jenjang pendidikan")
    pengalaman_struktural_tahun: Optional[str] = Field(None, description="Pengalaman struktural dalam tahun")
    current_eselon_id: Optional[str] = Field(None, description="ID eselon saat ini")
    target_eselon_id: Optional[str] = Field(None, description="ID eselon target")
    recommendation_label: Optional[str] = Field(None, description="Label rekomendasi")
    recommendation_type: Optional[str] = Field(None, description="Tipe rekomendasi")
    is_eligible: Optional[bool] = Field(None, description="Status eligible")
    rhk: List[str] = Field(default_factory=list, description="Daftar Rencana Hasil Kerja")

    model_config = {"from_attributes": True}


class SimulasiRequest(BaseModel):
    """Request DTO untuk simulasi pemetaan suksesor."""

    target_jabatan: str = Field(
        ..., description="Jabatan target suksesi, e.g. Inspektur I"
    )
    kandidat: List[KandidatSIASN] = Field(
        ..., min_length=1, max_length=50, description="Daftar kandidat SIASN"
    )


class SaveMatchingRequest(BaseModel):
    """Request DTO untuk menyimpan hasil matching ke riwayat."""

    target_jabatan: str = Field(
        ..., description="Jabatan target suksesi yang digunakan saat matching"
    )
    total_kandidat: int = Field(
        ..., ge=1, description="Total kandidat yang dievaluasi"
    )
    top_kandidat: List[Dict] = Field(
        ..., min_length=1, description="Daftar top kandidat hasil matching"
    )
    sub_tugas: Optional[List[Dict]] = Field(
        None, description="Sub-tugas dekomposisi dari pipeline"
    )
    catatan_reviewer: Optional[str] = Field(
        None, description="Catatan dari reviewer agent"
    )
```

- [ ] **Step 2: Verify imports load cleanly**

Run: `python -c "from app.domains.pemetaan_suksesor.dto.request import KandidatSIASN, SimulasiRequest; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/domains/pemetaan_suksesor/dto/request.py
git commit -m "refactor: replace KandidatSuksesi with KandidatSIASN flat schema

Replace nested KandidatSuksesi model (with KandidatProfil, RekamJejakEntry,
SertifikasiEntry, SKPTahun sub-models) with flat KandidatSIASN model matching
the SIASN data format. New model includes nip, nama_lengkap, jabatan_nama,
fungsi_jabatan, riwayat_jabatan, nilai_potensi, nilai_mansoskul, rhk, and
other SIASN-specific fields."
```

---

### Task 2: Replace response DTOs with SIASN schema

**Files:**
- Modify: `app/domains/pemetaan_suksesor/dto/response.py`

- [ ] **Step 1: Remove old sub-models and update `KandidatCard`**

Remove these classes: `RekamJejakItem`, `SertifikasiItem`, `SkpTahunItem`.

Replace `KandidatCard` with SIASN fields. The new `KandidatCard`:

```python
class KandidatCard(BaseModel):
    """Data kandidat SIASN untuk daftar pilihan di Step 3."""

    nip: str
    nama: str
    nama_lengkap: str
    jabatan_nama: str
    jabatan_terakhir: str
    fungsi_jabatan: List[str] = []
    riwayat_jabatan: List[str] = []
    riwayat_pendidikan: List[str] = []
    nilai_potensi: Optional[float] = None
    nilai_mansoskul: Optional[int] = None
    nilai_kinerja: Optional[float] = None
    nilai_kinerja_label: Optional[str] = None
    masa_kerja: Optional[int] = None
    diklat_pim_level: Optional[str] = None
    pengalaman_struktural_tahun: Optional[str] = None
    current_eselon_id: Optional[str] = None
    target_eselon_id: Optional[str] = None
    recommendation_label: Optional[str] = None
    recommendation_type: Optional[str] = None
    is_eligible: Optional[bool] = None
    rhk: List[str] = []
    posisi_nine_box_talenta: Optional[str] = None
    box_number: int
```

- [ ] **Step 2: Verify imports load cleanly**

Run: `python -c "from app.domains.pemetaan_suksesor.dto.response import KandidatCard; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/domains/pemetaan_suksesor/dto/response.py
git commit -m "refactor: replace KandidatCard sub-models with SIASN fields

Remove RekamJejakItem, SertifikasiItem, SkpTahunItem. Replace KandidatCard
fields with flat SIASN model: nip, nama_lengkap, jabatan_nama, riwayat_jabatan,
nilai_potensi, nilai_mansoskul, rhk, etc."
```

---

### Task 3: Update `simulation.py` — imports, type annotations, and `run()` method

**Files:**
- Modify: `app/domains/pemetaan_suksesor/services/simulation.py`

- [ ] **Step 1: Update imports**

Replace line 10 (`from ..dto.request import KandidatSuksesi`) with:

```python
from ..dto.request import KandidatSIASN
```

Replace lines 11-25 (the `from ..dto.response import ...` block) with:

```python
from ..dto.response import (
    DetailEvaluasi,
    KandidatCard,
    KandidatListData,
    KandidatListResponse,
    KandidatResult,
    NineBoxData,
    NineBoxItem,
    NineBoxResponse,
    SimulasiDataResponse,
    SimulasiResponse,
)
```

(Remove `RekamJejakItem`, `SertifikasiItem`, `SkpTahunItem` from imports.)

- [ ] **Step 2: Update `run()` method signature and `_eval_one` closure**

Change line 63 from:
```python
kandidat_list: List[KandidatSuksesi],
```
to:
```python
kandidat_list: List[KandidatSIASN],
```

Change the `_eval_one` closure (lines 102-119). Replace the entire closure:

```python
        async def _eval_one(kandidat: KandidatSIASN) -> Dict:
            kandidat_id = kandidat.nip
            kandidat_nama = kandidat.nama
            agent = await agent_queue.get()
            try:
                log.info(
                    f"🔍 [paralel] Mengevaluasi {kandidat_nama} ({kandidat_id})..."
                )
                evaluation = await self._evaluate_candidate(
                    kandidat, target_jabatan, sub_tasks, agent=agent
                )
                evaluation.setdefault(
                    "jabatan_saat_ini", kandidat.jabatan_nama
                )
                log.info(f"✅ [paralel] {kandidat_nama} ({kandidat_id}) selesai")
                return evaluation
            finally:
                agent_queue.put_nowait(agent)
```

- [ ] **Step 3: Update `_evaluate_candidate` signature and body**

Change the method signature on line 410-411 from:
```python
    async def _evaluate_candidate(
        self,
        kandidat: KandidatSuksesi,
```
to:
```python
    async def _evaluate_candidate(
        self,
        kandidat: KandidatSIASN,
```

Change the fallback values on lines 462-463 from:
```python
            parsed.setdefault("id_kandidat", kandidat.kandidat_suksesi.id)
            parsed.setdefault("nama", kandidat.kandidat_suksesi.nama)
```
to:
```python
            parsed.setdefault("id_kandidat", kandidat.nip)
            parsed.setdefault("nama", kandidat.nama)
```

Change the fallback return dict on lines 466-467 from:
```python
        log.warning(f"⚠️ Fallback evaluasi untuk {kandidat.kandidat_suksesi.id}")
        return {
            "id_kandidat": kandidat.kandidat_suksesi.id,
            "nama": kandidat.kandidat_suksesi.nama,
```
to:
```python
        log.warning(f"⚠️ Fallback evaluasi untuk {kandidat.nip}")
        return {
            "id_kandidat": kandidat.nip,
            "nama": kandidat.nama,
```

- [ ] **Step 4: Commit**

```bash
git add app/domains/pemetaan_suksesor/services/simulation.py
git commit -m "refactor: update simulation service to use KandidatSIASN

Change type annotations from KandidatSuksesi to KandidatSIASN.
Update field access from nested kandidat.kandidat_suksesi.id/nama
to flat kandidat.nip/nama. Update fallback evaluations accordingly."
```

---

### Task 4: Update `get_nine_box_data()` and `get_kandidat_by_boxes()`

**Files:**
- Modify: `app/domains/pemetaan_suksesor/services/simulation.py`

- [ ] **Step 1: Update `get_nine_box_data()` — change `c.get("kandidat_suksesi", {}).get("nama", "")` to `c.get("nama", "")`**

In the `get_nine_box_data` method (around line 173), replace:
```python
                nama = c.get("kandidat_suksesi", {}).get("nama", "")
```
with:
```python
                nama = c.get("nama", "")
```

- [ ] **Step 2: Update `get_kandidat_by_boxes()` — rewrite the candidate parsing**

Replace the entire `get_kandidat_by_boxes` method body (lines 198-271) with SIASN-compatible parsing:

```python
    @staticmethod
    def get_kandidat_by_boxes(boxes: List[int]) -> KandidatListResponse:
        """
        Mengembalikan kandidat yang berada di box-box terpilih,
        lengkap dengan ringkasan untuk kartu UI (format SIASN).
        """
        candidates = _load_candidates()

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
                filtered.append(
                    KandidatCard(
                        nip=c.get("nip", ""),
                        nama=c.get("nama", ""),
                        nama_lengkap=c.get("nama_lengkap", ""),
                        jabatan_nama=c.get("jabatan_nama", ""),
                        jabatan_terakhir=c.get("jabatan_terakhir", ""),
                        fungsi_jabatan=c.get("fungsi_jabatan", []),
                        riwayat_jabatan=c.get("riwayat_jabatan", []),
                        riwayat_pendidikan=c.get("riwayat_pendidikan", []),
                        nilai_potensi=c.get("nilai_potensi"),
                        nilai_mansoskul=c.get("nilai_mansoskul"),
                        nilai_kinerja=c.get("nilai_kinerja"),
                        nilai_kinerja_label=c.get("nilai_kinerja_label"),
                        masa_kerja=c.get("masa_kerja"),
                        diklat_pim_level=c.get("diklat_pim_level"),
                        pengalaman_struktural_tahun=c.get("pengalaman_struktural_tahun"),
                        current_eselon_id=c.get("current_eselon_id"),
                        target_eselon_id=c.get("target_eselon_id"),
                        recommendation_label=c.get("recommendation_label"),
                        recommendation_type=c.get("recommendation_type"),
                        is_eligible=c.get("is_eligible"),
                        rhk=c.get("rhk", []),
                        posisi_nine_box_talenta=posisi,
                        box_number=box_num,
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
```

- [ ] **Step 3: Commit**

```bash
git add app/domains/pemetaan_suksesor/services/simulation.py
git commit -m "refactor: update nine-box and kandidat parsing for SIASN format

Replace nested kandidat_suksesi access with flat SIASN fields.
KandidatCard now populated from nip, nama, jabatan_nama, etc.
instead of nested id/nama/jabatan_saat_ini/unit_kerja."
```

---

### Task 5: Update `api.py` — replace `KandidatSuksesi` import and sample endpoint

**Files:**
- Modify: `app/domains/pemetaan_suksesor/api.py`

- [ ] **Step 1: Update import**

Change line 7 from:
```python
    KandidatSuksesi,
```
to:
```python
    KandidatSIASN,
```

- [ ] **Step 2: Update the sample endpoint (line 249)**

Change:
```python
    kandidat_list = [KandidatSuksesi.model_validate(c) for c in raw_candidates]
```
to:
```python
    kandidat_list = [KandidatSIASN.model_validate(c) for c in raw_candidates]
```

- [ ] **Step 3: Commit**

```bash
git add app/domains/pemetaan_suksesor/api.py
git commit -m "refactor: update api.py to use KandidatSIASN

Replace KandidatSuksesi import with KandidatSIASN and update
the sample endpoint to validate candidates.json data against
the new SIASN schema."
```

---

### Task 6: Replace `candidates.json` with SIASN-format data

**Files:**
- Modify: `app/domains/pemetaan_suksesor/dto/candidates.json`
- Delete content of: `app/domains/pemetaan_suksesor/dto/input.json` (replace with SIASN sample)

- [ ] **Step 1: Replace `candidates.json` with SIASN-format data**

The 10 candidates must be converted from nested format to flat SIASN format. Each entry changes from:

Old:
```json
{
  "kandidat_suksesi": { "id": "KANDIDAT-001", "nama": "...", ... },
  "rekam_jejak": [{ "periode": "...", "jabatan": "...", ... }],
  "sertifikasi": [{ "nama_sertifikasi": "...", ... }],
  "skp": { "tahun_2024": { ... } },
  "posisi_nine_box_talenta": "..."
}
```

New:
```json
{
  "nip": "...",
  "nama": "...",
  "nama_lengkap": "...",
  "jabatan_nama": "...",
  "fungsi_jabatan": ["...", "..."],
  "riwayat_jabatan": ["...", "..."],
  "jabatan_terakhir": "...",
  "riwayat_pendidikan": ["...", "..."],
  "nilai_potensi": ...,
  "nilai_mansoskul": ...,
  "nilai_kinerja": ...,
  "nilai_kinerja_label": "...",
  "masa_kerja": ...,
  "diklat_pim_level": "...",
  "pengalaman_struktural_tahun": "...",
  "current_eselon_id": "...",
  "target_eselon_id": "...",
  "recommendation_label": "...",
  "recommendation_type": "...",
  "is_eligible": true,
  "rhk": ["...", "..."],
  "posisi_nine_box_talenta": "..."
}
```

Convert all 10 candidates. Key field mappings:
- `kandidat_suksesi.id` → `nip` (use the ID value like "198010312005012002")
- `kandidat_suksesi.nama` → `nama` and `nama_lengkap` (same value with gelar if available)
- `kandidat_suksesi.jabatan_saat_ini` → `jabatan_nama` and `jabatan_terakhir`
- `kandidat_suksesi.unit_kerja` → omit (no direct SIASN equivalent)
- `rekam_jejak[].jabatan` → each becomes an entry in `riwayat_jabatan`
- `sertifikasi` → `diklat_pim_level` (extract the diklat entry)
- `skp` → `nilai_kinerja` (numeric) and `nilai_kinerja_label` (label)
- `posisi_nine_box_talenta` → stays the same

- [ ] **Step 2: Replace `input.json` with a single SIASN sample**

Replace the content of `input.json` with a single SIASN-format candidate (copy the first entry from the new candidates.json, but as a single object not wrapped in an array).

- [ ] **Step 3: Verify JSON is valid**

Run: `python -c "import json; json.load(open('app/domains/pemetaan_suksesor/dto/candidates.json', encoding='utf-8')); print('candidates.json OK')"`

Run: `python -c "import json; json.load(open('app/domains/pemetaan_suksesor/dto/input.json', encoding='utf-8')); print('input.json OK')"`

- [ ] **Step 4: Commit**

```bash
git add app/domains/pemetaan_suksesor/dto/candidates.json app/domains/pemetaan_suksesor/dto/input.json
git commit -m "refactor: convert candidates.json and input.json to SIASN format

Replace nested kandidat_suksesi/rekam_jejak/sertifikasi/skp structure
with flat SIASN fields: nip, nama_lengkap, jabatan_nama, riwayat_jabatan,
nilai_potensi, nilai_mansoskul, rhk, etc."
```

---

### Task 7: Verify full integration

**Files:** None (verification only)

- [ ] **Step 1: Verify all imports load**

Run:
```bash
python -c "
from app.domains.pemetaan_suksesor.dto.request import KandidatSIASN, SimulasiRequest
from app.domains.pemetaan_suksesor.dto.response import KandidatCard, SimulasiResponse
from app.domains.pemetaan_suksesor.services.simulation import SimulationService
from app.domains.pemetaan_suksesor.api import router
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 2: Verify candidates.json loads into KandidatSIASN**

Run:
```bash
python -c "
import json
from app.domains.pemetaan_suksesor.dto.request import KandidatSIASN
with open('app/domains/pemetaan_suksesor/dto/candidates.json', encoding='utf-8') as f:
    data = json.load(f)
candidates = [KandidatSIASN.model_validate(c) for c in data]
print(f'Loaded {len(candidates)} candidates OK')
print(f'First: {candidates[0].nama} ({candidates[0].nip})')
"
```

Expected: `Loaded 10 candidates OK` and a name/nip printed.

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: integration fixes after SIASN schema migration"
```

(Only if needed — skip if everything passes.)

---

## Self-Review

**1. Spec coverage:**
- ✅ `KandidatSIASN` flat model — Task 1
- ✅ `SimulasiRequest` updated — Task 1
- ✅ `KandidatCard` updated to SIASN fields — Task 2
- ✅ Old sub-models removed (`RekamJejakItem`, `SertifikasiItem`, `SkpTahunItem`, `KandidatProfil`, `RekamJejakEntry`, `SertifikasiEntry`, `SKPTahun`) — Tasks 1 & 2
- ✅ `simulation.py` type annotations and field access updated — Tasks 3 & 4
- ✅ `api.py` import updated — Task 5
- ✅ `candidates.json` and `input.json` converted — Task 6
- ✅ Nine-box and kandidat parsing updated — Task 4
- ✅ Agent prompts NOT changed in this plan — they receive serialized JSON via `model_dump()` which now outputs SIASN format automatically

**2. Placeholder scan:**
- No TBDs, TODOs, or vague steps found.

**3. Type consistency:**
- `KandidatSIASN` used consistently in `request.py`, `simulation.py`, and `api.py`
- `KandidatCard` used consistently in `response.py` and `simulation.py`
- Field names consistent: `nip`, `nama`, `nama_lengkap`, `jabatan_nama`, `jabatan_terakhir` used in both DTO and service code

**Note on agent prompts:** The agent system prompts in `core/config.py` reference the old schema fields like `rekam_jejak`, `sertifikasi`, `skp` in their output format descriptions. Since the agents now receive SIASN-format data via `model_dump()`, they will naturally see the new field names. The output format (L-Eval, C-Eval sub-tasks, etc.) remains the same — only the input data format changes. No prompt changes are strictly required, but they could be improved in a follow-up to reference SIASN fields explicitly. This is intentionally left out of scope for this plan.