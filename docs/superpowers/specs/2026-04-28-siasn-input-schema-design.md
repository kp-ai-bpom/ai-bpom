# SIASN Input Schema Migration Design

## Goal

Replace the existing `KandidatSuksesi` schema (nested objects with `rekam_jejak`, `sertifikasi`, `skp`) with a flat SIASN-format schema that matches `req.json`. Update all endpoints, agent prompts, and sample data.

## Scope

All endpoints in the pemetaan_suksesor domain:
- Simulation (POST `/simulasi`)
- Nine-box grid (GET `/nine-box`)
- Kandidat list by boxes (GET `/kandidat`)
- Matching history (no change — it stores simulation results, not kandidat data)

## New Schema

### `KandidatSIASN` (replaces `KandidatSuksesi` + sub-models)

```python
class KandidatSIASN(BaseModel):
    nip: str
    nama: str
    nama_lengkap: str
    jabatan_nama: str
    foto_url: Optional[str] = None
    pool: Optional[int] = None
    pool_id: Optional[int] = None
    fungsi_jabatan: List[str] = []
    riwayat_jabatan: List[str] = []
    jabatan_terakhir: str
    riwayat_pendidikan: List[str] = []
    nilai_potensi: Optional[float] = None
    nilai_mansoskul: Optional[int] = None
    nilai_kinerja: Optional[float] = None
    nilai_kinerja_label: Optional[str] = None
    masa_kerja: Optional[int] = None
    masa_kerja_total_tahun: Optional[int] = None
    diklat_pim_level: Optional[str] = None
    jenjang_pendidikan_id: Optional[str] = None
    pengalaman_struktural_tahun: Optional[str] = None
    current_eselon_id: Optional[str] = None
    target_eselon_id: Optional[str] = None
    recommendation_label: Optional[str] = None
    recommendation_type: Optional[str] = None
    is_eligible: Optional[bool] = None
    rhk: List[str] = []
```

### `SimulasiRequest` (updated)

```python
class SimulasiRequest(BaseModel):
    target_jabatan: str = Field(..., description="Jabatan target suksesi")
    kandidat: List[KandidatSIASN] = Field(..., min_length=1, max_length=50, description="Daftar kandidat SIASN")
```

## Files Changed

### 1. `dto/request.py`
- Remove: `KandidatProfil`, `RekamJejakEntry`, `SertifikasiEntry`, `SKPTahun`, `KandidatSuksesi`
- Add: `KandidatSIASN` (flat model matching `req.json`)
- Update: `SimulasiRequest.kandidat` field type from `List[KandidatSuksesi]` to `List[KandidatSIASN]`

### 2. `dto/response.py`
- Update `KandidatCard` to use SIASN fields:
  - `id` → `nip`
  - `nama` stays (from `nama`)
  - `jabatan_saat_ini` → `jabatan_nama`
  - `unit_kerja` → derived from `fungsi_jabatan[0]` or empty
  - Remove `rekam_jejak` (structured), add `riwayat_jabatan` (list of strings)
  - Remove `sertifikasi` (structured), add `diklat_pim_level` (string)
  - Remove `skp` (structured), add `nilai_kinerja`/`nilai_kinerja_label`
  - Add new fields: `nilai_potensi`, `nilai_mansoskul`, `masa_kerja`, `pengalaman_struktural_tahun`, `rhk`, `recommendation_label`, `is_eligible`

New `KandidatCard`:
```python
class KandidatCard(BaseModel):
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

Remove: `RekamJejakItem`, `SertifikasiItem`, `SkpTahunItem`

Update `KandidatListData.kandidat` type to `List[KandidatCard]` (stays same field name, new type)

### 3. `services/simulation.py`

**`_evaluate_candidate`**: Change `kandidat.model_dump(mode="json")` serialization. The prompt now sends SIASN data instead of nested objects. Update prompt text to reference SIASN fields.

**`get_kandidat_by_boxes`**: Update to parse SIASN-format `candidates.json`. Map fields:
- `c.get("kandidat_suksesi", {}).get("nama", "")` → `c.get("nama", "")`
- `c.get("kandidat_suksesi", {}).get("id", "")` → `c.get("nip", "")`
- `c.get("kandidat_suksesi", {}).get("jabatan_saat_ini", "")` → `c.get("jabatan_nama", "")`
- `c.get("kandidat_suksesi", {}).get("unit_kerja", "")` → `c.get("fungsi_jabatan", [""])[0]` or empty string

Replace `RekamJejakItem`, `SertifikasiItem`, `SkpTahunItem` construction with direct field access from SIASN dict.

**`get_nine_box_data`**: Update `c.get("kandidat_suksesi", {}).get("nama", "")` → `c.get("nama", "")`

**`run()` method**: Change `kandidat.kandidat_suksesi.id` → `kandidat.nip`, `kandidat.kandidat_suksesi.nama` → `kandidat.nama`, `kandidat.kandidat_suksesi.jabatan_saat_ini` → `kandidat.jabatan_nama`

### 4. `services/helpers.py`
- `_load_candidates()`: No structural change needed — it just loads JSON. The format of the JSON changes.

### 5. `dto/candidates.json`
- Replace with SIASN-format data. Convert each entry from nested `kandidat_suksesi.id/nama/jabatan_saat_ini/unit_kerja` to flat `nip/nama/nama_lengkap/jabatan_nama`. Convert `rekam_jejak` (list of objects) to `riwayat_jabatan` (list of strings). Convert `sertifikasi`/`skp` to SIASN fields.

### 6. `dto/input.json`
- Remove or replace with SIASN-format sample.

### 7. `core/config.py`
- Update agent system prompts (Analysis, Synthesis, Orchestrator) to reference SIASN fields instead of old nested structure. Key changes:
  - References to `rekam_jejak` with `periode/jabatan/durasi_tahun/deskripsi_tugas_dan_fungsi` → `riwayat_jabatan` (list of strings)
  - References to `skp` with `rating_hasil_kerja/rating_perilaku_kerja` → `nilai_kinerja`/`nilai_kinerja_label`
  - References to `sertifikasi` with `nama_sertifikasi/tahun/keterangan` → `diklat_pim_level`
  - New fields to mention: `nilai_potensi`, `nilai_mansoskul`, `masa_kerja`, `pengalaman_struktural_tahun`, `rhk`, `is_eligible`

## Data Mapping

| Old Field | New SIASN Field | Notes |
|-----------|----------------|-------|
| `kandidat_suksesi.id` | `nip` | NIP as identifier |
| `kandidat_suksesi.nama` | `nama` | Short name |
| `kandidat_suksesi.jabatan_saat_ini` | `jabatan_nama` | Current position |
| `kandidat_suksesi.unit_kerja` | `fungsi_jabatan[0]` | Derived from functional positions |
| `rekam_jejak[{periode,jabatan,durasi,deskripsi}]` | `riwayat_jabatan: [str]` | Flat string list |
| `sertifikasi[{nama,tahun,keterangan}]` | `diklat_pim_level` | Just the training level |
| `skp[{rating_hasil_kerja,rating_perilaku_kerja,keterangan}]` | `nilai_kinerja`, `nilai_kinerja_label` | Numeric + label |
| `posisi_nine_box_talenta` | `posisi_nine_box_talenta` | Kept as-is |
| N/A | `nilai_potensi` | New: potential score |
| N/A | `nilai_mansoskul` | New: social competence score |
| N/A | `masa_kerja` | New: years of service |
| N/A | `pengalaman_struktural_tahun` | New: structural experience years |
| N/A | `rhk` | New: Rencana Hasil Kerja |
| N/A | `is_eligible` | New: eligibility flag |
| N/A | `recommendation_label` | New: recommendation |

## Edge Cases

- **Optional fields**: Most SIASN fields are optional (`None`/empty list). Agent prompts should handle missing data gracefully.
- **`fungsi_jabatan` as unit_kerja proxy**: The old `unit_kerja` doesn't have a direct SIASN equivalent. Use `fungsi_jabatan[0]` if available, else empty string.
- **`pengalaman_struktural_tahun` is string**: It's `"7.64"` not `7.64` — parse as float in agent prompts or keep as string.