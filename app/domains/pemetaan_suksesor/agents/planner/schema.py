from pydantic import BaseModel, Field


class DeskripsiJabatan(BaseModel):
    nama_jabatan: str
    atasan_langsung: str
    tugas: str
    fungsi: list[str]


class Persyaratan(BaseModel):
    model_config = {"extra": "allow"}


class ProfilJabatan(BaseModel):
    deskripsi_jabatan: DeskripsiJabatan
    persyaratan: Persyaratan
    sumber_data: list[str] = Field(
        default_factory=list,
        description='Provenance tracking: tools yang menghasilkan data ini. Contoh: ["jabatan_profile: Kepala Biro Umum"]',
    )


class SumberRegulasi(BaseModel):
    regulasi: str
    topik: str
    sumber_data: list[str] = Field(
        default_factory=list,
        description='Provenance tracking: tools yang menghasilkan regulasi ini. Contoh: ["neo4j: Peraturan BPOM Nomor 2 Tahun 2024"]',
    )


class KomponenPenilaian(BaseModel):
    komponen: str
    bobot: str


class KriteriaRekamJejak(BaseModel):
    nama: str
    bobot: str
    skala: list[str]


class RekamJejak(BaseModel):
    kriteria: list[KriteriaRekamJejak]


class PersyaratanJPT(BaseModel):
    jenis: str = Field(description="JPT Pratama atau JPT Madya")
    pangkat_minimal: str
    pendidikan_minimal: str
    pengalaman_jabatan: str
    diklat: str
    usia_maksimal: str


class SeleksiTerbukaJPT(BaseModel):
    komponen_penilaian: list[KomponenPenilaian]
    rekam_jejak: RekamJejak
    persyaratan_jpt: PersyaratanJPT


class KomponenEvaluasi(BaseModel):
    nama: str
    bobot: str
    detail: str


class KategoriEvaluasi(BaseModel):
    nama: str
    rentang_nilai: str


class EvaluasiTalenta(BaseModel):
    komponen: list[KomponenEvaluasi]
    kategori: list[KategoriEvaluasi]


class Pemetaan9Box(BaseModel):
    sumbu_y: str
    sumbu_x: str
    prioritas_suksesor: list[str]


class ManajemenTalenta(BaseModel):
    evaluasi_talenta: EvaluasiTalenta
    pemetaan_9box: Pemetaan9Box


class AturanPenilaian(BaseModel):
    sumber_regulasi: list[SumberRegulasi]
    seleksi_terbuka_jpt: SeleksiTerbukaJPT
    manajemen_talenta: ManajemenTalenta
    sumber_data: list[str] = Field(
        default_factory=list,
        description='Provenance tracking: KG entities yang menghasilkan aturan ini. Contoh: ["neo4j: Seleksi Terbuka JPT", "neo4j: Penilaian Rekam Jejak"]',
    )


class ProfilJabatanKandidat(BaseModel):
    """Profil jabatan posisi saat ini kandidat, dari query_jabatan_profile."""

    model_config = {"extra": "allow"}

    nama_jabatan: str
    atasan_langsung: str | None = None
    data: dict


class CandidateData(BaseModel):
    """Raw candidate data dari input.json — pass-through untuk mencegah halusinasi."""

    model_config = {"extra": "allow"}
    nip: str
    nama: str
    nama_lengkap: str
    jabatan_nama: str
    jabatan_terakhir: str
    fungsi_jabatan: list[str]
    riwayat_jabatan: list[str]
    riwayat_pendidikan: list[str]
    nilai_potensi: float
    nilai_mansoskul: int
    nilai_kinerja: int
    nilai_kinerja_label: str
    masa_kerja: int
    masa_kerja_total_tahun: int | None = None
    pengalaman_struktural_tahun: str
    diklat_pim_level: str | None = None
    jenjang_pendidikan_id: str
    current_eselon_id: str | None = None
    target_eselon_id: str
    recommendation_label: str
    recommendation_type: str
    is_eligible: bool
    rhk: list[str]
    pool: int | None = None
    pool_id: int | None = None
    foto_url: str | None = None
    sumber_data: list[str] = Field(
        default_factory=list,
        description='Provenance tracking: sumber data kandidat. Selalu: ["input_json"]',
    )


class RetrievedContext(BaseModel):
    syarat_pengalaman: str
    fungsi_utama: list[str]
    kompetensi_spesifik: list[str]
    profil_jabatan_kandidat: ProfilJabatanKandidat | None = None
    relevansi_riwayat: str | None = None
    sumber_data: list[str] = Field(
        default_factory=list,
        description='Provenance tracking: tools yang menghasilkan konteks ini. Contoh: ["jabatan_profile: Kepala Biro Umum", "jabatan_profile: Analis SDM", "neo4j: Seleksi Terbuka JPT", "input_json"]',
    )


class AnalyzingTask(BaseModel):
    task_id: str
    task_type: str
    instruction: str


class CandidateBlueprint(BaseModel):
    nip: str
    nama: str
    candidate_data: CandidateData
    retrieved_context: RetrievedContext
    analyzing_task: list[AnalyzingTask]


class Metadata(BaseModel):
    session_id: str
    source_agent: str = "Planner Agent"
    target_agent: str = "Orchestrator Agent"
    status: str = "Blueprint_Generated"


class XAIBlueprint(BaseModel):
    target_jabatan: str
    profil_jabatan: ProfilJabatan
    aturan_penilaian: AturanPenilaian
    candidates: list[CandidateBlueprint]


class PlannerOutput(BaseModel):
    metadata: Metadata
    xai_blueprint: XAIBlueprint
