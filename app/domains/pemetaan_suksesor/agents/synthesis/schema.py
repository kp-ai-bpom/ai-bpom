from pydantic import BaseModel


class SynthesisMetadata(BaseModel):
    session_id: str
    source_agent: str = "Synthesis Agent"
    target_agent: str = "Reviewer Agent"
    status: str = "Synthesis_Completed"


class BenchmarkRow(BaseModel):
    nip: str
    nama: str
    skor_domain_fit: float
    skor_regulatory_compliance: float
    skor_structural_proximity: float
    skor_komposit: float
    ranking: int
    kekuatan: list[str]
    kelemahan: list[str]
    rekomendasi_sistem: str
    sumber_data: list[str]


class SystemicGap(BaseModel):
    gap: str
    kandidat_terdampak: list[str]
    rekomendasi_mitigasi: str
    sumber_data: list[str]


class ComplianceCheck(BaseModel):
    kandidat: str
    nip: str
    persyaratan: str
    status: str
    detail: str
    sumber_data: list[str]


class ExecutiveSummary(BaseModel):
    narasi: str
    temuan_utama: list[str]
    saran_tindak_lanjut: list[str]
    sumber_data: list[str]


class SynthesisOutput(BaseModel):
    metadata: SynthesisMetadata
    target_jabatan: str
    comparison_matrix: list[BenchmarkRow]
    systemic_gaps: list[SystemicGap]
    regulatory_compliance: list[ComplianceCheck]
    executive_summary: ExecutiveSummary
