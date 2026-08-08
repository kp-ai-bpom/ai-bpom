from pydantic import BaseModel


class ConsistencyFlag(BaseModel):
    tipe: str
    deskripsi: str
    kandidat_terkait: list[str]
    sumber_data: list[str]


class EvidenceAuditItem(BaseModel):
    klaim: str
    sumber_data_terverifikasi: bool
    referensi: list[str]
    catatan: str


class FinalXAINarrative(BaseModel):
    narasi_pipeline: str
    justifikasi_rekomendasi: str
    alternatif_dipertimbangkan: str
    sumber_data: list[str]


class BiasLimitation(BaseModel):
    tipe: str
    deskripsi: str
    dampak: str
    rekomendasi_mitigasi: str


class ReviewerMetadata(BaseModel):
    session_id: str
    source_agent: str = "Reviewer Agent"
    target_agent: str = "User"
    status: str = "Final_XAI_Completed"


class ReviewerOutput(BaseModel):
    metadata: ReviewerMetadata
    target_jabatan: str
    consistency_flags: list[ConsistencyFlag]
    evidence_audit: list[EvidenceAuditItem]
    final_xai_narrative: FinalXAINarrative
    bias_limitations: list[BiasLimitation]
