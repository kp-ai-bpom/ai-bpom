from pydantic import BaseModel


class FeatureExtraction(BaseModel):
    kata_kunci_target_match: list[str]
    aktivitas_umum_match: list[str]
    sinyal_bidang_lain_dominan: list[str]
    sumber_data: list[str]


class LevelOfWorkRatio(BaseModel):
    strategis_manajerial: str
    operasional_teknis: str
    kesimpulan: str
    sumber_data: list[str]


class CumulativeExperienceGap(BaseModel):
    bidang_pengawasan_intern: str
    syarat_minimal: str
    status: str
    sumber_data: list[str]


class StructuralProximity(BaseModel):
    hop_distance: str
    same_directorate: str
    career_trajectory: str
    sumber_data: list[str]


class FunctionalOverlapScore(BaseModel):
    overlap_percentage: str
    shared_functions: list[str]
    candidate_only_functions: list[str]
    sumber_data: list[str]


class ExplainabilityMetrics(BaseModel):
    level_of_work_ratio: LevelOfWorkRatio
    stakeholder_network: list[str]
    cumulative_experience_gap: CumulativeExperienceGap
    structural_proximity: StructuralProximity
    functional_overlap_score: FunctionalOverlapScore


class MiningResults(BaseModel):
    feature_extraction: FeatureExtraction
    explainability_metrics: ExplainabilityMetrics


class FinalJustification(BaseModel):
    rekomendasi_sistem: str
    skor_domain_fit: float
    narasi_xai: str
    sumber_data: list[str]


class AlternatifJabatan(BaseModel):
    nama_jabatan: str
    skor_kesesuaian: float
    alasan: str
    gap_yang_terpenuhi: list[str]
    sumber_data: list[str]


class CandidateAnalysis(BaseModel):
    nip: str
    nama: str
    mining_results: MiningResults
    final_justification: FinalJustification
    alternatif_jabatan: list[AlternatifJabatan]


class AnalysisMetadata(BaseModel):
    session_id: str
    source_agent: str = "Analysis Agent"
    target_agent: str = "Synthesis Agent"
    status: str = "Justification_Ready"


class XAIJustificationReport(BaseModel):
    target_jabatan: str
    candidates: list[CandidateAnalysis]


class AnalysisOutput(BaseModel):
    metadata: AnalysisMetadata
    xai_justification_report: XAIJustificationReport
