"""
Agent module untuk multi-agent architecture menggunakan strands-agents.

Module ini menyediakan:
- AgentAdapter: Dataclass untuk menyimpan semua instance Agent
- AgentManager: Singleton untuk mengelola lifecycle Agent
- init_agents: Dependency injection factory
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from strands import Agent
from strands.models.openai import OpenAIModel
from strands.multiagent import Swarm

from app.core.config import settings
from app.core.logger import log

from .config import settings as local_settings


@dataclass
class AgentAdapter:
    """
    Adapter yang menyimpan semua instance Agent default.
    Dapat di-inject ke service/dependency manapun yang membutuhkan multi-agent.
    """

    orchestrator: Agent
    search: Agent
    analysis: Agent
    synthesis: Agent
    reviewer: Agent

    def get(self, name: str) -> Optional[Agent]:
        """Get agent by name from adapter."""
        return getattr(self, name, None)


# Default system prompts for each agent type — Pemetaan Suksesor flow
DEFAULT_PROMPTS = {
    "orchestrator": """Kamu adalah Planner Agent untuk Pemetaan Suksesor JPT BPOM.
Tugasmu menerima jabatan target dan memecah persyaratan menjadi sub-tugas evaluasi spesifik
berdasarkan regulasi PerBPOM No. 21 Tahun 2020 dan KepKabadan 322/2023.

Tahap 1 - DECOMPOSITION: Hasilkan 5 sub-tugas evaluasi:
1. Evaluasi pengalaman Jabatan Administrator (Eselon III) atau Jabatan Fungsional Ahli Madya minimal 2 tahun
2. Evaluasi pengalaman jabatan di bidang tugas terkait secara kumulatif minimal 5 tahun
3. Pencocokan semantik fungsional antara deskripsi tugas kandidat dengan fungsi jabatan target
4. Evaluasi kinerja (SKP) bernilai baik dalam 2 tahun terakhir dan posisi Kotak Manajemen Talenta (Prioritas Kotak 7,8,9)
5. Evaluasi syarat tambahan/diutamakan (Diklat PIM dan Kemampuan Bahasa Inggris)

WAJIB output JSON dengan format:
{
  "target_jabatan": "<nama jabatan>",
  "sub_tasks": [
    {"id": 1, "nama": "...", "syarat_mutlak": true/false, "bobot": 0-100, "deskripsi": "..."},
    ...
  ]
}""",
    "search": """Kamu adalah Extractor Agent untuk Pemetaan Suksesor JPT BPOM.
Tahap 2 - RETRIEVAL & EXTRACTION: Untuk setiap sub-tugas evaluasi, ekstrak informasi esensial dari data JSON kandidat.

Sumber data kandidat:
- rekam_jejak: riwayat jabatan, durasi, deskripsi tugas & fungsi
- sertifikasi: diklat, sertifikasi kompetensi, kemampuan bahasa
- skp: rating hasil kerja, rating perilaku, posisi nine-box talenta

Tugas: Saring fakta-fakta yang relevan untuk setiap sub-tugas evaluasi.
Tangkap frasa-frasa kunci (misal: "katalisator perubahan", "QAIP", "audit berbasis risiko")
yang memiliki kecocokan semantik dengan fungsi jabatan target.

WAJIB output JSON dengan format:
{
  "id_kandidat": "<id>",
  "extractions": [
    {"sub_task_id": 1, "fakta": "...", "sumber": "rekam_jejak/sertifikasi/skp"},
    ...
  ]
}""",
    "analysis": """Kamu adalah Analysis Agent untuk Pemetaan Suksesor JPT BPOM.
Tahap 3 - VALIDATION / AUDIT: Lakukan dua jenis evaluasi:

1. Logical Evaluation (L-Eval): Verifikasi setiap kriteria secara logika.
   - Apakah pengalaman total ≥ syarat mutlak?
   - Apakah durasi jabatan Eselon III/Madya ≥ 2 tahun?
   - Apakah Kotak Talenta termasuk prioritas (7,8,9)?
   Keputusan: ACCEPT (Valid) atau REJECT (Tidak Valid)

2. Counterfactual Evaluation (C-Eval): Uji sanggah dengan asumsi "Kandidat TIDAK MEMENUHI syarat".
   - Cari kontradiksi di data JSON yang membatalkan asumsi tersebut
   - Jika tidak ditemukan kontradiksi → ACCEPT (No Contradiction)
   - Jika ditemukan bukti penggugur → REJECT (Contradiction Found)

PENTING untuk field keterangan: SELALU kutip bukti spesifik dari data kandidat.
Contoh keterangan yang baik:
- "Valid. Menjabat Kepala Bagian TU (Eselon III) selama 4 tahun (2022-2026) dan Auditor Ahli Madya 4 tahun (2018-2022), total 8 tahun melebihi minimum 2 tahun."
- "Sangat Relevan. Deskripsi tugas memuat 'katalisator perubahan', 'QAIP', dan 'pengendalian intern' yang cocok dengan fungsi Inspektur I."
- "Valid. SKP 2024 dan 2025 Di Atas Ekspektasi. Posisi Kotak 9 termasuk prioritas promosi."
JANGAN tulis keterangan generik seperti "Memenuhi syarat" — selalu sertakan data spesifik.

WAJIB output JSON dengan format:
{
  "id_kandidat": "<id>",
  "nama": "<nama lengkap>",
  "jabatan_saat_ini": "<jabatan saat ini>",
  "l_eval": {"keputusan": "ACCEPT/REJECT", "alasan": "..."},
  "c_eval": {"keputusan": "ACCEPT/REJECT", "bukti_kontradiksi": "..."},
  "acceptances": 0-2,
  "detail_evaluasi": {
    "pengalaman": {"status": "Valid/Tidak Valid", "keterangan": "<kutip bukti spesifik dari data>"},
    "fungsi_semantik": {"status": "Sangat Relevan/Relevan/Kurang Relevan", "keterangan": "<sebutkan frasa yang cocok>"},
    "kinerja_talenta": {"status": "Valid/Tidak Valid", "keterangan": "<sebutkan rating SKP dan posisi nine-box>"},
    "kualifikasi_tambahan": {"status": "Valid/Tidak Valid", "keterangan": "<sebutkan sertifikasi dan tahun>"}
  }
}""",
    "synthesis": """Kamu adalah Synthesis Agent untuk Pemetaan Suksesor JPT BPOM.
Tahap 4 - CONFIDENCE UPDATER & SCORING: Gabungkan semua hasil evaluasi menjadi skor dan peringkat.

Untuk setiap kandidat, tentukan:
- Skor Kesesuaian (0-100): Berdasarkan kecocokan dengan seluruh kriteria
- Kategori Kesiapan: SUKSESOR (skor ≥ 80), POTENSIAL (skor 50-79), BELUM SIAP (skor < 50)
- Tingkat Keyakinan: Tinggi (2 Acceptances), Sedang (1 Acceptance), Rendah (0 Acceptances)
- Kesimpulan: Clear (2 Acceptances) atau Review Needed (< 2 Acceptances)
- Alasan Penilaian: Narasi detail mengapa skor tersebut diberikan, disertai bukti dukung spesifik dari data kandidat

PENTING untuk alasan_penilaian:
- Kutip bukti konkret dari data kandidat (nama jabatan, durasi, sertifikasi, skor SKP, posisi nine-box)
- Jelaskan kontribusi setiap aspek terhadap skor (pengalaman, fungsi, kinerja, kualifikasi)
- Bandingkan kekuatan dan kelemahan kandidat secara spesifik
- Contoh alasan yang baik: "Memenuhi syarat mutlak pengalaman Eselon III selama 4 tahun (Kepala Bagian TU, 2022-2026) dan Ahli Madya 4 tahun (2018-2022), total 8 tahun melebihi minimum 5 tahun. Fungsi katalisator perubahan dan QAIP sangat relevan. SKP Di Atas Ekspektasi 2 tahun berturut-turut dan Kotak 9. Diklat PIM III dan TOEFL 580 memenuhi syarat tambahan."

Urutkan kandidat berdasarkan Skor Kesesuaian dari tertinggi ke terendah.

WAJIB output JSON dengan format:
{
  "target_jabatan": "<nama jabatan>",
  "peringkat": [
    {
      "rank": 1,
      "id_kandidat": "<id>",
      "nama": "<nama>",
      "jabatan_saat_ini": "<jabatan saat ini>",
      "skor_kesesuaian": 0-100,
      "kategori_kesiapan": "SUKSESOR/POTENSIAL/BELUM SIAP",
      "confidence_level": "Tinggi/Sedang/Rendah",
      "acceptances": 0-2,
      "kesimpulan": "Clear/Review Needed",
      "alasan_penilaian": "<narasi detail dengan bukti dukung>",
      "detail_evaluasi": {
        "pengalaman": {"status": "...", "keterangan": "..."},
        "fungsi_semantik": {"status": "...", "keterangan": "..."},
        "kinerja_talenta": {"status": "...", "keterangan": "..."},
        "kualifikasi_tambahan": {"status": "...", "keterangan": "..."}
      }
    },
    ...
  ]
}""",
    "reviewer": """Kamu adalah Reviewer Agent untuk Pemetaan Suksesor JPT BPOM.
Tugas: Validasi output akhir, pastikan semua kandidat dievaluasi secara adil dan konsisten.

Periksa:
1. Semua kriteria evaluasi diterapkan secara konsisten pada setiap kandidat
2. Tidak ada kandidat yang diuntungkan atau dirugikan secara tidak adil
3. Skor akurat mencerminkan hasil evaluasi
4. Pemilihan top 5 dapat dipertanggungjawabkan
5. Format output sesuai standar

WAJIB output JSON dengan format:
{
  "valid": true/false,
  "catatan": "...",
  "rekomendasi": "..."
}""",
}

# Model tier mappings from config
MODEL_TIER_CONFIG = {
    "orchestrator": "AGENT_ORCHESTRATOR_MODEL",
    "search": "AGENT_SEARCH_MODEL",
    "analysis": "AGENT_ANALYSIS_MODEL",
    "synthesis": "AGENT_SYNTHESIS_MODEL",
    "reviewer": "AGENT_REVIEWER_MODEL",
}


class AgentManager:
    """
    Singleton class untuk mengelola instance Strands Agent.
    Memastikan hanya ada satu instance Agent di memori selama aplikasi berjalan.
    """

    _instance = None
    _agents: Dict[str, Agent] = {}

    def __new__(cls):
        """Override __new__ untuk memastikan hanya satu instance AgentManager yang dibuat."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._agents = {}
        return cls._instance

    def _create_model(self, model_tier: str) -> OpenAIModel:
        """
        Create Strands OpenAI model based on tier.
        Uses existing settings for API keys and base URLs.
        """
        model = OpenAIModel(
            client_args={
                "api_key": settings.OPENAI_API_KEY,
                "base_url": settings.AI_BASE_URL,
            }
            if settings.AI_BASE_URL
            else {"api_key": settings.OPENAI_API_KEY},
            model_id=settings.AI_INSTRUCT_MODEL_NAME
            if model_tier == "instruct"
            else settings.AI_THINK_MODEL_NAME
            if model_tier == "think"
            else settings.AI_DEEP_THINK_MODEL_NAME,
            # params={"temperature": 0.7},
        )
        return model

    def _get_model_tier(self, agent_name: str) -> str:
        """Get model tier for agent from local config."""
        config_key = MODEL_TIER_CONFIG.get(agent_name, "AGENT_ORCHESTRATOR_MODEL")
        return getattr(local_settings, config_key, "think")

    def _create_agent(
        self,
        name: str,
        model_tier: str,
        system_prompt: str,
        tools: Optional[List] = None,
    ) -> Agent:
        """
        Create a single agent with configuration.
        """
        model = self._create_model(model_tier)

        agent = Agent(
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=tools or [],
        )

        model_name = (
            settings.AI_INSTRUCT_MODEL_NAME
            if model_tier == "instruct"
            else settings.AI_THINK_MODEL_NAME
            if model_tier == "think"
            else settings.AI_DEEP_THINK_MODEL_NAME
        )
        log.info(
            f"🤖 Agent '{name}' initialized — tier: {model_tier}, model: {model_name}"
        )
        return agent

    def _initialize_default_agents(self):
        """
        Initialize 5 default BPOM agents.
        Called lazily when first agent is requested.
        """
        if self._agents:
            return

        default_agents = ["orchestrator", "search", "analysis", "synthesis", "reviewer"]

        for agent_name in default_agents:
            model_tier = self._get_model_tier(agent_name)
            system_prompt = DEFAULT_PROMPTS[agent_name]

            self._agents[agent_name] = self._create_agent(
                name=agent_name,
                model_tier=model_tier,
                system_prompt=system_prompt,
                tools=[],
            )

        log.info("🚀 All default agents initialized")

    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Get agent by name.
        Initializes default agents if not yet created.
        """
        self._initialize_default_agents()
        return self._agents.get(name)

    def register_agent(
        self,
        name: str,
        model_tier: str,
        system_prompt: str,
        tools: Optional[List] = None,
    ) -> Agent:
        """
        Register custom agent dynamically.
        """
        agent = self._create_agent(
            name=name,
            model_tier=model_tier,
            system_prompt=system_prompt,
            tools=tools or [],
        )

        self._agents[name] = agent
        log.info(f"✅ Custom agent '{name}' registered")
        return agent

    def get_adapter(self) -> AgentAdapter:
        """
        Get AgentAdapter with all default agents.
        """
        self._initialize_default_agents()

        return AgentAdapter(
            orchestrator=self._agents["orchestrator"],
            search=self._agents["search"],
            analysis=self._agents["analysis"],
            synthesis=self._agents["synthesis"],
            reviewer=self._agents["reviewer"],
        )

    def create_swarm(
        self,
        agent_names: List[str],
        entry_point: str = "orchestrator",
    ) -> Swarm:
        """
        Create a Swarm for multi-agent orchestration.

        Args:
            agent_names: List of agent names to include in the swarm
            entry_point: Starting agent name

        Returns:
            Swarm instance for orchestrated execution
        """
        self._initialize_default_agents()

        agents = [self._agents[name] for name in agent_names if name in self._agents]

        entry_agent = self._agents.get(entry_point, agents[0] if agents else None)

        swarm = Swarm(
            agents,
            entry_point=entry_agent,
            max_handoffs=local_settings.SWARM_MAX_HANDOFFS,
            max_iterations=local_settings.SWARM_MAX_ITERATIONS,
            execution_timeout=local_settings.SWARM_EXECUTION_TIMEOUT,
        )

        log.info(f"🐝 Swarm created with {len(agents)} agents, entry: {entry_point}")
        return swarm


# Dependency Injection Factory
def init_agents() -> AgentAdapter:
    """
    Dependency Injection untuk mendapatkan semua instance Agent.
    Bisa disuntikkan ke service mana pun yang membutuhkan multi-agent.
    """
    manager = AgentManager()
    return manager.get_adapter()
