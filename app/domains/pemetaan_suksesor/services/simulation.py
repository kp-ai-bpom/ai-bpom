import asyncio
import json
from typing import Any, Dict, List

from fastapi import HTTPException, status

from app.core.logger import log

from ..core.agent import AgentAdapter, init_agents
from ..dto.request import KandidatSIASN
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
from ..core.config import settings as local_settings
from .helpers import _extract_json, _load_candidates, _parse_box_number, _run_agent_async


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
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    async def _run_and_track(self, agent, prompt: str) -> str:
        """Run agent, accumulate token usage, return text output."""
        result = await _run_agent_async(agent, prompt)
        if result is None:
            return ""
        usage = result.metrics.accumulated_usage
        self._total_input_tokens += usage.get("inputTokens", 0)
        self._total_output_tokens += usage.get("outputTokens", 0)
        return str(result)

    # ── Public API ────────────────────────────────────────────────

    async def run(
        self,
        target_jabatan: str,
        kandidat_list: List[KandidatSIASN],
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
        agent_queue: asyncio.Queue = asyncio.Queue()
        for a in self._agents.analysis_pool:
            agent_queue.put_nowait(a)

        async def _eval_one(kandidat: KandidatSIASN) -> Dict:
            kandidat_id = kandidat.nip
            kandidat_nama = kandidat.nama

            # Tahap 2: Search agent retrieves RAG context
            search_result = await self._search_with_rag(
                kandidat, target_jabatan, sub_tasks
            )

            agent = await agent_queue.get()
            try:
                log.info(
                    f"🔍 [paralel] Mengevaluasi {kandidat_nama} ({kandidat_id})..."
                )
                evaluation = await self._evaluate_candidate(
                    kandidat, target_jabatan, sub_tasks, agent=agent,
                    search_result=search_result,
                )
                evaluation.setdefault(
                    "jabatan_saat_ini", kandidat.jabatan_nama
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
            f"✅ Tahap 2+3 selesai (RAG-enhanced) — {len(evaluation_results)} kandidat dievaluasi "
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
            input_token=f"{self._total_input_tokens} token",
            output_token=f"{self._total_output_tokens} token",
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

        box_candidates: Dict[int, List[str]] = {i: [] for i in range(1, 10)}
        for c in candidates:
            posisi = c.get("posisi_nine_box_talenta", "")
            box_num = _parse_box_number(posisi)
            if box_num and 1 <= box_num <= 9:
                nama = c.get("nama", "")
                if nama:
                    box_candidates[box_num].append(nama)

        boxes = []
        for i in range(1, 10):
            defn = local_settings.NINE_BOX_DEFINITIONS[i]
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

    # ── Rules Loader (hardcoded — nanti diganti Hybrid RAG) ─────────

    @staticmethod
    def _load_all_jabatan_rules() -> List[Dict]:
        """
        Load semua aturan jabatan dari folder jabatan_rules/.
        Setiap file JSON berisi data satu jabatan (deskripsi + persyaratan).
        """
        import app.domains.pemetaan_suksesor.core.config as _c

        if _c._jabatan_rules_cache is not None:
            assert _c._jabatan_rules_cache is not None
            return _c._jabatan_rules_cache

        rules_dir = local_settings.JABATAN_RULES_DIR
        if not rules_dir.is_dir():
            log.warning(f"⚠️ Folder jabatan_rules tidak ditemukan: {rules_dir}")
            _c._jabatan_rules_cache = []
            return []

        entries: List[Dict] = []
        for json_file in sorted(rules_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "deskripsi_jabatan" in data:
                    entries.append(data)
                else:
                    log.warning(f"⚠️ {json_file.name}: format tidak valid (tanpa deskripsi_jabatan)")
            except json.JSONDecodeError:
                log.warning(f"⚠️ {json_file.name}: JSON tidak valid")

        _c._jabatan_rules_cache = entries
        log.info(f"📋 Jabatan rules loaded — {len(entries)} posisi dari {rules_dir}")
        return entries

    @staticmethod
    def _load_jabatan_rules(target_jabatan: str) -> Dict | None:
        """
        Cari aturan jabatan berdasarkan nama jabatan.
        Mengembalikan seluruh data (deskripsi + persyaratan) jika ditemukan.
        """
        all_rules = SimulationService._load_all_jabatan_rules()
        normalized = target_jabatan.lower().strip()
        for entry in all_rules:
            nama = entry.get("deskripsi_jabatan", {}).get("nama_jabatan", "")
            if nama.lower().strip() == normalized:
                return entry
        return None

    @staticmethod
    def list_available_jabatan() -> List[str]:
        """Return daftar nama jabatan yang tersedia di folder jabatan_rules."""
        all_rules = SimulationService._load_all_jabatan_rules()
        return [
            entry.get("deskripsi_jabatan", {}).get("nama_jabatan", "")
            for entry in all_rules
            if entry.get("deskripsi_jabatan", {}).get("nama_jabatan")
        ]

    # ── Tahap 1: Decomposition ────────────────────────────────────

    async def _decompose(self, target_jabatan: str) -> List[Dict]:
        """
        Orchestrator agent menerima aturan jabatan dan menghasilkan sub-tugas evaluasi.
        Sub-tugas di-generate oleh agent, bukan hardcoded.
        Source: (1) jabatan_rules.json → (2) LLM knowledge jika tidak ada rules
        """
        rules = self._load_jabatan_rules(target_jabatan)

        if rules:
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

        raw = await self._run_and_track(self._agents.orchestrator, prompt)
        parsed = _extract_json(raw)

        if parsed and isinstance(parsed, dict) and "sub_tasks" in parsed:
            return parsed["sub_tasks"]

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

    # ── Tahap 2: Search with RAG ──────────────────────────────────

    async def _search_with_rag(
        self,
        kandidat: KandidatSIASN,
        target_jabatan: str,
        sub_tasks: List[Dict],
    ) -> Dict:
        """Search agent retrieves RAG context and extracts candidate info per sub-task."""
        kandidat_json = kandidat.model_dump(mode="json")
        prompt = (
            f"Jabatan Target: {target_jabatan}\n"
            f"Data Kandidat:\n```json\n{json.dumps(kandidat_json, ensure_ascii=False, indent=2)}\n```\n\n"
            f"Sub-Tugas Evaluasi:\n```json\n{json.dumps(sub_tasks, ensure_ascii=False, indent=2)}\n```\n\n"
            "Untuk setiap sub-tugas, tentukan apakah perlu konteks RAG tentang jabatan target.\n"
            "Jika ya, gunakan tool RAG yang tepat lalu ekstrak info relevan dari data kandidat.\n"
            "Output WAJIB JSON sesuai format di system prompt Search Agent."
        )
        raw = await self._run_and_track(self._agents.search, prompt)
        parsed = _extract_json(raw)

        if parsed and isinstance(parsed, dict):
            parsed.setdefault("id_kandidat", kandidat.nip)
            return parsed

        log.warning(f"⚠️ Search agent fallback untuk {kandidat.nip}")
        return {
            "id_kandidat": kandidat.nip,
            "extractions": [],
            "rag_context": {},
        }

    # ── Tahap 2+3: Evaluate single candidate ──────────────────────

    async def _evaluate_candidate(
        self,
        kandidat: KandidatSIASN,
        target_jabatan: str,
        sub_tasks: List[Dict],
        agent: Any = None,
        search_result: Dict | None = None,
    ) -> Dict:
        """Search (extract) + Analysis (L-Eval + C-Eval) untuk satu kandidat."""
        eval_agent = agent or self._agents.analysis
        kandidat_json = kandidat.model_dump(mode="json")

        rules = self._load_jabatan_rules(target_jabatan)
        context_extra = ""
        if rules:
            deskripsi = rules.get("deskripsi_jabatan", {})
            fungsi = deskripsi.get("fungsi", [])
            if fungsi:
                context_extra += f"\nFungsi Jabatan {target_jabatan}:\n"
                for i, f in enumerate(fungsi, 1):
                    context_extra += f"  {i}. {f}\n"

            pengalaman = rules.get("persyaratan", {}).get("pengalaman_bidang_tugas", {})
            kompetensi = pengalaman.get("kompetensi_spesifik", [])
            if kompetensi:
                context_extra += "\nKompetensi Spesifik:\n"
                for i, k in enumerate(kompetensi, 1):
                    context_extra += f"  {i}. {k}\n"

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

        # Include RAG context from search agent
        rag_context = ""
        if search_result:
            rag_data = search_result.get("rag_context", {})
            if rag_data.get("vector"):
                rag_context += f"\n[Konteks VectorRAG tentang Jabatan Target]\n{rag_data['vector']}\n"
            if rag_data.get("graph"):
                rag_context += f"\n[Konteks GraphRAG tentang Jabatan Target]\n{rag_data['graph']}\n"

            extractions = search_result.get("extractions", [])
            if extractions:
                rag_context += f"\n[Ekstraksi Search Agent]\n{json.dumps(extractions, ensure_ascii=False, indent=2)}\n"

        prompt = (
            f"Jabatan Target: {target_jabatan}\n"
            f"{context_extra}\n"
            f"{rag_context}\n"
            f"Data Kandidat:\n```json\n{json.dumps(kandidat_json, ensure_ascii=False, indent=2)}\n```\n\n"
            f"Sub-Tugas Evaluasi:\n```json\n{json.dumps(sub_tasks, ensure_ascii=False, indent=2)}\n```\n\n"
            "Tahap 3 — Lakukan Logical Evaluation (L-Eval) dan Counterfactual Evaluation (C-Eval).\n"
            "Gunakan konteks RAG di atas untuk memperkaya evaluasi.\n"
            "Output WAJIB JSON sesuai format yang ditentukan di system prompt Analysis Agent."
        )

        raw = await self._run_and_track(eval_agent, prompt)
        parsed = _extract_json(raw)

        if parsed and isinstance(parsed, dict):
            parsed.setdefault("id_kandidat", kandidat.nip)
            parsed.setdefault("nama", kandidat.nama)
            return parsed

        log.warning(f"⚠️ Fallback evaluasi untuk {kandidat.nip}")
        return {
            "id_kandidat": kandidat.nip,
            "nama": kandidat.nama,
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

        raw = await self._run_and_track(self._agents.synthesis, prompt)
        parsed = _extract_json(raw)

        if parsed and isinstance(parsed, dict) and "peringkat" in parsed:
            return parsed["peringkat"][:top_n]

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
                    "skor_kesesuaian": acc * 50,
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

        raw = await self._run_and_track(self._agents.reviewer, prompt)
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

            confidence = entry.get("confidence_level", "")
            if not confidence:
                acc = int(acceptances)
                confidence = (
                    "Tinggi" if acc == 2 else "Sedang" if acc == 1 else "Rendah"
                )

            kesimpulan = entry.get("kesimpulan", "")
            if not kesimpulan:
                kesimpulan = "Clear" if int(acceptances) == 2 else "Review Needed"

            alasan = entry.get("alasan_penilaian", "")
            if not alasan and detail_models:
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