import asyncio
import json
from typing import Any, Dict, List

from fastapi import HTTPException, status

from app.core.logger import log

from ..core.agent import AgentAdapter, init_agents
from ..dto.request import KandidatSuksesi
from ..dto.response import (
    DetailEvaluasi,
    KandidatCard,
    KandidatListData,
    KandidatListResponse,
    KandidatResult,
    NineBoxData,
    NineBoxItem,
    NineBoxResponse,
    RekamJejakItem,
    SertifikasiItem,
    SimulasiDataResponse,
    SimulasiResponse,
    SkpTahunItem,
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
                nama = c.get("kandidat_suksesi", {}).get("nama", "")
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
        lengkap dengan ringkasan untuk kartu UI.
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
        import app.domains.pemetaan_suksesor.core.config as _c

        if _c._jabatan_rules_cache is None:
            try:
                with open(local_settings.JABATAN_RULES_PATH, encoding="utf-8") as f:
                    raw = json.load(f)
                if "deskripsi_jabatan" in raw:
                    _c._jabatan_rules_cache = [raw]
                elif isinstance(raw, list):
                    _c._jabatan_rules_cache = raw
                else:
                    _c._jabatan_rules_cache = []
                log.info(
                    f"📋 Jabatan rules loaded — {len(_c._jabatan_rules_cache)} posisi"
                )
            except FileNotFoundError:
                log.warning("⚠️ jabatan_rules.json tidak ditemukan")
                _c._jabatan_rules_cache = []
                return None
            except json.JSONDecodeError:
                log.warning("⚠️ jabatan_rules.json format tidak valid")
                _c._jabatan_rules_cache = []
                return None

        assert _c._jabatan_rules_cache is not None

        normalized = target_jabatan.lower().strip()
        for entry in _c._jabatan_rules_cache:
            nama = entry.get("deskripsi_jabatan", {}).get("nama_jabatan", "")
            if nama.lower().strip() == normalized:
                return entry

        return None

    @staticmethod
    def list_available_jabatan() -> List[str]:
        """Return daftar nama jabatan yang tersedia di jabatan_rules.json."""
        import app.domains.pemetaan_suksesor.core.config as _c

        if _c._jabatan_rules_cache is None:
            SimulationService._load_jabatan_rules("")

        assert _c._jabatan_rules_cache is not None
        return [
            entry.get("deskripsi_jabatan", {}).get("nama_jabatan", "")
            for entry in _c._jabatan_rules_cache
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

        prompt = (
            f"Jabatan Target: {target_jabatan}\n"
            f"{context_extra}\n"
            f"Data Kandidat:\n```json\n{json.dumps(kandidat_json, ensure_ascii=False, indent=2)}\n```\n\n"
            f"Sub-Tugas Evaluasi:\n```json\n{json.dumps(sub_tasks, ensure_ascii=False, indent=2)}\n```\n\n"
            "Tahap 2 — Ekstrak informasi esensial kandidat untuk setiap sub-tugas.\n"
            "Tahap 3 — Lakukan Logical Evaluation (L-Eval) dan Counterfactual Evaluation (C-Eval).\n"
            "Output WAJIB JSON sesuai format yang ditentukan di system prompt Analysis Agent."
        )

        raw = await self._run_and_track(eval_agent, prompt)
        parsed = _extract_json(raw)

        if parsed and isinstance(parsed, dict):
            parsed.setdefault("id_kandidat", kandidat.kandidat_suksesi.id)
            parsed.setdefault("nama", kandidat.kandidat_suksesi.nama)
            return parsed

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