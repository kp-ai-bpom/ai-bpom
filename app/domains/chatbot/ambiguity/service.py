from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from app.core.llm import LLMAdapter
from app.core.logger import log

from .config import AmbiguityConfig
from .parsers import parse_detection_output, parse_refined_question
from .prompts import build_detection_prompt, build_refinement_prompt
from .types import AmbiguityDetectionResult, AmbiguityResolutionResult


_FOLLOWUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "tampilkan/sertakan/tambahkan ... beserta/juga/dengan ..."
    re.compile(r"^\s*(tampilkan|sertakan|tambahkan|tambah|sebutkan|tunjukkan|tunjukan)\b.{0,80}\b(beserta|juga|dengan|dan)\b", re.IGNORECASE),
    # "ada berapa", "berapa jumlahnya", "berapa orangnya"
    re.compile(r"^\s*(ada\s+berapa|berapa\s+(jumlah|orang|total))\b", re.IGNORECASE),
    # "tampilkan juga emailnya/jabatannya/...nya"
    re.compile(r"^\s*tampilkan\s+juga\b", re.IGNORECASE),
    # "filter ... saja", "yang aktif saja"
    re.compile(r"^\s*(filter|yang)\s+\w+(\s+\w+){0,3}\s+saja\b", re.IGNORECASE),
    # "urutkan/sort berdasarkan ..."
    re.compile(r"^\s*(urutkan|urutkanlah|sort)\b", re.IGNORECASE),
    # "termasuk juga ..."
    re.compile(r"^\s*termasuk\b", re.IGNORECASE),
)

# Kata-kata yang menandakan pertanyaan SUDAH membawa scope sendiri
# (mis. nama universitas, jurusan, jabatan, modifier eksplisit). Kalau ada,
# pertanyaan dianggap self-contained → tidak boleh di-short-circuit.
_SCOPE_KEYWORDS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(universitas|institut|sekolah\s+tinggi|politeknik)\b", re.IGNORECASE),
    re.compile(r"\b(farmasi|hukum|kedokteran|teknik|biologi|kimia|akuntansi|manajemen|ekonomi|komunikasi|psikologi)\b", re.IGNORECASE),
    re.compile(r"\b(termasuk\s+(non\s*aktif|pensiun|berhenti)|semua\s+status|sejak\s+\d{4}|tahun\s+\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(pranata|ahli|kepala|direktur|deputi)\b", re.IGNORECASE),
    re.compile(r"\b(generasi|gen[\s-]?(x|y|z)|millennial|baby\s*boom)", re.IGNORECASE),
    re.compile(r"\b(unit\s+kerja|balai|deputi|biro|direktorat)\b", re.IGNORECASE),
    re.compile(r"\b(pendidikan|jenjang|s1|s2|s3|d3|d4|sma|sd|smp)\b", re.IGNORECASE),
)


class AmbiguityService:
    def __init__(
        self,
        llm_adapter: LLMAdapter,
        config: AmbiguityConfig,
    ):
        self._llm_adapter = llm_adapter
        self._config = config

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "rate" in text and "limit" in text

    def _build_recent_history(
        self,
        conversation_history: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if not conversation_history:
            return []

        rows: list[dict[str, Any]] = []
        for item in conversation_history[-self._config.max_history_turns :]:
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if not role or not content:
                continue
            rows.append(
                {
                    "role": role,
                    "content": content[:500],
                    "timestamp": str(item.get("timestamp") or self._now_iso()),
                }
            )
        return rows

    @staticmethod
    def _is_inheriting_followup(
        question: str,
        history: list[dict[str, Any]],
    ) -> bool:
        """Deteksi deterministik pertanyaan follow-up yang mewarisi konteks.

        Aturan: pertanyaan dianggap inheriting follow-up bila SEMUA berikut
        terpenuhi:
        1. Cocok dengan salah satu pola follow-up yang umum (lihat
           ``_FOLLOWUP_PATTERNS`` — mis. "tampilkan ... beserta ...",
           "ada berapa", "filter ... saja", "urutkan ...").
        2. Tidak membawa scope/filter sendiri (tidak match ``_SCOPE_KEYWORDS``).
        3. Pendek (≤ 10 kata) — heuristik tambahan untuk membedakan dari
           pertanyaan self-contained yang kebetulan diawali "tampilkan".
        4. Ada minimal satu turn user sebelumnya di ``history`` yang panjang
           (≥ 5 kata) — diasumsikan turn tersebut sudah menetapkan scope.

        Kalau semua syarat terpenuhi, kita yakin pertanyaan ini harus
        diperlakukan sebagai follow-up dan tidak boleh di-flag ambigu.
        """
        text = (question or "").strip()
        if not text:
            return False
        word_count = len(text.split())
        if word_count > 10:
            return False
        if not any(pat.search(text) for pat in _FOLLOWUP_PATTERNS):
            return False
        if any(pat.search(text) for pat in _SCOPE_KEYWORDS):
            return False
        for turn in history:
            if str(turn.get("role") or "").lower() != "user":
                continue
            prior = str(turn.get("content") or "").strip()
            if len(prior.split()) >= 5:
                return True
        return False

    async def detect(
        self,
        question: str,
        schema_context: str,
        conversation_history: list[dict[str, Any]] | None,
        original_question: str | None = None,
    ) -> AmbiguityDetectionResult:
        if not self._config.enabled:
            return AmbiguityDetectionResult(is_ambiguous=False)

        history = self._build_recent_history(conversation_history)

        # Deterministik short-circuit: kalau pertanyaan jelas-jelas adalah
        # follow-up pendek tanpa scope sendiri tetapi punya turn user
        # sebelumnya yang menetapkan scope, langsung lanjutkan tanpa LLM
        # detection. Ini menjamin compliance dengan ATURAN INHERITANCE
        # FOLLOW-UP di prompt walaupun LLM kadang abaikan rule tersebut.
        #
        # Cek pada DUA bentuk pertanyaan:
        #   1. ``original_question`` — pertanyaan apa adanya dari user, sebelum
        #      di-rewrite oleh question_rewriter. Ini yang paling relevan untuk
        #      pola follow-up (mis. "tampilkan nama beserta jabatannya"),
        #      karena rewriter sering memperluas query menjadi >10 kata atau
        #      menambahkan scope keyword sehingga short-circuit pada bentuk
        #      ter-rewrite tidak fire.
        #   2. ``question`` — pertanyaan ter-rewrite (atau sama dengan original
        #      bila tidak ada rewrite). Tetap dicek sebagai fallback.
        candidates: list[str] = []
        if original_question:
            stripped_original = original_question.strip()
            if stripped_original:
                candidates.append(stripped_original)
        if question and question.strip() not in {c for c in candidates}:
            candidates.append(question.strip())

        for candidate in candidates:
            if self._is_inheriting_followup(candidate, history):
                log.info(
                    "Ambiguity detection short-circuit: inheriting follow-up "
                    "pattern matched, skip LLM detect (candidate=%r, "
                    "matched_on=%s)",
                    candidate[:120],
                    "original" if candidate == (original_question or "").strip()
                    else "rewritten",
                )
                return AmbiguityDetectionResult(is_ambiguous=False)

        prompt = build_detection_prompt(
            question=question,
            schema_context=schema_context,
            conversation_history=history,
        )

        # First pass with configured temperature.
        temperatures = [self._config.temperature, 0.0]

        for attempt, temperature in enumerate(temperatures, start=1):
            try:
                response = await asyncio.wait_for(
                    self._llm_adapter.instruct.bind(
                        max_tokens=self._config.detection_max_tokens,
                        temperature=temperature,
                    ).ainvoke(
                        [
                            {
                                "role": "system",
                                "content": "Anda adalah detektor ambiguitas. Keluarkan JSON valid saja.",
                            },
                            {"role": "user", "content": prompt},
                        ]
                    ),
                    timeout=self._config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                log.warning("Ambiguity detection timeout; fallback non-ambiguous")
                return AmbiguityDetectionResult(is_ambiguous=False)
            except Exception as exc:
                if self._is_rate_limit_error(exc):
                    max_retries = self._config.rate_limit_retries
                    for retry in range(max_retries):
                        await asyncio.sleep(2**retry)
                        try:
                            response = await asyncio.wait_for(
                                self._llm_adapter.instruct.bind(
                                    max_tokens=self._config.detection_max_tokens,
                                    temperature=temperature,
                                ).ainvoke(
                                    [
                                        {
                                            "role": "system",
                                            "content": "Anda adalah detektor ambiguitas. Keluarkan JSON valid saja.",
                                        },
                                        {"role": "user", "content": prompt},
                                    ]
                                ),
                                timeout=self._config.timeout_seconds,
                            )
                            break
                        except Exception:
                            response = None
                    if response is None:
                        log.warning("Ambiguity detection rate-limit fallback non-ambiguous")
                        return AmbiguityDetectionResult(is_ambiguous=False)
                else:
                    log.warning("Ambiguity detection failed: %s", exc)
                    return AmbiguityDetectionResult(is_ambiguous=False)

            content = str(getattr(response, "content", "") or "").strip()
            parsed = parse_detection_output(content)
            if parsed is not None:
                return parsed

            if attempt == len(temperatures):
                log.warning("Ambiguity detection invalid JSON after retry; fallback non-ambiguous")
                return AmbiguityDetectionResult(is_ambiguous=False)

        return AmbiguityDetectionResult(is_ambiguous=False)

    async def refine(
        self,
        original_question: str,
        ambiguity_type: str,
        clarification_question: str,
        user_response: str,
    ) -> AmbiguityResolutionResult:
        prompt = build_refinement_prompt(
            original_question=original_question,
            ambiguity_type=ambiguity_type,
            clarification_question=clarification_question,
            user_response=user_response,
        )

        try:
            response = await asyncio.wait_for(
                self._llm_adapter.instruct.bind(
                    max_tokens=self._config.refine_max_tokens,
                    temperature=self._config.temperature,
                ).ainvoke(
                    [
                        {
                            "role": "system",
                            "content": "Anda adalah query refiner. Keluarkan satu kalimat final saja.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                ),
                timeout=self._config.timeout_seconds,
            )
            refined = parse_refined_question(str(getattr(response, "content", "") or ""))
        except Exception as exc:
            log.warning("Ambiguity refinement failed: %s", exc)
            refined = ""

        if not refined:
            refined = f"{original_question} (klarifikasi: {user_response})"

        refined = refined.strip()
        if len(refined) > 500:
            refined = refined[:500].rstrip()

        return AmbiguityResolutionResult(
            unambiguous_question=refined,
            user_response=user_response,
        )
