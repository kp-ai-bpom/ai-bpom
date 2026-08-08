from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.core.llm import LLMAdapter
from app.core.logger import log

from .config import AmbiguityConfig
from .parsers import (
    parse_detection_output,
    parse_interpretation_sample,
    parse_refined_question,
)
from .prompts import (
    build_clarification_from_samples_prompt,
    build_detection_prompt,
    build_interpretation_sampler_prompt,
    build_refinement_prompt,
)
from .types import AmbiguityDetectionResult, AmbiguityResolutionResult
from .uq import compute_uq_signal


_FOLLOWUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(tampilkan|sertakan|tambahkan|tambah|sebutkan|tunjukkan|tunjukan)\b.{0,80}\b(beserta|juga|dengan|dan)\b", re.IGNORECASE),
    re.compile(r"^\s*(ada\s+berapa|berapa\s+(jumlah|orang|total))\b", re.IGNORECASE),
    re.compile(r"^\s*tampilkan\s+juga\b", re.IGNORECASE),
    re.compile(r"^\s*(filter|yang)\s+\w+(\s+\w+){0,3}\s+saja\b", re.IGNORECASE),
    re.compile(r"^\s*(urutkan|urutkanlah|sort)\b", re.IGNORECASE),
    re.compile(r"^\s*termasuk\b", re.IGNORECASE),
)

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
    """Stage 1 (Tahap 3) ambiguity detector dengan UQ pipeline.

    Pipeline produksi (CHATBOT_AMBIGUITY_UQ_ENABLED=true, default):

      1. Short-circuit deterministik untuk inheriting follow-up.
      2. M=5 paralel sampling interpretasi @ T=1.0 via LLM (lihat
         ``build_interpretation_sampler_prompt``).
      3. Embed setiap sample → cluster single-link cosine @ tau=0.80 →
         hitung normalized entropy ``H_norm``.
      4. Bila ``H_norm <= tau_U`` (default 0.30, hasil kalibrasi 60 testcase
         contextual ROC AUC=0.8928, F1=0.873) → non-ambiguous, lanjut.
      5. Bila ambigu → 1 LLM call tambahan @ T=0.1 untuk generate
         clarification + interpretation_options ramah pengguna non-teknis.

    Fallback (CHATBOT_AMBIGUITY_UQ_ENABLED=false): legacy single-call
    detection ``build_detection_prompt`` yang dipertahankan untuk rollback.
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        config: AmbiguityConfig,
    ):
        self._llm_adapter = llm_adapter
        self._config = config
        # Dedicated embedding client (lazy). Sengaja TIDAK pakai
        # llm_adapter.embeddings karena model yang dipakai untuk vector store
        # knowledge entities (AI_EMBEDDINGS_MODEL_NAME) berbeda dengan model
        # yang dipakai saat kalibrasi UQ (text-embedding-3-small). Memisahkan
        # client menjamin parity τ_cluster=0.80 dan τ_U=0.30.
        self._uq_embeddings: Any = None

    def _get_uq_embeddings(self) -> Any:
        if self._uq_embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            from pydantic import SecretStr

            self._uq_embeddings = OpenAIEmbeddings(
                model=self._config.uq_embedding_model,
                api_key=SecretStr(settings.OPENAI_API_KEY),
                base_url=settings.AI_BASE_URL,
            )
            log.info(
                "✅ UQ Embeddings Initialized (model=%s)",
                self._config.uq_embedding_model,
            )
        return self._uq_embeddings

    async def _invoke_llm_with_retry(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        op_label: str,
    ) -> Any | None:
        """Invoke instruct LLM dengan timeout + rate-limit retry/backoff.

        Mirror pola legacy ``_detect_legacy``: bila exception adalah rate-limit
        error → exponential backoff up to ``rate_limit_retries`` attempts.
        Selain itu (timeout/transport/parse) → return None dan caller tentukan
        fallback. ``op_label`` hanya untuk logging.
        """
        try:
            return await asyncio.wait_for(
                self._llm_adapter.instruct.bind(
                    max_tokens=max_tokens,
                    temperature=temperature,
                ).ainvoke(messages),
                timeout=self._config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            log.warning("%s timeout", op_label)
            return None
        except Exception as exc:
            if not self._is_rate_limit_error(exc):
                log.warning("%s failed: %s", op_label, exc)
                return None
            for retry in range(self._config.rate_limit_retries):
                await asyncio.sleep(2**retry)
                try:
                    return await asyncio.wait_for(
                        self._llm_adapter.instruct.bind(
                            max_tokens=max_tokens,
                            temperature=temperature,
                        ).ainvoke(messages),
                        timeout=self._config.timeout_seconds,
                    )
                except Exception as inner:
                    log.warning(
                        "%s rate-limit retry %d failed: %s",
                        op_label, retry + 1, inner,
                    )
            log.warning("%s rate-limit exhausted retries", op_label)
            return None

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

    # ── Public API ──────────────────────────────────────────────────────────

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

        # Short-circuit: inheriting follow-up (deterministic, no LLM call).
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

        if self._config.uq_enabled:
            return await self._detect_with_uq(
                question=question,
                schema_context=schema_context,
                history=history,
            )
        return await self._detect_legacy(
            question=question,
            schema_context=schema_context,
            history=history,
        )

    # ── UQ pipeline (Stage 1 production) ────────────────────────────────────

    async def _sample_interpretation(
        self,
        question: str,
        history: list[dict[str, Any]],
    ) -> str:
        """Satu panggilan sampler @ T=T_SAMPLING.

        Mengembalikan string ``pertanyaan_mandiri`` dari output JSON, atau
        string kosong bila gagal. Caller mengagregasi M panggilan paralel.
        """
        prompt = build_interpretation_sampler_prompt(
            question=question,
            conversation_history=history,
        )
        response = await self._invoke_llm_with_retry(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Anda adalah penginterpretasi pertanyaan. "
                        "Keluarkan JSON valid saja sesuai format."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self._config.uq_sample_max_tokens,
            temperature=self._config.uq_t_sampling,
            op_label="UQ sampler",
        )
        if response is None:
            # Sample gagal → masuk distribusi sebagai ERROR fingerprint
            # (lihat compute_uq_signal). Ini DISENGAJA mirror lampiran_c_uq.py.
            return ""
        content = str(getattr(response, "content", "") or "")
        return parse_interpretation_sample(content)

    async def _embed_samples(self, samples: list[str]) -> list[list[float]]:
        """Embed samples in a worker thread (LangChain embeddings sync only).

        Memakai dedicated UQ embeddings client (model dari
        ``CHATBOT_AMBIGUITY_EMBEDDING_MODEL``, default ``text-embedding-3-small``)
        — JANGAN pakai ``llm_adapter.embeddings`` karena model di sana dikunci
        untuk vector store knowledge entities.
        """
        loop = asyncio.get_running_loop()
        client = self._get_uq_embeddings()

        def _embed_all() -> list[list[float]]:
            return [list(client.embed_query(s)) for s in samples]

        try:
            return await loop.run_in_executor(None, _embed_all)
        except Exception as exc:
            log.warning("UQ embedding failed: %s", exc)
            return []

    async def _detect_with_uq(
        self,
        question: str,
        schema_context: str,
        history: list[dict[str, Any]],
    ) -> AmbiguityDetectionResult:
        m = self._config.uq_m_sampling

        # Step 1: M paralel samples @ T_SAMPLING.
        samples = await asyncio.gather(
            *[self._sample_interpretation(question, history) for _ in range(m)]
        )
        valid_samples = [s for s in samples if s and s.strip()]

        # Bila SEMUA sample gagal, tidak ada apa pun untuk di-cluster.
        # Distribusi murni ERROR menghasilkan H_norm=0 → tetap konsisten
        # dengan compute_uq_signal, tapi short-circuit di sini menghindari
        # call embedding kosong.
        if not valid_samples:
            log.warning(
                "UQ degraded: 0/%d valid samples, fallback non-ambiguous", m
            )
            return AmbiguityDetectionResult(is_ambiguous=False)

        # Step 2: embed sample valid. Sample yang gagal masuk distribusi
        # sebagai fingerprint ``"ERROR"`` (lihat compute_uq_signal) — mirror
        # persis perilaku kalibrasi lampiran_c_uq.py.
        embeddings = await self._embed_samples(valid_samples)
        if len(embeddings) != len(valid_samples) or not embeddings:
            log.warning("UQ degraded: embedding failed, fallback non-ambiguous")
            return AmbiguityDetectionResult(is_ambiguous=False)

        # Step 3: cluster + entropy.
        signal = compute_uq_signal(
            samples=valid_samples,
            embeddings=embeddings,
            m_total=m,
            tau_cluster=self._config.uq_tau_cluster,
        )
        h_norm = float(signal["h_norm"])
        unique_outcomes = int(signal["unique_outcomes"])
        majority_ratio = float(signal["majority_ratio"])

        log.info(
            "UQ detect: M=%d valid=%d H_norm=%.4f tau_U=%.2f unique=%d "
            "maj_ratio=%.3f mic=%.4f → %s",
            m, len(valid_samples), h_norm, self._config.uq_tau_u,
            unique_outcomes, majority_ratio,
            float(signal["mean_intra_cosine"]),
            "AMBIGUOUS" if h_norm > self._config.uq_tau_u else "confident",
        )

        # Bundle UQ telemetry untuk dipropagasi ke pipeline_trace.
        # Dipakai oleh Stage 1 (question_rewriting) metadata.uncertainty
        # serta Stage 3 (ambiguity_detection) output di services.py.
        from dataclasses import replace as _dc_replace
        uq_fields = dict(
            h_norm=h_norm,
            tau_u=float(self._config.uq_tau_u),
            m_samples=m,
            unique_clusters=unique_outcomes,
        )

        # Step 4: gate.
        if h_norm <= self._config.uq_tau_u:
            return AmbiguityDetectionResult(is_ambiguous=False, **uq_fields)

        # Step 5: ambigu → generate clarification dari M sample.
        clarification = await self._generate_clarification_from_samples(
            question=question,
            schema_context=schema_context,
            history=history,
            sample_interpretations=valid_samples,
        )
        # Inject sinyal UQ ke hasil clarification (dataclass frozen → replace).
        return _dc_replace(clarification, **uq_fields)

    async def _generate_clarification_from_samples(
        self,
        question: str,
        schema_context: str,
        history: list[dict[str, Any]],
        sample_interpretations: list[str],
    ) -> AmbiguityDetectionResult:
        prompt = build_clarification_from_samples_prompt(
            question=question,
            schema_context=schema_context,
            conversation_history=history,
            sample_interpretations=sample_interpretations,
        )
        response = await self._invoke_llm_with_retry(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Anda adalah generator klarifikasi. "
                        "Keluarkan JSON valid saja sesuai format."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self._config.detection_max_tokens,
            temperature=self._config.llm_temperature,
            op_label="UQ clarification generator",
        )
        if response is None:
            # UQ sudah memutuskan ambigu — JANGAN diam-diam balik jadi
            # non-ambiguous. Pakai default deterministik dari sample.
            return self._safe_clarification_default(sample_interpretations)

        content = str(getattr(response, "content", "") or "").strip()
        parsed = parse_detection_output(content)
        if parsed is None or not parsed.is_ambiguous:
            # Generator menolak / parse error: tetap ambigu (UQ sudah
            # memutuskan). Pakai default deterministik dari sample.
            return self._safe_clarification_default(sample_interpretations)
        return parsed

    @staticmethod
    def _safe_clarification_default(
        sample_interpretations: list[str],
    ) -> AmbiguityDetectionResult:
        """Fallback deterministik bila clarification generator gagal.

        UQ sudah memutuskan ambigu; kita TIDAK boleh diam-diam mengubahnya
        menjadi non-ambiguous. Tawarkan sample interpretations sebagai opsi
        apa adanya supaya pengguna tetap punya pilihan.
        """
        from .types import InterpretationOption

        seen: set[str] = set()
        options: list[InterpretationOption] = []
        for raw in sample_interpretations:
            label = (raw or "").strip()
            if not label or label in seen:
                continue
            seen.add(label)
            options.append(InterpretationOption(label=label[:120], description=""))
            if len(options) >= 4:
                break
        if not options:
            return AmbiguityDetectionResult(is_ambiguous=False)
        return AmbiguityDetectionResult(
            is_ambiguous=True,
            ambiguity_type="scope",
            clarification_question=(
                "Saya menemukan beberapa kemungkinan interpretasi. "
                "Mana yang Anda maksud?"
            ),
            interpretation_options=options,
        )

    # ── Legacy single-call detection (fallback if UQ disabled) ──────────────

    async def _detect_legacy(
        self,
        question: str,
        schema_context: str,
        history: list[dict[str, Any]],
    ) -> AmbiguityDetectionResult:
        prompt = build_detection_prompt(
            question=question,
            schema_context=schema_context,
            conversation_history=history,
        )
        temperatures = [self._config.llm_temperature, 0.0]

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
                    response = None
                    for retry in range(self._config.rate_limit_retries):
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

    # ── Refinement (unchanged) ──────────────────────────────────────────────

    async def refine(
        self,
        original_question: str,
        ambiguity_type: str,
        clarification_question: str,
        user_response: str,
        conversation_history: list[dict] | None = None,
    ) -> AmbiguityResolutionResult:
        prompt = build_refinement_prompt(
            original_question=original_question,
            ambiguity_type=ambiguity_type,
            clarification_question=clarification_question,
            user_response=user_response,
            conversation_history=conversation_history,
        )
        try:
            response = await asyncio.wait_for(
                self._llm_adapter.instruct.bind(
                    max_tokens=self._config.refine_max_tokens,
                    temperature=self._config.llm_temperature,
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
