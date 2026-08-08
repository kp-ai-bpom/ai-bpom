import csv
import json
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional
from uuid import uuid4

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logger import log
from app.core.llm import LLMAdapter, init_llm
from app.db.database import get_db

from .semantic_disambiguation import (
    AmbiguityDetectionResult,
    AmbiguityService,
    InterpretationOption,
    get_ambiguity_config,
)
from .question_contextual_rewriting import (
    QuestionRewritingService,
    get_question_rewriting_config,
)
from .repositories import ChatbotRepository
from .database_schema_filtering import (
    PreparedSchemaContext,
    RetrievedTable,
    SemanticMemoryPipeline,
    get_semantic_memory_config,
)


class _StageHandle:
    """Mutable handle yang dilewatkan ke caller dalam blok ``recorder.stage(...)``.

    Memberikan API kecil untuk mengisi ``input``, ``output``, ``summary`` dan
    ``metadata`` tanpa perlu mengetahui struktur dict internal recorder.
    """

    def __init__(self, entry: dict[str, Any]) -> None:
        self._entry = entry

    def set_input(self, value: dict[str, Any]) -> None:
        self._entry["input"] = value

    def set_output(self, value: dict[str, Any]) -> None:
        self._entry["output"] = value

    def set_summary(self, value: str) -> None:
        self._entry["summary"] = value

    def set_metadata(self, value: dict[str, Any]) -> None:
        self._entry["metadata"] = value


class _PipelineTraceRecorder:
    """Mengumpulkan jejak eksekusi 5 tahap pipeline ``send_message``.

    Konvensi label mengikuti penomoran tahap pada skripsi:

    - Stage 1 — Question Rewriting
    - Stage 2 — Schema Retrieval (Semantic Memory)
    - Stage 3 — Ambiguity Detection & Clarification
    - Stage 4 — SQL Generation
    - Stage 5 — SQL Validation

    Pemakaian:

    .. code-block:: python

        recorder = _PipelineTraceRecorder()
        with recorder.stage("question_rewriting") as st:
            st.set_input({"current_query": message})
            result = await rewriter.rewrite(...)
            st.set_output({"rewritten_query": result.rewritten_query})
            st.set_summary("Rewrite: ...")

        # untuk skip:
        recorder.skip("question_rewriting", "Working memory kosong")

        # untuk entry post-hoc (durasi tidak perlu diukur):
        recorder.record_post_hoc(
            "sql_validation",
            status="executed",
            summary="PASS (5/5 dimensi)",
            output={"validation_status": "PASS", ...},
        )

        payload = recorder.to_payload()
    """

    STAGE_LABELS: dict[str, str] = {
        "question_rewriting": "Stage 1 — Question Rewriting",
        "schema_retrieval": "Stage 2 — Schema Retrieval (Semantic Memory)",
        "schema_uq": "Stage 2b — Schema Interpretation UQ",
        "ambiguity_detection": "Stage 3 — Ambiguity Detection & Clarification",
        "sql_generation": "Stage 4 — SQL Generation",
        "sql_validation": "Stage 5 — SQL Validation",
    }

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def _make_entry(
        self,
        stage_id: str,
        *,
        label: str | None = None,
        status: str = "executed",
        duration_ms: int = 0,
        summary: str = "",
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "stage": stage_id,
            "label": label or self.STAGE_LABELS.get(stage_id, stage_id),
            "status": status,
            "duration_ms": duration_ms,
            "summary": summary,
            "input": input,
            "output": output,
            "metadata": metadata,
            "error": error,
        }

    @contextmanager
    def stage(self, stage_id: str, label: str | None = None):
        entry = self._make_entry(stage_id, label=label, status="executed")
        start = time.perf_counter()
        handle = _StageHandle(entry)
        try:
            yield handle
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            entry["duration_ms"] = int((time.perf_counter() - start) * 1000)
            self._entries.append(entry)
            raise
        else:
            entry["duration_ms"] = int((time.perf_counter() - start) * 1000)
            self._entries.append(entry)

    def skip(
        self,
        stage_id: str,
        summary: str,
        *,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._entries.append(
            self._make_entry(
                stage_id,
                label=label,
                status="skipped",
                summary=summary,
                metadata=metadata,
            )
        )

    def record_post_hoc(
        self,
        stage_id: str,
        *,
        status: str = "executed",
        summary: str = "",
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        duration_ms: int = 0,
        label: str | None = None,
    ) -> None:
        self._entries.append(
            self._make_entry(
                stage_id,
                label=label,
                status=status,
                duration_ms=duration_ms,
                summary=summary,
                input=input,
                output=output,
                metadata=metadata,
            )
        )

    def to_payload(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._entries]


_PIPELINE_STAGE_ORDER: tuple[str, ...] = (
    "question_rewriting",
    "schema_retrieval",
    "schema_uq",
    "ambiguity_detection",
    "sql_generation",
    "sql_validation",
)


def _merge_pipeline_traces(
    snapshot: list[dict[str, Any]] | None,
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge trace turn ASK (snapshot) dengan trace turn RESUME (current).

    Aturan merge per-stage:
    1. Bila stage *executed* di ``current`` (turn ini), pakai entry tsb.
    2. Else bila stage *executed* di ``snapshot`` (turn klarifikasi sebelumnya),
       pakai entry snapshot + tandai ``metadata.replayed_from_clarification_turn``
       supaya UI bisa render badge "(direplay dari turn klarifikasi)".
    3. Else fall back ke entry ``current`` (biasanya skipped), lalu snapshot,
       lalu ``None`` (stage tidak ada di kedua trace).

    Hasil di-urut menurut ``_PIPELINE_STAGE_ORDER`` supaya UI selalu
    menampilkan Stage 1 → 5 berurutan, tidak peduli urutan eksekusi internal.
    Entry untuk stage lain (di luar 5 stage utama) di-append di akhir
    mengikuti urutan ``current`` lalu ``snapshot`` (defensive).
    """
    by_snapshot: dict[str, dict[str, Any]] = {
        entry["stage"]: entry for entry in (snapshot or []) if entry.get("stage")
    }
    by_current: dict[str, dict[str, Any]] = {
        entry["stage"]: entry for entry in current if entry.get("stage")
    }

    merged: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for stage_id in _PIPELINE_STAGE_ORDER:
        snap = by_snapshot.get(stage_id)
        curr = by_current.get(stage_id)
        chosen: dict[str, Any] | None = None
        is_replay = False
        if curr is not None and curr.get("status") == "executed":
            chosen = dict(curr)
        elif snap is not None and snap.get("status") == "executed":
            chosen = dict(snap)
            is_replay = True
        elif curr is not None:
            chosen = dict(curr)
        elif snap is not None:
            chosen = dict(snap)
            is_replay = True
        if chosen is None:
            continue
        if is_replay:
            md = dict(chosen.get("metadata") or {})
            md["replayed_from_clarification_turn"] = True
            chosen["metadata"] = md
        merged.append(chosen)
        consumed.add(stage_id)

    for entry in current:
        sid = entry.get("stage")
        if sid and sid not in consumed:
            merged.append(dict(entry))
            consumed.add(sid)
    for entry in snapshot or []:
        sid = entry.get("stage")
        if sid and sid not in consumed:
            replay_entry = dict(entry)
            md = dict(replay_entry.get("metadata") or {})
            md["replayed_from_clarification_turn"] = True
            replay_entry["metadata"] = md
            merged.append(replay_entry)
            consumed.add(sid)
    return merged


def _summarize_predicted_tables(
    predicted_tables: dict[str, "RetrievedTable"],
    *,
    top_n: int = 5,
    top_columns: int = 8,
    column_roles: dict[str, dict[str, str]] | None = None,
    tau_column: float = 0.30,
) -> list[dict[str, Any]]:
    """Bentuk ringkasan tabel prediksi untuk payload trace Stage 2.

    Berbeda dari versi lama yang hanya menampilkan nama kolom teratas,
    versi ini mengembalikan **per-kolom**: ``name``, ``score``, ``kept``
    (boolean — apakah lolos τ_column ATAU kolom kunci), dan ``role``
    (``"PK"`` / ``"FK→<target>"`` / ``"display"`` bila tersedia).

    Parameter:
        column_roles: peta peran kolom dari
            :py:meth:`SemanticMemoryPipeline._build_join_graph_and_roles`.
            Dilewatkan dari ``PreparedSchemaContext.column_roles``.
        tau_column: ambang skor kolom (dari konfigurasi pipeline, default
            ``0.30`` pada operating point ter-freeze). Dipakai untuk
            menghitung field ``kept`` per kolom.

    Output urut menurut skor tabel desc dan dipotong ``top_n``. Per tabel,
    kolom diurut menurut skor desc (kolom dengan role tetap muncul, walau
    skor rendah).
    """
    column_roles = column_roles or {}
    items: list[dict[str, Any]] = []
    for value in predicted_tables.values():
        table_key = f"{value.schema}.{value.table}"
        roles_for_table = column_roles.get(table_key, {})
        scores = value.column_scores or {}

        sorted_cols = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        col_entries: list[dict[str, Any]] = []
        for name, score in sorted_cols[:top_columns]:
            score_f = round(float(score or 0.0), 4)
            role = roles_for_table.get(name)
            kept = score_f >= float(tau_column) or role is not None
            entry: dict[str, Any] = {
                "name": name,
                "score": score_f,
                "kept": bool(kept),
            }
            if role:
                entry["role"] = role
            col_entries.append(entry)

        items.append(
            {
                "schema": value.schema,
                "table": value.table,
                "score": round(float(value.score or 0.0), 4),
                "columns": col_entries,
                # Backward-compat: konsumen lama yang membaca ``top_columns``
                # (list nama saja) tetap dilayani.
                "top_columns": [e["name"] for e in col_entries[:5]],
            }
        )
    items.sort(key=lambda item: item["score"], reverse=True)
    return items[:top_n]


def _build_schema_retrieval_output(
    prepared: "PreparedSchemaContext",
    *,
    include_context_excerpt: bool = False,
    include_context_chars: bool = False,
) -> dict[str, Any]:
    """Bentuk payload ``output`` lengkap untuk stage ``schema_retrieval``.

    Mengandung tiga blok yang dipakai UI Stage 2:

    - ``config``: ``{K, tau_table, tau_column}`` — operating point pipeline,
      ditarik dari ``get_semantic_memory_config()`` (frozen K=30,
      τ_table=0.40, τ_column=0.30 per kesepakatan iterasi terbaru).
    - ``predicted_tables``: list tabel terprediksi + per-kolom
      ``{name, score, kept, role}`` (lihat
      :py:func:`_summarize_predicted_tables`).
    - ``join_graph``: edge FK antar tabel prediksi, tiap edge berlabel
      ``source="documented"|"inferred"`` (lihat
      ``database_schema_filtering/fk_graph.py``).

    Field tambahan (``keywords``, dan opsional ``schema_context_chars``
    atau ``schema_context_excerpt``) di-include untuk konsistensi dengan
    payload lama.
    """
    cfg = get_semantic_memory_config()
    column_roles = getattr(prepared, "column_roles", {}) or {}
    payload: dict[str, Any] = {
        "config": {
            "N": int(cfg.top_n_per_keyword),
            "tau_table": round(float(cfg.retrieval_threshold), 4),
            "tau_column": round(float(cfg.column_similarity_threshold), 4),
        },
        "predicted_tables": _summarize_predicted_tables(
            prepared.predicted_tables,
            column_roles=column_roles,
            tau_column=float(cfg.column_similarity_threshold),
        ),
        "join_graph": list(getattr(prepared, "join_graph", []) or []),
        "keywords": prepared.keywords,
    }
    if include_context_chars:
        payload["schema_context_chars"] = len(prepared.context or "")
    if include_context_excerpt:
        payload["schema_context_excerpt"] = (prepared.context or "")[:500]
    return payload


def _build_validation_stage_payload(
    pipeline_result: Any,
) -> tuple[str, str, dict[str, Any]]:
    """Bangun ``(status, summary, output)`` untuk stage validasi SQL post-hoc."""
    status = getattr(pipeline_result, "validation_status", None) or "UNKNOWN"
    iterations = getattr(pipeline_result, "validation_iterations", None) or {}
    rubric = getattr(pipeline_result, "validation_rubric", None) or {}
    revisions = getattr(pipeline_result, "validation_revisions", None) or []

    exec_iter = (
        iterations.get("execution_validation")
        or iterations.get("execution")
        or 0
    )
    sem_iter = (
        iterations.get("semantic_validation")
        or iterations.get("semantic")
        or 0
    )

    pass_count = sum(
        1
        for dim in rubric.values()
        if isinstance(dim, dict) and str(dim.get("label", "")).upper() == "PASS"
    )
    total_dims = len(rubric) if rubric else 0
    rubric_summary = (
        f", {pass_count}/{total_dims} dimensi PASS" if total_dims else ""
    )
    summary = (
        f"{status} ({exec_iter} iterasi eksekusi, {sem_iter} iterasi semantik"
        f"{rubric_summary})"
    )

    output: dict[str, Any] = {
        "validation_status": status,
        "validation_iterations": iterations,
        "validation_rubric": rubric,
        "validation_revisions": revisions,
    }
    return status, summary, output


class ChatbotService:
    """
    Service for handling chatbot LLM interactions.
    Uses LLMAdapter to access instruct, think, and deep_think models.
    """

    _chat_storage_ready = False
    _chat_storage_lock = Lock()

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        repository: ChatbotRepository | None = None,
    ):
        self._llm_adapter = llm_adapter
        self._repository = repository
        self._semantic_pipeline: SemanticMemoryPipeline | None = None
        self._question_rewriting_service: QuestionRewritingService | None = None
        self._ambiguity_config = get_ambiguity_config()
        self._ambiguity_service = AmbiguityService(
            llm_adapter=llm_adapter,
            config=self._ambiguity_config,
        )
        if self._repository is not None:
            self._semantic_pipeline = SemanticMemoryPipeline(
                llm_adapter=self._llm_adapter,
                repository=self._repository,
                config=get_semantic_memory_config(),
            )
            self._question_rewriting_service = QuestionRewritingService(
                llm_adapter=self._llm_adapter,
                repository=self._repository,
                config=get_question_rewriting_config(),
            )

    async def _ensure_chat_storage_ready(self) -> None:
        if self._repository is None:
            raise RuntimeError("Chat repository is not configured")

        if self.__class__._chat_storage_ready:
            return

        await self._repository.ensure_chat_memory_tables()
        await self._repository.ensure_procedural_rules_table()
        with self.__class__._chat_storage_lock:
            self.__class__._chat_storage_ready = True

    @staticmethod
    def _resolve_vector_table_name(table_name: str | None) -> str:
        default_table = get_semantic_memory_config().vector_table
        resolved = (table_name or default_table).strip()

        if resolved.lower() == "string":
            log.warning(
                "Invalid placeholder table_name='string' received; fallback to default table=%s",
                default_table,
            )
            return default_table

        return resolved

    async def _save_ambiguity_metadata(
        self,
        user_id: str,
        session_id: str,
        metadata: dict[str, Any],
        *,
        persist_to_chat_message: bool = True,
    ) -> None:
        """Persist jejak resolusi ambiguitas.

        Args:
            persist_to_chat_message: Bila ``True`` (default), metadata juga
                di-attach ke row ``chat_messages`` paling baru milik session
                ini. Set ke ``False`` pada jalur ``ASK clarification`` (turn
                yang belum punya SQL final dan TIDAK menulis baris baru di
                ``chat_messages``) — tanpa guard ini, update "latest row"
                akan nyangkut ke turn sebelumnya yang tidak terkait,
                menghasilkan badge klarifikasi salah pada history.

                Catatan: pending clarification yang aktif tetap bisa di-render
                UI lewat field ``pending_clarification`` pada response history
                (tabel ``chat_pending_clarifications``), jadi tidak ada
                kehilangan data UX dari skip ini.
        """
        if self._repository is None:
            return

        try:
            await self._repository.update_latest_episode_ambiguity(
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
            )
        except Exception as exc:
            log.warning(
                "Failed to persist ambiguity metadata (episode) user_id=%s session_id=%s error=%s",
                user_id,
                session_id,
                exc,
            )

        if not persist_to_chat_message:
            return

        # Duplicate ke ``chat_messages.ambiguity_metadata`` supaya history
        # endpoint (``GET /api/chatbot/chat``) bisa expose 4 field flat
        # (clarification_asked, clarification_options, interpretation_chosen,
        # ambiguity_type) tanpa join ke ``question_rewriting_episodes``.
        # Mode best-effort: kegagalan tidak mengganggu user flow.
        try:
            await self._repository.update_latest_chat_message_ambiguity(
                session_id=session_id,
                metadata=metadata,
            )
        except Exception as exc:
            log.warning(
                "Failed to persist ambiguity metadata (chat_message) user_id=%s session_id=%s error=%s",
                user_id,
                session_id,
                exc,
            )

    @staticmethod
    def _normalize_resolution_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _match_resolution_option(
        self,
        options: list[str],
        candidates: list[str],
    ) -> str | None:
        for candidate in candidates:
            normalized_candidate = self._normalize_resolution_text(candidate)
            if not normalized_candidate:
                continue
            for option in options:
                normalized_option = self._normalize_resolution_text(option)
                if not normalized_option:
                    continue
                if (
                    normalized_candidate == normalized_option
                    or normalized_candidate in normalized_option
                    or normalized_option in normalized_candidate
                ):
                    return option
        return None

    # ------------------------------------------------------------------
    # Procedural memory helpers (per user_id)
    # ------------------------------------------------------------------

    def _is_reset_command(self, message: str) -> bool:
        """Return True if message looks like a reset-preference command."""
        keywords = self._ambiguity_config.procedural_reset_keywords
        if not keywords:
            return False
        normalized = message.strip().lower()
        return any(kw in normalized for kw in keywords)

    def _strip_reset_keywords(self, message: str) -> str:
        """Remove the reset keyword prefix to recover the underlying pattern,
        e.g. 'reset preferensi tampilkan data pegawai' -> 'tampilkan data pegawai'.
        """
        normalized = message.strip()
        lowered = normalized.lower()
        for kw in self._ambiguity_config.procedural_reset_keywords:
            if kw and lowered.startswith(kw):
                return normalized[len(kw):].strip(" :,-")
        # If keyword appears mid-string, just remove first occurrence
        for kw in self._ambiguity_config.procedural_reset_keywords:
            if kw and kw in lowered:
                idx = lowered.index(kw)
                return (normalized[:idx] + normalized[idx + len(kw):]).strip(" :,-")
        return normalized

    def _embed_text(self, text_value: str) -> list[float] | None:
        try:
            vector = self._llm_adapter.embeddings.embed_query(text_value)
        except Exception as exc:
            log.warning("Failed to embed text for procedural memory: %s", exc)
            return None
        if not vector:
            return None
        return [float(v) for v in vector]

    async def _try_procedural_match(
        self,
        user_id: str,
        message: str,
    ) -> dict[str, Any] | None:
        """Embed message and look up an active procedural rule for this user
        within the configured similarity threshold and TTL window. Returns the
        rule row (with similarity score) on hit, else None.
        """
        if not self._ambiguity_config.procedural_enabled:
            return None
        if self._repository is None:
            return None
        embedding = self._embed_text(message)
        if embedding is None:
            return None
        rule = await self._repository.find_matching_procedural_rule(
            user_id=user_id,
            embedding=embedding,
            similarity_threshold=self._ambiguity_config.procedural_similarity_threshold,
            ttl_days=self._ambiguity_config.procedural_ttl_days,
        )
        return rule

    async def _learn_procedural_rule(
        self,
        user_id: str,
        original_question: str,
        canonical_resolution: str,
        ambiguity_type: str | None,
        clarification_question: str | None,
        options: list[str] | None,
    ) -> str | None:
        """Persist a procedural rule extracted from a successful clarification
        cycle. If a similar rule already exists for this user, supersede it
        with a new versioned entry.
        """
        if not self._ambiguity_config.procedural_enabled:
            return None
        if self._repository is None:
            return None
        if not original_question.strip() or not canonical_resolution.strip():
            return None

        embedding = self._embed_text(original_question)
        if embedding is None:
            return None

        existing = await self._repository.find_existing_rule_for_pattern(
            user_id=user_id,
            embedding=embedding,
            similarity_threshold=self._ambiguity_config.procedural_similarity_threshold,
        )
        next_version = 1
        if existing:
            try:
                next_version = int(existing.get("version") or 1) + 1
            except (TypeError, ValueError):
                next_version = 2

        new_rule_id = await self._repository.insert_procedural_rule(
            user_id=user_id,
            question_pattern=original_question,
            embedding=embedding,
            canonical_resolution=canonical_resolution,
            ambiguity_type=ambiguity_type,
            source_clarification_question=clarification_question,
            source_options=options,
            version=next_version,
        )
        if new_rule_id and existing:
            await self._repository.supersede_procedural_rule(
                old_rule_id=str(existing.get("rule_id") or ""),
                new_rule_id=new_rule_id,
            )
        return new_rule_id

    async def _try_auto_resolve(
        self,
        user_id: str,
        detection_ambiguity_type: str | None,
        options: list[InterpretationOption],
        conversation_history: list[dict[str, Any]],
    ) -> str | None:
        if not options:
            return None

        option_labels = [opt.label for opt in options if opt.label]
        if not option_labels:
            return None

        candidates: list[str] = []
        for turn in reversed(conversation_history):
            role = str(turn.get("role") or "").strip().lower()
            if role != "user":
                continue
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            candidates.append(content)
            if len(candidates) >= self._ambiguity_config.auto_resolve_window:
                break

        if self._repository is not None and detection_ambiguity_type:
            episodic_candidates = await self._repository.get_recent_ambiguity_resolutions(
                user_id=user_id,
                ambiguity_type=detection_ambiguity_type,
                limit=self._ambiguity_config.auto_resolve_window,
            )
            candidates.extend(episodic_candidates)

        return self._match_resolution_option(options=option_labels, candidates=candidates)

    @staticmethod
    def _parse_json_payload(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _validation_payload(pipeline_result: Any) -> dict[str, Any]:
        """Bangun fragmen payload metadata validasi SQL untuk respons API.

        Mengambil field opsional ``validation_*`` dari ``PipelineResult`` dan
        memetakannya ke struktur yang siap dimasukkan ke ``SendMessageDataResponse``.
        Mengembalikan dict kosong jika pipeline tidak menyertakan metadata
        (mis. validasi dilewati pada procedural-hit path).
        """
        if pipeline_result is None:
            return {}

        status = getattr(pipeline_result, "validation_status", None)
        iterations = getattr(pipeline_result, "validation_iterations", None)
        rubric = getattr(pipeline_result, "validation_rubric", None)
        revisions = getattr(pipeline_result, "validation_revisions", None)

        payload: dict[str, Any] = {}
        if status is not None:
            payload["validation_status"] = status
        if iterations is not None:
            payload["validation_iterations"] = iterations
        if rubric is not None:
            payload["validation_rubric"] = rubric
        if revisions is not None:
            payload["validation_revisions"] = revisions
        return payload

    def instruct(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Invoke the instruct model for general purpose conversation"""
        response = self._llm_adapter.instruct.bind(max_tokens=max_tokens).invoke(
            messages
        )
        return str(response.content)

    def think(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Invoke the think model for reasoning-focused responses"""
        response = self._llm_adapter.think.bind(max_tokens=max_tokens).invoke(messages)
        return str(response.content)

    def deep_think(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Invoke the deep_think model for in-depth analysis"""
        response = self._llm_adapter.deep_think.bind(max_tokens=max_tokens).invoke(
            messages
        )
        return str(response.content)

    async def a_instruct(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Async invoke the instruct model"""
        response = await self._llm_adapter.instruct.bind(max_tokens=max_tokens).ainvoke(
            messages
        )
        return str(response.content)

    async def a_think(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Async invoke the think model"""
        response = await self._llm_adapter.think.bind(max_tokens=max_tokens).ainvoke(
            messages
        )
        return str(response.content)

    async def a_deep_think(
        self, messages: List, max_tokens: Optional[int] = None
    ) -> str:
        """Async invoke the deep_think model"""
        response = await self._llm_adapter.deep_think.bind(
            max_tokens=max_tokens
        ).ainvoke(messages)
        return str(response.content)

    async def send_message(
        self,
        user_id: str,
        message: str,
        session_id: Optional[str] = None,
        ablate_stages: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        """Run semantic pipeline and persist chat session/message in database.

        Auto-detects: jika session punya pending clarification,
        treat message sebagai respon klarifikasi; jika tidak, treat sebagai pertanyaan baru.

        ``ablate_stages`` (eval-only): himpunan nama tahap yang ingin
        dimatikan untuk benchmark ablation. Nilai yang dikenali:
            - ``"rewriting"``     → matikan Stage 1 (question rewriting)
            - ``"ambiguity"``     → matikan Stage 3 (ambiguity detection)
            - ``"active_filter"`` → matikan Stage 5b (deterministic active filter)
        Nilai lain diabaikan. Default: kosong (semua tahap aktif).
        """
        ablate_rewriting = "rewriting" in ablate_stages
        ablate_ambiguity = "ambiguity" in ablate_stages
        ablate_active_filter = "active_filter" in ablate_stages

        normalized_user_id = str(user_id).strip()
        normalized_message = str(message).strip()

        if not normalized_user_id:
            raise ValueError("user_id cannot be empty")
        if not normalized_message:
            raise ValueError("message cannot be empty")

        normalized_session_id = session_id.strip() if session_id is not None else None
        if session_id is not None and not normalized_session_id:
            raise ValueError("session_id cannot be empty")
        resolved_session_id = normalized_session_id or str(uuid4())

        if self._semantic_pipeline is None:
            raise RuntimeError("Semantic pipeline is not configured")
        if self._repository is None:
            raise RuntimeError("Chat repository is not configured")

        await self._ensure_chat_storage_ready()

        # Pipeline trace recorder. Setiap return path (reset, resume,
        # procedural, auto-resolve, clarification, normal answer) wajib
        # menyertakan ``pipeline_trace`` agar UI skripsi dapat menampilkan
        # dropdown per-tahap dan history percakapan dapat di-replay dengan
        # jejak utuh dari kolom ``chat_messages.pipeline_trace``.
        recorder = _PipelineTraceRecorder()

        # Reset-preference command: user explicitly asks to forget a learned
        # procedural rule (e.g. "reset preferensi tampilkan data pegawai").
        # Takes precedence over pending clarification — user signals they want
        # to start over.
        if self._is_reset_command(normalized_message):
            pattern_text = self._strip_reset_keywords(normalized_message)
            archived_count = 0
            if pattern_text and self._ambiguity_config.procedural_enabled:
                pattern_embedding = self._embed_text(pattern_text)
                if pattern_embedding is not None:
                    archived_count = await self._repository.archive_procedural_rules_by_pattern(
                        user_id=normalized_user_id,
                        embedding=pattern_embedding,
                        similarity_threshold=self._ambiguity_config.procedural_similarity_threshold,
                    )
            # Also clear any pending clarification for this session so the next
            # turn starts cleanly.
            existing_pending = await self._repository.load_latest_pending_clarification(
                user_id=normalized_user_id,
                session_id=resolved_session_id,
            )
            if existing_pending:
                await self._repository.delete_pending_clarification(
                    existing_pending.get("pending_id") or ""
                )
            reset_skip_reason = (
                "Pipeline utama dilewati: pesan adalah perintah reset preferensi"
            )
            for _stage_id in (
                "question_rewriting",
                "schema_retrieval",
                "ambiguity_detection",
                "sql_generation",
                "sql_validation",
            ):
                recorder.skip(_stage_id, reset_skip_reason)
            # Catatan desain: turn ``reset preferensi`` adalah perintah kontrol,
            # bukan pertanyaan natural. Karena tidak menghasilkan SQL/jawaban,
            # turn ini tidak dipersist ke ``chat_messages`` (konsisten dengan
            # perilaku sebelum iter 14). Trace tetap dikembalikan ke client
            # supaya UI dapat menampilkan bahwa 5 stage memang dilewati.
            return {
                "type": "answer",
                "user_id": normalized_user_id,
                "session_id": resolved_session_id,
                "question": normalized_message,
                "query": "",
                "explanation": (
                    f"Preferensi untuk pertanyaan serupa telah direset "
                    f"({archived_count} aturan diarsipkan). Silakan kirim "
                    f"pertanyaan ulang untuk memilih interpretasi baru."
                ),
                "options": [],
                "pipeline_trace": recorder.to_payload(),
            }

        conversation_history: list[dict[str, Any]] = []
        if self._question_rewriting_service is not None:
            conversation_history = self._question_rewriting_service.get_session_memory_snapshot(
                user_id=normalized_user_id,
                session_id=resolved_session_id,
                max_turns=self._ambiguity_config.max_history_turns,
            )

        # Auto-detect: apakah ada pending clarification di session ini?
        pending = await self._repository.load_latest_pending_clarification(
            user_id=normalized_user_id,
            session_id=resolved_session_id,
        )
        resume_mode = pending is not None

        if resume_mode:
            # Resume mode: treat message as clarification response
            user_response = normalized_message

            options_payload = self._parse_json_payload(pending.get("options"))
            # Stored shape (preferred): list of dict {label, description}.
            # Backward-compat: list of plain strings.
            #
            # ``options`` (list[str]) dipakai untuk indexing/resolve user
            # response. ``options_dicts`` (list[dict{label, description}])
            # dipakai untuk persist ke ``ambiguity_metadata.options_offered``
            # supaya UI history bisa render label + description (bukan label
            # saja). Tanpa varian dict, description hilang setelah refresh.
            options: list[str] = []
            options_dicts: list[dict[str, str]] = []
            for item in options_payload or []:
                if isinstance(item, dict):
                    label_text = str(item.get("label") or "").strip()
                    description_text = str(item.get("description") or "").strip()
                else:
                    label_text = str(item or "").strip()
                    description_text = ""
                if label_text:
                    options.append(label_text)
                    options_dicts.append(
                        {"label": label_text, "description": description_text}
                    )
            defaulted = False

            # Map option-id (e.g. "opt_1") back to the underlying interpretation label.
            import re as _re
            opt_match = _re.match(r"^opt_(\d+)$", normalized_message.strip().lower())
            if opt_match and options:
                idx = int(opt_match.group(1)) - 1
                if 0 <= idx < len(options):
                    user_response = options[idx]

            # Check if pending has expired
            expires_at = pending.get("expires_at")
            if isinstance(expires_at, datetime) and expires_at <= datetime.now(timezone.utc):
                if options:
                    user_response = options[0]
                    defaulted = True

            original_question = str(pending.get("standalone_question") or "").strip()
            if not original_question:
                original_question = normalized_message or ""

            refine_result = await self._ambiguity_service.refine(
                original_question=original_question,
                ambiguity_type=str(pending.get("ambiguity_type") or "scope"),
                clarification_question=str(pending.get("clarification_question") or ""),
                user_response=user_response,
                conversation_history=conversation_history,
            )

            relevant_schema_payload = self._parse_json_payload(
                pending.get("relevant_schema")
            ) or {}
            predicted_tables_payload = relevant_schema_payload.get("predicted_tables") or {}
            predicted_tables: dict[str, RetrievedTable] = {}
            if isinstance(predicted_tables_payload, dict):
                for table_key, raw_table in predicted_tables_payload.items():
                    if not isinstance(raw_table, dict):
                        continue
                    predicted_tables[str(table_key)] = RetrievedTable(
                        schema=str(raw_table.get("schema") or ""),
                        table=str(raw_table.get("table") or ""),
                        score=float(raw_table.get("score") or 0.0),
                        column_scores={
                            str(col): float(score)
                            for col, score in (raw_table.get("column_scores") or {}).items()
                        },
                    )

            schema_tables_payload = relevant_schema_payload.get("schema_tables")
            schema_tables = (
                schema_tables_payload if isinstance(schema_tables_payload, list) else []
            )
            prepared_keywords = [
                str(item).strip()
                for item in (relevant_schema_payload.get("keywords") or [])
                if str(item).strip()
            ]
            prepared_context = str(pending.get("schema_context") or "")
            prepared = PreparedSchemaContext(
                keywords=prepared_keywords,
                predicted_tables=predicted_tables,
                context=prepared_context,
                schema_tables=schema_tables,
            )
            # Deteksi *dangling case*: pending dibuat oleh short-circuit Stage 1
            # (lihat blok ``stage1_uq_signal.verdict == "ambiguous"`` jauh di
            # bawah). Pada kasus ini Stage 2 (schema retrieval) belum pernah
            # jalan, sehingga ``relevant_schema`` di pending kosong. Tanpa
            # deteksi ini, Stage 4 (SQL generation) dipanggil dengan schema
            # context kosong → SQL hallucinate / gagal validasi.
            needs_fresh_schema_retrieval = (
                not prepared_context
                and not prepared.predicted_tables
                and not prepared.schema_tables
            )

            unambiguous_question = refine_result.unambiguous_question
            # Pastikan keyword penting dari clarification user (mis. "ahli
            # komputer") ikut terbawa ke generator agar role-intent validator
            # dan instruksi prompt menangkapnya. Refiner kadang menormalisasi
            # ulang pertanyaan dan menanggalkan istilah jabatan yang spesifik.
            generator_query = unambiguous_question
            if user_response and user_response.strip():
                user_response_norm = user_response.strip()
                if user_response_norm.lower() not in unambiguous_question.lower():
                    generator_query = (
                        f"{unambiguous_question} (klarifikasi user: {user_response_norm})"
                    )

            # Stage 1 selalu skip di resume: rewriting dilakukan pada turn ASK.
            # Snapshot turn ASK akan menyediakan IPO Stage 1 saat di-merge.
            recorder.skip(
                "question_rewriting",
                "Dilewati: turn ini adalah lanjutan klarifikasi (stage 1 telah "
                "dijalankan pada turn sebelumnya)",
            )
            if needs_fresh_schema_retrieval:
                # Dangling case: Stage 2 BELUM pernah jalan. Jalankan sekarang
                # dengan ``unambiguous_question`` (hasil refine) sebagai query
                # retrieval supaya Stage 4 punya schema context yang benar.
                with recorder.stage("schema_retrieval") as _stage:
                    _stage.set_input(
                        {
                            "query_for_retrieval": unambiguous_question,
                            "source": (
                                "fresh retrieval pada turn resume — dangling "
                                "reference dari Stage 1 turn sebelumnya"
                            ),
                        }
                    )
                    prepared = await self._semantic_pipeline.prepare_context(
                        unambiguous_question
                    )
                    _stage.set_output(
                        _build_schema_retrieval_output(
                            prepared, include_context_chars=True
                        )
                    )
                    _stage.set_summary(
                        f"{len(prepared.predicted_tables)} tabel di-retrieve "
                        f"({len(prepared.context or '')} karakter konteks)"
                    )
            else:
                # Scope/role/temporal case: Stage 2 sudah dijalankan turn ASK,
                # hasilnya tersimpan di pending. Skip di sini; snapshot turn
                # ASK akan menyediakan IPO Stage 2 saat di-merge.
                recorder.skip(
                    "schema_retrieval",
                    "Dilewati: skema relevan dimuat ulang dari pending "
                    "clarification (stage 2 telah dijalankan pada turn "
                    "sebelumnya)",
                    metadata={
                        "schema_context_chars": len(prepared.context or ""),
                        "predicted_tables_count": len(prepared.predicted_tables),
                        "keywords": prepared.keywords,
                    },
                )
            recorder.record_post_hoc(
                "ambiguity_detection",
                status="executed",
                summary=(
                    "Klarifikasi diselesaikan oleh user"
                    + (" (default karena pending kedaluwarsa)" if defaulted else "")
                ),
                input={
                    "original_question": original_question,
                    "ambiguity_type": str(pending.get("ambiguity_type") or "scope"),
                    "clarification_question": str(
                        pending.get("clarification_question") or ""
                    ),
                    "options_offered": options,
                    "user_response": user_response,
                },
                output={
                    "is_ambiguous": True,
                    "ambiguity_type": str(pending.get("ambiguity_type") or "scope"),
                    "interpretation_chosen": user_response,
                    "unambiguous_question": unambiguous_question,
                    "auto_resolved": False,
                    "defaulted": defaulted,
                },
            )

            with recorder.stage("sql_generation") as _stage:
                _stage.set_input(
                    {
                        "query": generator_query,
                        "schema_context_chars": len(prepared.context or ""),
                        "predicted_tables_count": len(prepared.predicted_tables),
                    }
                )
                pipeline_result = await self._semantic_pipeline.run_from_prepared(
                    query=generator_query,
                    prepared=prepared,
                    ablate_active_filter=ablate_active_filter,
                )
                _stage.set_output(
                    {
                        "sql": pipeline_result.sql,
                        "explanation": pipeline_result.explanation,
                    }
                )
                _stage.set_summary(
                    f"SQL dihasilkan ({len(pipeline_result.sql or '')} karakter)"
                )
                _stage.set_metadata(
                    {
                        "note": (
                            "Durasi mencakup eksekusi SQL dan validasi semantik "
                            "(lihat Stage 5 untuk detail validasi)."
                        )
                    }
                )

            _v_status, _v_summary, _v_output = _build_validation_stage_payload(
                pipeline_result
            )
            recorder.record_post_hoc(
                "sql_validation",
                status="executed",
                summary=_v_summary,
                input={"sql": pipeline_result.sql},
                output=_v_output,
                metadata={
                    "note": (
                        "Durasi validasi tergabung dalam Stage 4 (sql_generation) "
                        "karena dijalankan oleh ``run_from_prepared``."
                    )
                },
            )
            # Merge trace turn ASK (Stage 1 — atau Stage 1+2+3 untuk scope
            # case) ke trace turn resume supaya UI menampilkan IPO utuh 5
            # stage. Tanpa merge ini, Stage 1 (dan Stage 2 untuk non-dangling
            # case) hanya muncul sebagai "skipped" karena recorder turn ini
            # tidak mengeksekusinya ulang.
            _trace_snapshot_raw = pending.get("pipeline_trace_snapshot")
            _trace_snapshot = self._parse_json_payload(_trace_snapshot_raw)
            if not isinstance(_trace_snapshot, list):
                _trace_snapshot = None
            pipeline_trace_payload = _merge_pipeline_traces(
                _trace_snapshot, recorder.to_payload()
            )

            await self._repository.delete_pending_clarification(
                pending.get("pending_id") or ""
            )

            title_seed = original_question or normalized_message or unambiguous_question
            await self._repository.persist_chat_message(
                user_id=normalized_user_id,
                session_id=resolved_session_id,
                title=title_seed[:255],
                question=original_question or unambiguous_question,
                standalone_question=unambiguous_question,
                query=pipeline_result.sql,
                explanation=pipeline_result.explanation,
                pipeline_trace=pipeline_trace_payload,
            )

            ambiguity_metadata = {
                "was_ambiguous": True,
                "ambiguity_type": str(pending.get("ambiguity_type") or "scope"),
                "clarification_asked": str(pending.get("clarification_question") or ""),
                # ``options_offered`` di-persist supaya UI bisa "menggambar
                # ulang" tampilan history "user pernah memilih X dari [A,B,C]"
                # setelah refresh halaman. Disimpan sebagai list[dict] dengan
                # ``label`` + ``description`` agar UI history menampilkan
                # konteks opsi yang sama persis dengan saat clarification
                # ditampilkan, bukan hanya label.
                "options_offered": list(options_dicts),
                "user_response": user_response,
                "interpretation_chosen": user_response,
                "auto_resolved": False,
                "defaulted": defaulted,
            }
            await self._save_ambiguity_metadata(
                user_id=normalized_user_id,
                session_id=resolved_session_id,
                metadata=ambiguity_metadata,
            )

            # Procedural memory: learn rule from this resolution. Pattern is
            # the user's original vague question; canonical resolution is the
            # refined unambiguous question. Skip if we defaulted (no real
            # user input).
            if not defaulted:
                try:
                    await self._learn_procedural_rule(
                        user_id=normalized_user_id,
                        original_question=original_question,
                        canonical_resolution=unambiguous_question,
                        ambiguity_type=str(pending.get("ambiguity_type") or "scope"),
                        clarification_question=str(
                            pending.get("clarification_question") or ""
                        ),
                        options=options,
                    )
                except Exception as exc:
                    log.warning(
                        "Failed to learn procedural rule for user_id=%s: %s",
                        normalized_user_id,
                        exc,
                    )

            return {
                "type": "answer",
                "user_id": normalized_user_id,
                "session_id": resolved_session_id,
                "question": unambiguous_question,
                "query": pipeline_result.sql,
                "explanation": pipeline_result.explanation,
                "options": [],
                **self._validation_payload(pipeline_result),
                "pipeline_trace": pipeline_trace_payload,
            }

        # Procedural memory check: before invoking rewriter + ambiguity
        # detection, see if this user has a learned rule for a semantically
        # equivalent past question. If hit, bypass clarification entirely.
        procedural_rule = await self._try_procedural_match(
            user_id=normalized_user_id,
            message=normalized_message,
        )
        if procedural_rule is not None:
            canonical = str(procedural_rule.get("canonical_resolution") or "").strip()
            rule_id = str(procedural_rule.get("rule_id") or "")
            if canonical:
                proc_similarity = float(procedural_rule.get("similarity") or 0.0)
                proc_skip_meta = {
                    "procedural_rule_id": rule_id,
                    "procedural_similarity": proc_similarity,
                    "canonical_resolution": canonical,
                }
                recorder.skip(
                    "question_rewriting",
                    "Dilewati: procedural memory hit (interpretasi kanonik diambil "
                    "langsung dari aturan tersimpan)",
                    metadata=proc_skip_meta,
                )

                with recorder.stage("schema_retrieval") as _stage:
                    _stage.set_input({"query_for_retrieval": canonical})
                    prepared = await self._semantic_pipeline.prepare_context(canonical)
                    _stage.set_output(
                        _build_schema_retrieval_output(
                            prepared, include_context_excerpt=True
                        )
                    )
                    _stage.set_summary(
                        f"{len(prepared.predicted_tables)} tabel relevan ditemukan"
                    )

                recorder.skip(
                    "ambiguity_detection",
                    "Dilewati: procedural memory hit (interpretasi sebelumnya sudah "
                    "ditetapkan oleh user)",
                    metadata={
                        "ambiguity_type": procedural_rule.get("ambiguity_type"),
                        **proc_skip_meta,
                    },
                )

                # Procedural memory hit → SQL telah terbukti benar pada
                # interaksi sebelumnya, sehingga lewati pipeline validasi
                # (Tahap 5) untuk hemat latensi dan biaya LLM.
                with recorder.stage("sql_generation") as _stage:
                    _stage.set_input(
                        {
                            "query": canonical,
                            "schema_context_chars": len(prepared.context or ""),
                            "predicted_tables_count": len(prepared.predicted_tables),
                        }
                    )
                    pipeline_result = await self._semantic_pipeline.run_from_prepared(
                        query=canonical,
                        prepared=prepared,
                        skip_validation=True,
                        ablate_active_filter=ablate_active_filter,
                    )
                    _stage.set_output(
                        {
                            "sql": pipeline_result.sql,
                            "explanation": pipeline_result.explanation,
                        }
                    )
                    _stage.set_summary(
                        f"SQL dihasilkan ({len(pipeline_result.sql or '')} karakter)"
                    )
                    _stage.set_metadata(
                        {"validation_skipped_reason": "procedural memory hit"}
                    )

                recorder.skip(
                    "sql_validation",
                    "Dilewati: validasi SQL di-bypass karena procedural memory hit "
                    "(SQL telah terbukti benar pada interaksi sebelumnya)",
                    metadata=proc_skip_meta,
                )
                pipeline_trace_payload = recorder.to_payload()

                await self._repository.persist_chat_message(
                    user_id=normalized_user_id,
                    session_id=resolved_session_id,
                    title=normalized_message[:255],
                    question=normalized_message,
                    standalone_question=canonical,
                    query=pipeline_result.sql,
                    explanation=pipeline_result.explanation,
                    pipeline_trace=pipeline_trace_payload,
                )

                ambiguity_metadata = {
                    "was_ambiguous": True,
                    "ambiguity_type": procedural_rule.get("ambiguity_type"),
                    "clarification_asked": None,
                    # Auto-resolved via procedural memory: tidak ada opsi yang
                    # ditawarkan (skip clarification), tetap kirim list kosong
                    # demi konsistensi schema.
                    "options_offered": [],
                    "user_response": None,
                    "interpretation_chosen": canonical,
                    "auto_resolved": True,
                    "defaulted": False,
                    "procedural_hit": True,
                    "procedural_rule_id": rule_id,
                    "procedural_similarity": float(
                        procedural_rule.get("similarity") or 0.0
                    ),
                }
                await self._save_ambiguity_metadata(
                    user_id=normalized_user_id,
                    session_id=resolved_session_id,
                    metadata=ambiguity_metadata,
                )

                if rule_id:
                    await self._repository.record_procedural_rule_hit(rule_id)

                return {
                    "type": "answer",
                    "user_id": normalized_user_id,
                    "session_id": resolved_session_id,
                    "question": canonical,
                    "query": pipeline_result.sql,
                    "explanation": pipeline_result.explanation,
                    "options": [],
                    **self._validation_payload(pipeline_result),
                    "pipeline_trace": pipeline_trace_payload,
                }

        standalone_query = normalized_message
        # Hanya rewrite jika ada working memory di sesi ini (true follow-up turn).
        # Tanpa guard ini, episodic memory lintas-sesi bisa "memperkaya" pertanyaan
        # vague menjadi specific sehingga lolos dari detektor ambiguitas.
        rewrite_invoked = False
        stage1_uq_signal: dict[str, Any] | None = None
        if ablate_rewriting:
            recorder.skip(
                "question_rewriting",
                "Dilewati: Stage 1 dimatikan via ablate_stages (eval).",
            )
        if (
            not ablate_rewriting
            and self._question_rewriting_service is not None
            and conversation_history
        ):
            with recorder.stage("question_rewriting") as _stage:
                _stage.set_input(
                    {
                        "current_query": normalized_message,
                        "working_memory_turns": len(conversation_history),
                        "working_memory_preview": conversation_history[-6:],
                    }
                )
                rewrite_result = await self._question_rewriting_service.rewrite(
                    user_id=normalized_user_id,
                    session_id=resolved_session_id,
                    current_query=normalized_message,
                )
                rewritten_query = rewrite_result.rewritten_query.strip()
                if rewritten_query:
                    standalone_query = rewritten_query
                rewrite_invoked = True
                _stage_output: dict[str, Any] = {
                    "original_query": rewrite_result.original_query,
                    "rewritten_query": rewrite_result.rewritten_query,
                    "episodic_matches_count": rewrite_result.episodic_matches_count,
                    "top_similarity": round(
                        float(rewrite_result.top_similarity or 0.0), 4
                    ),
                }
                # Surface sinyal UQ Stage 1 (M-sampling rewriter NL→NL)
                # langsung dari rewrite_result.uncertainty. Mirror format
                # yang sebelumnya di-copy post-hoc dari Stage 3, namun
                # sekarang benar-benar dihitung di Stage 1. Threshold
                # τ_U=0.40 berasal dari kalibrasi rewriter (frozen).
                if rewrite_result.uncertainty is not None:
                    _uq = rewrite_result.uncertainty
                    _uq_payload = {
                        "h_norm": round(float(_uq.get("h_norm", 0.0)), 4),
                        "tau_u": float(_uq.get("tau_u", 0.40)),
                        "verdict": _uq.get("verdict", "confident"),
                        "m_samples": int(_uq.get("m_total", 0)),
                        "m_valid": int(_uq.get("m_valid", 0)),
                        "unique_clusters": int(_uq.get("unique_outcomes", 0)),
                        "majority_ratio": round(
                            float(_uq.get("majority_ratio", 0.0)), 4
                        ),
                        "mean_intra_cosine": round(
                            float(_uq.get("mean_intra_cosine", 0.0)), 4
                        ),
                        "n_error": int(_uq.get("n_error", 0)),
                        "source_stage": "question_rewriting",
                    }
                    _stage_output["uncertainty"] = _uq_payload
                    _stage.set_metadata({"uncertainty": _uq_payload})
                    # Hoist sinyal UQ Stage 1 ke outer scope sehingga Stage 3
                    # (ambiguity_detection) dapat menghormati verdict ini untuk
                    # men-trigger clarification — menjaga konsistensi Stage 1↔3
                    # ↔4↔5. Tanpa hoist, downstream akan men-treat
                    # ``rewritten_query`` majority cluster sebagai ground truth
                    # dan menjawab dengan SQL berdasar tebakan tunggal.
                    stage1_uq_signal = dict(_uq)
                    stage1_uq_signal["verdict"] = _uq_payload["verdict"]
                _stage.set_output(_stage_output)
                if rewritten_query and rewritten_query != normalized_message:
                    _stage.set_summary(
                        "Rewriter aktif: pertanyaan ditulis ulang menjadi standalone"
                    )
                else:
                    _stage.set_summary(
                        "Rewriter aktif: pertanyaan tetap (sudah cukup mandiri)"
                    )
        else:
            recorder.skip(
                "question_rewriting",
                "Dilewati: working memory sesi kosong (turn pertama)",
            )

        # Pastikan turn user saat ini direkam ke working memory walaupun
        # ``rewrite()`` di-skip (turn pertama atau session memory masih kosong).
        # Tanpa pencatatan ini, ``add_to_working_memory`` tidak pernah
        # tereksekusi karena guard ``and conversation_history`` di atas
        # membentuk chicken-and-egg: working memory tetap kosong selamanya
        # sehingga turn berikutnya juga skip rewrite, dan inheritance rule
        # (mis. di ``AmbiguityService._is_inheriting_followup``) tidak
        # pernah punya prior user turn untuk dipakai.
        if (
            self._question_rewriting_service is not None
            and not rewrite_invoked
        ):
            self._question_rewriting_service.add_to_working_memory(
                user_id=normalized_user_id,
                session_id=resolved_session_id,
                role="user",
                content=normalized_message,
            )

        # Short-circuit Stage 1: bila rewriter UQ (Stage 1) sudah menilai
        # ambigu karena dangling reference (demonstrative tanpa antecedent
        # di working/episodic memory), maka Stage 2-5 tidak perlu dieksekusi
        # sama sekali — langsung minta klarifikasi ke user dengan opsi
        # referent yang sudah ter-enumerasi pre-check. Ini menjaga
        # konsistensi pipeline (1 ambigu ⇒ stop), mempercepat respons
        # (hemat ~7 detik schema retrieval + Stage 3 LLM + SQL gen),
        # dan menghindari Stage 4 menjawab berbasis tebakan majority
        # cluster ("per provinsi") yang menghilangkan ambiguitas secara
        # artifisial.
        if (
            stage1_uq_signal is not None
            and stage1_uq_signal.get("verdict") == "ambiguous"
            and isinstance(stage1_uq_signal.get("dangling"), dict)
        ):
            _dang = stage1_uq_signal["dangling"]
            _refs = [
                str(r) for r in (_dang.get("referents") or []) if str(r).strip()
            ]
            if _refs:
                _noun = str(_dang.get("noun") or "").strip()
                _dem = str(_dang.get("demonstrative") or "").strip()
                _clar_q = (
                    f"Maaf, '{_noun} {_dem}' di pertanyaan Anda merujuk "
                    f"ke apa? Mohon pilih salah satu interpretasi berikut:"
                ).strip()
                _opts_payload_db = [
                    {"label": r, "description": ""} for r in _refs
                ]
                _short_skip_reason = (
                    "Dilewati: Stage 1 sudah men-detect ambiguitas "
                    "(dangling reference) — pipeline langsung minta "
                    "klarifikasi ke user."
                )
                for _sid in (
                    "schema_retrieval",
                    "ambiguity_detection",
                    "sql_generation",
                    "sql_validation",
                ):
                    recorder.skip(_sid, _short_skip_reason)

                _pending_expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=self._ambiguity_config.session_ttl_seconds
                )
                # Snapshot trace turn ASK: hanya Stage 1 yang benar-benar
                # executed; stage 2-5 di-skip (lihat blok di atas) karena
                # short-circuit dangling. Saat resume, merge logic akan
                # mengganti skip-skip ini dengan eksekusi nyata (Stage 2 fresh
                # + Stage 3 refine + Stage 4 + Stage 5).
                _ask_trace_snapshot = recorder.to_payload()
                _pending_id = await self._repository.create_pending_clarification(
                    user_id=normalized_user_id,
                    session_id=resolved_session_id,
                    standalone_question=normalized_message,
                    schema_context="",
                    relevant_schema={
                        "keywords": [],
                        "predicted_tables": {},
                        "schema_tables": [],
                    },
                    clarification_question=_clar_q,
                    options=_opts_payload_db,
                    ambiguity_type="scope",
                    expires_at=_pending_expires_at,
                    pipeline_trace_snapshot=_ask_trace_snapshot,
                )
                await self._save_ambiguity_metadata(
                    user_id=normalized_user_id,
                    session_id=resolved_session_id,
                    metadata={
                        "was_ambiguous": True,
                        "ambiguity_type": "scope",
                        "clarification_asked": _clar_q,
                        "user_response": None,
                        "interpretation_chosen": None,
                        "auto_resolved": False,
                        "defaulted": False,
                        "pending_id": _pending_id,
                        "source_stage": "question_rewriting",
                    },
                    # Turn ASK clarification: TIDAK ada row baru di
                    # ``chat_messages``. Skip update agar tidak nyangkut ke
                    # turn lama. UI tetap dapat opsi via ``pending_clarification``.
                    persist_to_chat_message=False,
                )

                _options_resp = [
                    {
                        "id": f"opt_{i + 1}",
                        "label": r,
                        "description": "",
                    }
                    for i, r in enumerate(_refs)
                ]
                return {
                    "type": "clarification",
                    "user_id": normalized_user_id,
                    "session_id": resolved_session_id,
                    "question": _clar_q,
                    "query": "",
                    "explanation": "",
                    "options": _options_resp,
                    "pipeline_trace": recorder.to_payload(),
                }

        with recorder.stage("schema_retrieval") as _stage:
            _stage.set_input({"query_for_retrieval": standalone_query})
            prepared = await self._semantic_pipeline.prepare_context(standalone_query)
            _stage.set_output(
                _build_schema_retrieval_output(
                    prepared, include_context_excerpt=True
                )
            )
            _stage.set_summary(
                f"{len(prepared.predicted_tables)} tabel relevan ditemukan"
            )
            _stage.set_metadata(
                {"schema_context_chars": len(prepared.context or "")}
            )

        if self._question_rewriting_service is not None:
            conversation_history = self._question_rewriting_service.get_session_memory_snapshot(
                user_id=normalized_user_id,
                session_id=resolved_session_id,
                max_turns=self._ambiguity_config.max_history_turns,
            )

        # ── Stage 2 UQ: multiple-interpretation schema uncertainty ──
        # Skema candidate sudah diretrieval SEKALI di atas (``prepared``). Di
        # sini LLM meng-generate M interpretasi skema (pemetaan konsep query →
        # tabel/kolom kandidat) @ T>0 → embed teks interpretasi → cluster
        # semantik → entropy gate τ_U. Bila interpretasi bercabang ke >1 yang
        # sama-sama masuk akal, langsung minta klarifikasi (short-circuit
        # Stage 3 LLM detection). Yang di-cluster adalah OUTPUT reasoning LLM,
        # bukan schema candidate/embedding tabel. Dilewati saat ablate_ambiguity
        # (eval) atau fitur nonaktif.
        schema_clarification: AmbiguityDetectionResult | None = None
        schema_clarification_source: str | None = None
        if not ablate_ambiguity:
            # Safeguard DETERMINISTIK (terpisah dari UQ): bila query memuat istilah
            # domain yang memang ambigu (mis. "senior"), kita JAMIN klarifikasi
            # walau sampler UQ konvergen. Verdict UQ tetap dilaporkan apa adanya.
            lexicon_match = self._semantic_pipeline.build_lexicon_clarification(
                standalone_query
            )
            lexicon_term = lexicon_match[0] if lexicon_match else None
            lexicon_clar = lexicon_match[1] if lexicon_match else None

            with recorder.stage("schema_uq") as _stage:
                _stage.set_input({"query": standalone_query})
                schema_uq_result = (
                    await self._semantic_pipeline.assess_schema_uncertainty(
                        standalone_query,
                        prepared=prepared,
                    )
                )
                _stage.set_output(schema_uq_result.to_metadata())
                if schema_uq_result.is_uncertain:
                    _stage.set_summary(
                        f"Interpretasi skema bercabang → minta klarifikasi "
                        f"({schema_uq_result.unique_clusters} interpretasi berbeda; "
                        f"H_norm={schema_uq_result.h_norm:.3f})"
                    )
                elif lexicon_clar is not None:
                    _stage.set_summary(
                        f"UQ tidak memicu, tetapi safeguard istilah-ambigu "
                        f"deterministik cocok ('{lexicon_term}') → minta "
                        f"klarifikasi (BUKAN dari UQ; rule-based)"
                    )
                elif not schema_uq_result.enabled:
                    _stage.set_summary("Dilewati: Stage 2 UQ skema dimatikan")
                elif schema_uq_result.degraded:
                    _stage.set_summary(
                        f"Tidak terukur — {schema_uq_result.degraded_reason} "
                        f"(fail-open, tanpa klarifikasi)"
                    )
                else:
                    _stage.set_summary(
                        f"Interpretasi skema konvergen "
                        f"({schema_uq_result.unique_clusters} interpretasi; "
                        f"H_norm={schema_uq_result.h_norm:.3f})"
                    )

            if schema_uq_result.is_uncertain:
                schema_clarification = (
                    await self._semantic_pipeline.generate_schema_clarification(
                        standalone_query,
                        schema_uq_result,
                        schema_context=prepared.context,
                    )
                )
                schema_clarification_source = "uq"
            elif lexicon_clar is not None:
                schema_clarification = lexicon_clar
                schema_clarification_source = "lexicon"

        if ablate_ambiguity:
            recorder.skip(
                "ambiguity_detection",
                "Dilewati: Stage 3 dimatikan via ablate_stages (eval).",
            )
            detection = AmbiguityDetectionResult(is_ambiguous=False)
        elif schema_clarification is not None:
            _skip_reason = (
                "Dilewati: safeguard istilah-ambigu deterministik (rule-based, "
                "BUKAN UQ) sudah memicu klarifikasi sebelum Stage 3."
                if schema_clarification_source == "lexicon"
                else "Dilewati: Stage 2 UQ skema sudah memicu klarifikasi "
                "sebelum Stage 3."
            )
            recorder.skip("ambiguity_detection", _skip_reason)
            detection = schema_clarification
        else:
            with recorder.stage("ambiguity_detection") as _stage:
                _stage.set_input(
                    {
                        "question": standalone_query,
                        "original_question": normalized_message,
                        "history_turns": len(conversation_history),
                    }
                )
                detection = await self._ambiguity_service.detect(
                    question=standalone_query,
                    schema_context=prepared.context,
                    conversation_history=conversation_history,
                    original_question=normalized_message,
                )
                _ambiguity_output: dict[str, Any] = {
                    "is_ambiguous": detection.is_ambiguous,
                    "ambiguity_type": detection.ambiguity_type,
                    "clarification_question": detection.clarification_question,
                    "options": [
                        {"label": opt.label, "description": opt.description}
                        for opt in (detection.interpretation_options or [])
                    ],
                }
                if detection.h_norm is not None:
                    _ambiguity_output["uncertainty"] = {
                        "h_norm": round(float(detection.h_norm), 4),
                        "tau_u": detection.tau_u,
                        "verdict": (
                            "ambiguous"
                            if float(detection.h_norm) > float(detection.tau_u or 0.40)
                            else "confident"
                        ),
                        "m_samples": detection.m_samples,
                        "unique_clusters": detection.unique_clusters,
                    }
                _stage.set_output(_ambiguity_output)
                # Catatan: kasus Stage 1 ambigu+dangling sudah di-short-circuit
                # sebelum Stage 2 di atas, sehingga blok ini hanya tereksekusi
                # untuk path non-dangling (LLM-based Stage 3 detection).
                if detection.is_ambiguous:
                    _stage.set_summary(
                        f"Ambigu terdeteksi (tipe={detection.ambiguity_type}, "
                        f"{len(detection.interpretation_options or [])} opsi)"
                    )
                else:
                    _stage.set_summary("Tidak ambigu (PASS)")

            # Catatan: sinyal UQ Stage 1 (question_rewriting) sekarang
            # dihitung langsung di QuestionRewritingService.rewrite() via
            # M-sampling rewriter NL→NL dan disurface saat stage tersebut
            # direkam (lihat blok ``with recorder.stage("question_rewriting")``
            # di atas). Block post-hoc copy dari Stage 3 → Stage 1 yang
            # sebelumnya ada di sini SUDAH DIHAPUS — production sekarang
            # mirror persis kalibrasi Stage 1.

        if detection.is_ambiguous:
            if schema_clarification is not None:
                # Stage 2 schema-UQ uncertainty: retrieval skema bercabang ke
                # >1 interpretasi. WAJIB minta klarifikasi eksplisit ke user
                # SEBELUM generate SQL — jangan auto-resolve dari memori
                # episodik (percabangan interpretasi-skema harus dikonfirmasi).
                # Memaksa auto_response=None membuat alur jatuh ke jalur
                # pending-clarification (ASK) di bawah.
                auto_response = None
            else:
                auto_response = await self._try_auto_resolve(
                    user_id=normalized_user_id,
                    detection_ambiguity_type=detection.ambiguity_type,
                    options=detection.interpretation_options,
                    conversation_history=conversation_history,
                )

            if auto_response:
                refine_result = await self._ambiguity_service.refine(
                    original_question=standalone_query,
                    ambiguity_type=detection.ambiguity_type or "scope",
                    clarification_question=detection.clarification_question or "",
                    user_response=auto_response,
                    conversation_history=conversation_history,
                )
                unambiguous_question = refine_result.unambiguous_question

                # Update entry stage 3 (ambiguity_detection) terakhir agar
                # mencakup hasil auto-resolve dari episodic memory.
                if recorder._entries:
                    _last = recorder._entries[-1]
                    if _last.get("stage") == "ambiguity_detection":
                        _last["summary"] = (
                            f"Ambigu (tipe={detection.ambiguity_type}) → "
                            f"diselesaikan otomatis dari memori episodik"
                        )
                        _existing_output = dict(_last.get("output") or {})
                        _existing_output.update(
                            {
                                "auto_resolved": True,
                                "auto_response": auto_response,
                                "unambiguous_question": unambiguous_question,
                            }
                        )
                        _last["output"] = _existing_output

                with recorder.stage("sql_generation") as _stage:
                    _stage.set_input(
                        {
                            "query": unambiguous_question,
                            "schema_context_chars": len(prepared.context or ""),
                            "predicted_tables_count": len(prepared.predicted_tables),
                        }
                    )
                    pipeline_result = await self._semantic_pipeline.run_from_prepared(
                        query=unambiguous_question,
                        prepared=prepared,
                        ablate_active_filter=ablate_active_filter,
                    )
                    _stage.set_output(
                        {
                            "sql": pipeline_result.sql,
                            "explanation": pipeline_result.explanation,
                        }
                    )
                    _stage.set_summary(
                        f"SQL dihasilkan ({len(pipeline_result.sql or '')} karakter)"
                    )
                    _stage.set_metadata(
                        {
                            "note": (
                                "Durasi mencakup eksekusi SQL dan validasi "
                                "semantik (lihat Stage 5 untuk detail)."
                            )
                        }
                    )

                _v_status, _v_summary, _v_output = _build_validation_stage_payload(
                    pipeline_result
                )
                recorder.record_post_hoc(
                    "sql_validation",
                    status="executed",
                    summary=_v_summary,
                    input={"sql": pipeline_result.sql},
                    output=_v_output,
                    metadata={
                        "note": (
                            "Durasi validasi tergabung dalam Stage 4 "
                            "(sql_generation)."
                        )
                    },
                )
                pipeline_trace_payload = recorder.to_payload()

                await self._repository.persist_chat_message(
                    user_id=normalized_user_id,
                    session_id=resolved_session_id,
                    title=normalized_message[:255],
                    question=normalized_message,
                    standalone_question=unambiguous_question,
                    query=pipeline_result.sql,
                    explanation=pipeline_result.explanation,
                    pipeline_trace=pipeline_trace_payload,
                )

                ambiguity_metadata = {
                    "was_ambiguous": True,
                    "ambiguity_type": detection.ambiguity_type,
                    "clarification_asked": detection.clarification_question,
                    "user_response": auto_response,
                    "interpretation_chosen": auto_response,
                    "auto_resolved": True,
                    "defaulted": False,
                }
                await self._save_ambiguity_metadata(
                    user_id=normalized_user_id,
                    session_id=resolved_session_id,
                    metadata=ambiguity_metadata,
                )

                return {
                    "type": "answer",
                    "user_id": normalized_user_id,
                    "session_id": resolved_session_id,
                    "question": unambiguous_question,
                    "query": pipeline_result.sql,
                    "explanation": pipeline_result.explanation,
                    "options": [],
                    **self._validation_payload(pipeline_result),
                    "pipeline_trace": pipeline_trace_payload,
                }

            pending_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=self._ambiguity_config.session_ttl_seconds
            )
            # Snapshot trace turn ASK: Stage 1+2+3 sudah executed di atas
            # (Stage 3 = ambiguity detection yang men-detect ambigu). Stage 4+5
            # baru di-skip di bawah create_pending — tidak ikut snapshot karena
            # tidak informatif. Saat resume, Stage 3 di-replace dengan refine
            # (turn ini) dan Stage 4+5 dieksekusi nyata.
            _ask_trace_snapshot = recorder.to_payload()
            pending_id_created = await self._repository.create_pending_clarification(
                user_id=normalized_user_id,
                session_id=resolved_session_id,
                standalone_question=standalone_query,
                schema_context=prepared.context,
                relevant_schema={
                    "keywords": prepared.keywords,
                    "predicted_tables": {
                        key: {
                            "schema": value.schema,
                            "table": value.table,
                            "score": value.score,
                            "column_scores": value.column_scores,
                        }
                        for key, value in prepared.predicted_tables.items()
                    },
                    "schema_tables": prepared.schema_tables,
                },
                clarification_question=detection.clarification_question or "",
                options=[
                    {"label": opt.label, "description": opt.description}
                    for opt in (detection.interpretation_options or [])
                    if opt.label
                ],
                ambiguity_type=detection.ambiguity_type or "scope",
                expires_at=pending_expires_at,
                pipeline_trace_snapshot=_ask_trace_snapshot,
            )

            ambiguity_metadata = {
                "was_ambiguous": True,
                "ambiguity_type": detection.ambiguity_type,
                "clarification_asked": detection.clarification_question,
                "user_response": None,
                "interpretation_chosen": None,
                "auto_resolved": False,
                "defaulted": False,
                "pending_id": pending_id_created,
            }
            await self._save_ambiguity_metadata(
                user_id=normalized_user_id,
                session_id=resolved_session_id,
                metadata=ambiguity_metadata,
                # Turn ASK clarification (Stage 3 berhenti, no SQL final):
                # TIDAK ada row baru di ``chat_messages``. Skip update agar
                # tidak nyangkut ke turn sebelumnya. Pending clarification
                # tetap dikirim ke UI lewat field ``pending_clarification``.
                persist_to_chat_message=False,
            )

            options_payload = [
                {
                    "id": f"opt_{i + 1}",
                    "label": opt.label,
                    "description": opt.description,
                }
                for i, opt in enumerate(detection.interpretation_options or [])
                if opt.label
            ]

            _clar_skip_reason = (
                "Dilewati: pipeline berhenti pada Stage 3 untuk meminta "
                "klarifikasi user (akan dilanjutkan pada turn berikutnya)"
            )
            recorder.skip(
                "sql_generation",
                _clar_skip_reason,
                metadata={"pending_id": pending_id_created},
            )
            recorder.skip(
                "sql_validation",
                _clar_skip_reason,
                metadata={"pending_id": pending_id_created},
            )

            # Catatan desain: turn ``clarification`` belum lengkap (tidak ada
            # SQL final) sehingga tidak dipersist ke ``chat_messages``.
            # Resolusi user akan dipersist di turn berikutnya (Path B), yang
            # mencatat trace utuh termasuk Stage 3 dengan konteks resolusi.
            return {
                "type": "clarification",
                "user_id": normalized_user_id,
                "session_id": resolved_session_id,
                "question": detection.clarification_question or "",
                "query": "",
                "explanation": "",
                "options": options_payload,
                "pipeline_trace": recorder.to_payload(),
            }

        with recorder.stage("sql_generation") as _stage:
            _stage.set_input(
                {
                    "query": standalone_query,
                    "schema_context_chars": len(prepared.context or ""),
                    "predicted_tables_count": len(prepared.predicted_tables),
                }
            )
            pipeline_result = await self._semantic_pipeline.run_from_prepared(
                query=standalone_query,
                prepared=prepared,
                ablate_active_filter=ablate_active_filter,
            )
            _stage.set_output(
                {
                    "sql": pipeline_result.sql,
                    "explanation": pipeline_result.explanation,
                }
            )
            _stage.set_summary(
                f"SQL dihasilkan ({len(pipeline_result.sql or '')} karakter)"
            )
            _stage.set_metadata(
                {
                    "note": (
                        "Durasi mencakup eksekusi SQL dan validasi semantik "
                        "(lihat Stage 5 untuk detail)."
                    )
                }
            )

        query = pipeline_result.sql
        explanation = pipeline_result.explanation

        _v_status, _v_summary, _v_output = _build_validation_stage_payload(
            pipeline_result
        )
        recorder.record_post_hoc(
            "sql_validation",
            status="executed",
            summary=_v_summary,
            input={"sql": pipeline_result.sql},
            output=_v_output,
            metadata={
                "note": (
                    "Durasi validasi tergabung dalam Stage 4 (sql_generation)."
                )
            },
        )
        pipeline_trace_payload = recorder.to_payload()

        await self._repository.persist_chat_message(
            user_id=normalized_user_id,
            session_id=resolved_session_id,
            title=normalized_message[:255],
            question=normalized_message,
            standalone_question=standalone_query,
            query=query,
            explanation=explanation,
            pipeline_trace=pipeline_trace_payload,
        )

        # Inject jawaban final ke working memory sebagai assistant turn agar
        # turn berikutnya melihat KONTEKS NYATA (bukan placeholder generik).
        # Tanpa ini, follow-up question seperti "berapa jumlahnya di lokasi
        # tersebut?" akan kehilangan grounding subject (mis. "pegawai S2")
        # dan LLM rewriter berhalusinasi ke domain lain. Pakai ``explanation``
        # (NL summary jawaban) bukan SQL mentah supaya prompt rewriter tetap
        # bisa di-embed dengan baik secara semantik.
        if (
            self._question_rewriting_service is not None
            and isinstance(explanation, str)
            and explanation.strip()
        ):
            try:
                self._question_rewriting_service.add_to_working_memory(
                    user_id=normalized_user_id,
                    session_id=resolved_session_id,
                    role="assistant",
                    content=explanation.strip(),
                )
            except Exception:
                log.exception(
                    "Gagal menyimpan assistant turn (explanation) ke working memory"
                )

        ambiguity_metadata = {
            "was_ambiguous": False,
            "ambiguity_type": None,
            "clarification_asked": None,
            "user_response": None,
            "interpretation_chosen": None,
            "auto_resolved": False,
        }
        await self._save_ambiguity_metadata(
            user_id=normalized_user_id,
            session_id=resolved_session_id,
            metadata=ambiguity_metadata,
        )

        return {
            "type": "answer",
            "user_id": normalized_user_id,
            "session_id": resolved_session_id,
            "question": standalone_query,
            "query": query,
            "explanation": explanation,
            "options": [],
            **self._validation_payload(pipeline_result),
            "pipeline_trace": pipeline_trace_payload,
        }

    async def get_session_messages(
        self, user_id: str, session_id: str
    ) -> Optional[dict[str, Any]]:
        """Get persisted session details and conversations for a user."""
        normalized_user_id = str(user_id).strip()
        normalized_session_id = session_id.strip()

        if not normalized_user_id:
            raise ValueError("user_id cannot be empty")
        if not normalized_session_id:
            raise ValueError("session_id cannot be empty")

        if self._repository is None:
            raise RuntimeError("Chat repository is not configured")

        await self._ensure_chat_storage_ready()
        return await self._repository.get_chat_session_messages(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
        )

    async def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """List all persisted sessions that belong to a user."""
        normalized_user_id = str(user_id).strip()

        if not normalized_user_id:
            raise ValueError("user_id cannot be empty")

        if self._repository is None:
            raise RuntimeError("Chat repository is not configured")

        await self._ensure_chat_storage_ready()
        return await self._repository.list_chat_sessions(user_id=normalized_user_id)

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete a persisted session if it belongs to the requesting user."""
        normalized_user_id = str(user_id).strip()
        normalized_session_id = session_id.strip()

        if not normalized_user_id:
            raise ValueError("user_id cannot be empty")
        if not normalized_session_id:
            raise ValueError("session_id cannot be empty")

        if self._repository is None:
            raise RuntimeError("Chat repository is not configured")

        await self._ensure_chat_storage_ready()

        deleted = await self._repository.delete_chat_session(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
        )
        if not deleted:
            return False

        QuestionRewritingService.clear_session_memory(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
        )
        return True

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed text using the embeddings model"""
        vector = self._llm_adapter.embeddings.embed_query(text)
        return vector

    async def import_base_knowledge_csv(
        self,
        csv_path: str = "data/base_knowledge_rasl.csv",
        table_name: str | None = None,
        batch_size: int = 50,
        truncate_before_insert: bool = True,
    ) -> dict[str, Any]:
        """Import CSV base knowledge, embed its content, then upsert to pgvector table."""
        if self._repository is None:
            raise RuntimeError("Semantic pipeline is not configured")

        resolved_table = self._resolve_vector_table_name(table_name)
        if not resolved_table:
            raise ValueError("table_name cannot be empty")

        log.info(
            "🚀 Starting CSV import to vector table=%s csv_path=%s batch_size=%s truncate=%s",
            resolved_table,
            csv_path,
            batch_size,
            truncate_before_insert,
        )

        csv_file = Path(csv_path).expanduser()
        if not csv_file.is_absolute():
            csv_file = (Path.cwd() / csv_file).resolve()
        if not csv_file.exists() or not csv_file.is_file():
            raise ValueError(f"CSV file not found: {csv_file}")

        resolved_batch_size = max(1, min(500, int(batch_size)))

        await self._repository.ensure_base_knowledge_vector_table(resolved_table)
        if truncate_before_insert:
            await self._repository.truncate_vector_table(resolved_table)

        expected_dimensions = await self._repository.get_vector_table_embedding_dimensions(
            resolved_table
        )

        with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            csv_rows = list(reader)

        total_rows = len(csv_rows)
        targeted_rows = total_rows

        processed_rows = 0
        inserted_rows = 0
        failed_rows = 0
        sample_errors: list[str] = []
        pending_rows: list[dict[str, Any]] = []

        for row in csv_rows[:targeted_rows]:
            processed_rows += 1
            # Prefer kolom `entity_text` (skema RASL baru), fallback ke `content`
            # untuk backward compat dengan CSV base_knowledge lama.
            content = str(
                row.get("entity_text") or row.get("content") or ""
            ).strip()

            if not content:
                failed_rows += 1
                if len(sample_errors) < 10:
                    sample_errors.append(
                        f"row={processed_rows}: empty content"
                    )
                continue

            column_name_raw = str(row.get("column_name") or "").strip()

            # Derive schema_name + table_name dari kolom `table_name` raw.
            # Skema RASL baru menyimpan ini sebagai "public.propinsi_tm".
            # Jika sudah ada kolom `schema_name` eksplisit (CSV lama),
            # gunakan apa adanya tanpa split.
            raw_table_ref = str(row.get("table_name") or "").strip()
            explicit_schema = str(row.get("schema_name") or "").strip()
            if explicit_schema:
                derived_schema = explicit_schema or None
                derived_table = raw_table_ref or None
            elif "." in raw_table_ref:
                schema_part, _, table_part = raw_table_ref.partition(".")
                derived_schema = schema_part.strip() or None
                derived_table = table_part.strip() or None
            else:
                derived_schema = None
                derived_table = raw_table_ref or None

            # Derive entity_type: "table" untuk baris ringkasan tabel
            # (column_name kosong), "column" untuk baris kolom. Override
            # bila CSV menyimpan entity_type eksplisit (mis. value entity).
            explicit_entity_type = str(row.get("entity_type") or "").strip()
            if explicit_entity_type:
                derived_entity_type = explicit_entity_type
            else:
                derived_entity_type = "table" if not column_name_raw else "column"

            try:
                embedding = self._llm_adapter.embeddings.embed_query(content)
                if not embedding:
                    raise ValueError("empty embedding returned")
                pending_rows.append(
                    {
                        "entity_type": derived_entity_type,
                        "schema_name": derived_schema,
                        "table_name": derived_table,
                        "table_description": str(
                            row.get("table_description") or ""
                        ).strip()
                        or None,
                        "column_name": column_name_raw or None,
                        "column_alias": str(row.get("column_alias") or "").strip() or None,
                        "column_description": str(
                            row.get("column_description") or ""
                        ).strip()
                        or None,
                        "value_description": str(
                            row.get("value_description") or ""
                        ).strip()
                        or None,
                        "content": content,
                        "embedding": embedding,
                    }
                )
            except Exception as exc:
                failed_rows += 1
                if len(sample_errors) < 10:
                    sample_errors.append(f"row={processed_rows}: {exc}")

            if len(pending_rows) >= resolved_batch_size:
                try:
                    inserted_rows += await self._repository.insert_base_knowledge_rows(
                        vector_table=resolved_table,
                        rows=pending_rows,
                        expected_dimensions=expected_dimensions,
                    )
                except Exception as exc:
                    failed_rows += len(pending_rows)
                    if len(sample_errors) < 10:
                        sample_errors.append(f"batch@row={processed_rows}: {exc}")
                finally:
                    pending_rows = []

        if pending_rows:
            try:
                inserted_rows += await self._repository.insert_base_knowledge_rows(
                    vector_table=resolved_table,
                    rows=pending_rows,
                    expected_dimensions=expected_dimensions,
                )
            except Exception as exc:
                failed_rows += len(pending_rows)
                if len(sample_errors) < 10:
                    sample_errors.append(f"final-batch: {exc}")

        result = {
            "table_name": resolved_table,
            "csv_path": str(csv_file),
            "embedding_model": settings.AI_EMBEDDINGS_MODEL_NAME,
            "total_rows": total_rows,
            "targeted_rows": targeted_rows,
            "processed_rows": processed_rows,
            "inserted_rows": inserted_rows,
            "failed_rows": failed_rows,
            "truncated": bool(truncate_before_insert),
            "expected_dimensions": expected_dimensions,
            "sample_errors": sample_errors,
        }

        log.info(
            "✅ CSV import finished table=%s inserted=%s failed=%s targeted=%s",
            resolved_table,
            inserted_rows,
            failed_rows,
            targeted_rows,
        )
        if failed_rows > 0 and sample_errors:
            log.warning(
                "CSV import encountered errors table=%s samples=%s",
                resolved_table,
                sample_errors[:3],
            )

        return result


def get_llm_adapter() -> LLMAdapter:
    """Dependency untuk mendapatkan LLMAdapter"""
    return init_llm()


# dependencies injection
def get_chatbot_service(
    llm_adapter: LLMAdapter = Depends(get_llm_adapter),
) -> ChatbotService:
    """Dependency untuk mendapatkan ChatbotService"""
    return ChatbotService(llm_adapter)


async def get_chatbot_pipeline_service(
    llm_adapter: LLMAdapter = Depends(get_llm_adapter),
    db: AsyncSession = Depends(get_db),
) -> ChatbotService:
    """Dependency untuk mendapatkan ChatbotService dengan semantic pipeline."""
    repository = ChatbotRepository(db)
    return ChatbotService(llm_adapter=llm_adapter, repository=repository)
