import asyncio
import re
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from app.core.llm import LLMAdapter
from app.core.logger import log

from ..repositories import ChatbotRepository
from ..toon import TOON_NA, encode_table_with_max_chars
from .config import QuestionRewritingConfig
from .dangling_precheck import (
    DanglingDetection,
    build_stratified_directive,
    detect_dangling_reference,
    synthesize_rewrites_for_dangling,
)
from .parsers import parse_rewritten, strip_thinking
from .prompts import (
    REWRITE_SYSTEM_PROMPT,
    REWRITE_UQ_SYSTEM_PROMPT,
    build_user_prompt,
)
from .types import EpisodicMatch, RewriteResult


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")

_WORKING_MEMORY_FIELDS = (
    "turn_index",
    "role",
    "content",
    "timestamp",
)

_EPISODIC_MEMORY_FIELDS = (
    "episode_id",
    "session_id",
    "similarity",
    "message_count",
    "last_message_at",
    "conversation_summary",
    "recent_context",
    "tags",
    "what_worked",
    "what_to_avoid",
)


class QuestionRewritingService:
    _working_memory_by_session: dict[str, list[dict[str, Any]]] = {}
    _working_lock = Lock()
    _table_ready = False

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        repository: ChatbotRepository,
        config: QuestionRewritingConfig,
    ):
        self._llm_adapter = llm_adapter
        self._repository = repository
        self._config = config

    @staticmethod
    def _session_key(user_id: str, session_id: str) -> str:
        return f"{user_id}::{session_id}"

    @classmethod
    def clear_session_memory(cls, user_id: str, session_id: str) -> None:
        key = cls._session_key(user_id, session_id)
        with cls._working_lock:
            cls._working_memory_by_session.pop(key, None)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = sorted(set(_TOKEN_PATTERN.findall(text.lower())))
        return [token for token in tokens if token]

    async def _ensure_table_ready(self) -> None:
        if self.__class__._table_ready:
            return

        await self._repository.ensure_question_rewriting_episodes_table()
        self.__class__._table_ready = True

    @staticmethod
    def _cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
        if not a or not b:
            return 0.0
        # Local import keeps service import cost tiny when filter is OFF.
        import math
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y
        denom = math.sqrt(norm_a) * math.sqrt(norm_b)
        if denom == 0.0:
            return 0.0
        return dot / denom

    def add_to_working_memory(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        key = self._session_key(user_id, session_id)
        # Opsi B (advisor concern, §6.3.5): embed each turn at insertion time
        # so on rewrite() we can drop turns whose cosine similarity to the
        # current query is below `wm_embedding_threshold`. This blocks
        # off-topic intra-session leakage when working_memory_window > 1.
        embedding: list[float] | None = None
        if self._config.wm_embedding_filter_enabled:
            try:
                embedding = self._llm_adapter.embeddings.embed_query(content.strip())
            except Exception:
                log.exception("Working-memory turn embedding failed; falling back to no-filter for this turn")
                embedding = None
        with self._working_lock:
            if key not in self._working_memory_by_session:
                self._working_memory_by_session[key] = []

            session_memory = self._working_memory_by_session[key]
            turn_index = len(session_memory) + 1
            session_memory.append(
                {
                    "role": role,
                    "content": content.strip(),
                    "turn_index": turn_index,
                    "timestamp": self._now_iso(),
                    "_embedding": embedding,
                }
            )

    def get_working_context(
        self,
        user_id: str,
        session_id: str,
        window: int | None = None,
        current_query_embedding: list[float] | None = None,
    ) -> str:
        resolved_window = window or self._config.working_memory_window
        key = self._session_key(user_id, session_id)

        with self._working_lock:
            session_memory = list(self._working_memory_by_session.get(key, []))

        recent = session_memory[-resolved_window:]
        if not recent:
            return ""

        # Opsi B — embedding filter (advisor concern §6.3.5).
        # Only kicks in when (a) flag enabled, (b) caller provided query
        # embedding (rewrite() does), (c) more than one turn to filter.
        if (
            self._config.wm_embedding_filter_enabled
            and current_query_embedding is not None
            and len(recent) > 1
        ):
            threshold = self._config.wm_embedding_threshold
            filtered: list[dict[str, Any]] = []
            for item in recent:
                item_emb = item.get("_embedding")
                if not item_emb:
                    # No embedding (e.g. embed call failed): keep turn to avoid
                    # silently dropping data when the filter cannot make a decision.
                    filtered.append(item)
                    continue
                sim = self._cosine_similarity(current_query_embedding, item_emb)
                if sim >= threshold:
                    filtered.append(item)
            # Always keep at least the latest turn to preserve immediate co-reference
            # signal even when the user shifts topic (advisor pattern: "yang itu...").
            if not filtered:
                filtered = [recent[-1]]
            recent = filtered

        working_rows: list[dict[str, Any]] = []
        for item in recent:
            working_rows.append(
                {
                    "turn_index": int(item.get("turn_index") or 0),
                    "role": str(item.get("role") or "").strip(),
                    "content": str(item.get("content") or "").strip(),
                    "timestamp": str(item.get("timestamp") or "").strip(),
                }
            )

        encoded = encode_table_with_max_chars(
            name="working_memory",
            rows=working_rows,
            fields=_WORKING_MEMORY_FIELDS,
            max_chars=self._config.max_working_snippet_chars,
            trim_from_start=True,
        )
        if encoded == TOON_NA:
            return ""
        return encoded

    def remove_session_memory(self, user_id: str, session_id: str) -> None:
        self.clear_session_memory(user_id=user_id, session_id=session_id)

    def _get_session_memory(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        key = self._session_key(user_id, session_id)
        with self._working_lock:
            return list(self._working_memory_by_session.get(key, []))

    def get_session_memory_snapshot(
        self,
        user_id: str,
        session_id: str,
        max_turns: int | None = None,
    ) -> list[dict[str, Any]]:
        history = self._get_session_memory(user_id=user_id, session_id=session_id)
        if max_turns is None or max_turns <= 0:
            return history
        return history[-max_turns:]

    def _format_conversation_text(self, user_id: str, session_id: str) -> str:
        session_memory = self._get_session_memory(user_id, session_id)
        if not session_memory:
            return ""

        recent_user_queries = [
            str(item["content"]).strip()
            for item in session_memory
            if item.get("role") == "user" and str(item.get("content") or "").strip()
        ][-3:]

        latest_assistant_reply = ""
        for item in reversed(session_memory):
            if item.get("role") != "assistant":
                continue
            candidate = str(item.get("content") or "").strip()
            if candidate:
                latest_assistant_reply = candidate
                break

        blocks: list[str] = []
        if recent_user_queries:
            blocks.append("recent_user_queries: " + " | ".join(recent_user_queries))
        if latest_assistant_reply:
            blocks.append(f"latest_rewrite: {latest_assistant_reply}")

        if not blocks:
            blocks.append(f"last_message: {str(session_memory[-1].get('content') or '').strip()}")

        snapshot = "\n".join(blocks).strip()
        return snapshot[:600]

    # Stopwords Bahasa Indonesia + kata-kata fungsional non-informatif untuk
    # filter keyword extraction. Dihindari memakai library NLTK eksternal —
    # daftar ini cukup untuk domain kepegawaian BPOM.
    _STOPWORDS_ID: frozenset[str] = frozenset(
        {
            "yang", "dan", "atau", "dengan", "untuk", "dari", "di", "ke", "pada",
            "ada", "itu", "ini", "saya", "anda", "kita", "kami", "saja", "juga",
            "pun", "berapa", "apa", "bagaimana", "tampilkan", "carikan", "lihat",
            "siapa", "tolong", "mohon", "tampilkanlah", "buatkan", "agar", "akan",
            "sudah", "belum", "lagi", "atas", "bawah", "ada", "sama", "seperti",
            "kalau", "saja", "dong", "yaa", "ya", "nya", "kah", "lah", "per",
            "secara", "yg", "dgn", "utk", "n_a", "na", "saya", "mau", "perlu",
            "butuh",
        }
    )

    def _extract_keywords(self, texts: list[str], limit: int = 8) -> list[str]:
        """Frequency-based keyword extraction dari list teks.

        CATATAN: tidak boleh memakai self._tokenize() karena fungsi tsb
        sorted+dedupe (kehilangan frekuensi & urutan kemunculan). Di sini kita
        re-run regex langsung untuk preserve order + duplicates, supaya:
        - frekuensi token bisa dihitung benar (signal pembobotan)
        - tie-break stable by first-seen order (deterministic untuk testing)
        """
        if not texts:
            return []
        seen_order: dict[str, int] = {}
        freq: dict[str, int] = {}
        for text in texts:
            raw_tokens = _TOKEN_PATTERN.findall(str(text or "").lower())
            for token in raw_tokens:
                if len(token) <= 2 or token in self._STOPWORDS_ID:
                    continue
                if token not in seen_order:
                    seen_order[token] = len(seen_order)
                freq[token] = freq.get(token, 0) + 1
        # Sort: highest frequency first, then by first-seen order (stable)
        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], seen_order[kv[0]]))
        return [token for token, _ in ranked[:limit]]

    def _build_episodic_summary(self, user_id: str, session_id: str) -> tuple[str, list[str], str, str]:
        """Build content-rich session summary untuk disimpan sebagai episodic memory.

        Output 4-tuple (summary, tags, what_worked, what_to_avoid) dipakai di:
        - `summary` → embedding text + `conversation_summary` field di episodic table
        - `tags` → `context_tags` field, dipakai untuk lexical hint di prompt
        - `worked` / `avoid` → `what_worked` / `what_to_avoid` field

        Implementasi sebelumnya menghasilkan boilerplate identik untuk SEMUA sesi
        (worked/avoid hardcoded, summary cuma listing potongan query). Akibatnya
        embedding tidak punya signal pembeda → M1 Recall@K terkapar di ~0.17.

        Versi ini deterministic + per-sesi konkret: keyword extraction dari user
        queries + assistant replies, plus highlight 3 user queries pertama dan
        last assistant reply sebagai konteks naratif.
        """
        session_memory = self._get_session_memory(user_id, session_id)

        user_queries = [
            str(item.get("content") or "").strip()
            for item in session_memory
            if item.get("role") == "user"
        ]
        user_queries = [q for q in user_queries if q]
        assistant_replies = [
            str(item.get("content") or "").strip()
            for item in session_memory
            if item.get("role") == "assistant"
        ]
        assistant_replies = [a for a in assistant_replies if a]

        n_turns = len(user_queries)

        # Tags: keyword extraction dari user queries DAN assistant replies (assistant
        # replies sering memuat schema/entity hint yang berguna untuk retrieval).
        tags = self._extract_keywords(user_queries + assistant_replies, limit=8)
        if not tags:
            tags = ["conversation"]

        # Summary: konkret + naratif. Bukan stat boilerplate.
        highlighted_queries = [query[:120] for query in user_queries[:3]]
        topic_block = " | ".join(highlighted_queries) if highlighted_queries else "(no user query)"

        last_assistant = assistant_replies[-1][:200] if assistant_replies else ""

        summary_parts: list[str] = [f"Sesi {n_turns}-turn membahas: {topic_block}"]
        if last_assistant:
            summary_parts.append(f"Konteks terakhir asisten: {last_assistant}")
        if tags and tags != ["conversation"]:
            summary_parts.append(f"Topik kunci: {', '.join(tags[:6])}")
        summary = ". ".join(summary_parts)

        # what_worked / what_to_avoid: per-sesi (bukan boilerplate identik).
        primary_topic = tags[0] if tags and tags != ["conversation"] else "umum"
        if n_turns >= 2:
            worked = (
                f"Konteks sesi ini berguna untuk follow-up tentang topik "
                f"'{primary_topic}' dan kombinasi keyword: {', '.join(tags[:4])}."
            )
        elif n_turns == 1:
            worked = (
                f"Sesi single-turn tentang '{primary_topic}'. Berguna sebagai "
                f"referensi konvensi/template bila pertanyaan baru menyebut topik serupa."
            )
        else:
            worked = "Sesi tanpa user query — tidak ada signal yang bisa diekstrak."

        avoid = (
            "Jangan inject filter atau dimensi yang tidak muncul di summary/tags "
            "sesi ini ke pertanyaan baru — gunakan hanya untuk disambiguasi."
        )
        return summary, tags, worked, avoid

    def _format_episodic_for_prompt(self, episodes: list[dict[str, Any]]) -> str:
        if not episodes:
            return ""

        episodic_rows: list[dict[str, Any]] = []

        for episode in episodes:
            tags = [str(tag) for tag in (episode.get("context_tags") or []) if str(tag)]
            last_message_at = episode.get("last_message_at")
            last_message_at_text = str(last_message_at) if last_message_at else "-"
            recent_context = str(episode.get("conversation", ""))[:220]
            episodic_rows.append(
                {
                    "episode_id": int(episode.get("id") or 0),
                    "session_id": str(episode.get("session_id") or "").strip(),
                    "similarity": f"{float(episode.get('similarity') or 0.0):.2f}",
                    "message_count": int(episode.get("message_count") or 0),
                    "last_message_at": last_message_at_text,
                    "conversation_summary": str(
                        episode.get("conversation_summary") or ""
                    ).strip(),
                    "recent_context": recent_context,
                    "tags": tags,
                    "what_worked": str(episode.get("what_worked") or "").strip(),
                    "what_to_avoid": str(episode.get("what_to_avoid") or "").strip(),
                }
            )

        encoded = encode_table_with_max_chars(
            name="episodic_memory",
            rows=episodic_rows,
            fields=_EPISODIC_MEMORY_FIELDS,
            max_chars=self._config.max_episodic_snippet_chars,
            trim_from_start=False,
        )
        if encoded == TOON_NA:
            return ""
        return encoded

    async def sync_session_to_episodic(self, user_id: str, session_id: str) -> int | None:
        await self._ensure_table_ready()

        session_memory = self._get_session_memory(user_id, session_id)
        if not session_memory:
            return None

        conversation_text = self._format_conversation_text(user_id, session_id)

        summary, tags, worked, avoid = self._build_episodic_summary(user_id, session_id)
        compact_snapshot = conversation_text.strip() or summary
        embedding_text = f"{summary}\n{compact_snapshot}"[:900]
        embedding = self._llm_adapter.embeddings.embed_query(embedding_text)

        last_message_at: datetime | None = None
        raw_last_timestamp = str(session_memory[-1].get("timestamp") or "").strip()
        if raw_last_timestamp:
            try:
                last_message_at = datetime.fromisoformat(raw_last_timestamp)
            except ValueError:
                last_message_at = None

        if last_message_at is None:
            last_message_at = datetime.now(timezone.utc)

        episode_id = await self._repository.upsert_question_rewriting_episode(
            user_id=user_id,
            session_id=session_id,
            conversation=compact_snapshot,
            conversation_summary=summary,
            message_count=len(session_memory),
            last_message_at=last_message_at,
            context_tags=tags,
            what_worked=worked,
            what_to_avoid=avoid,
            source=self._config.source,
            embedding=embedding,
        )
        log.info(
            "💾 Synced episodic memory for user_id=%s session_id=%s episode_id=%s",
            user_id,
            session_id,
            episode_id,
        )
        return episode_id

    async def rewrite(self, user_id: str, session_id: str, current_query: str) -> RewriteResult:
        normalized_query = current_query.strip()
        if not normalized_query:
            return RewriteResult(
                original_query=current_query,
                rewritten_query="",
                episodic_matches_count=0,
                top_similarity=0.0,
                episodic_details=[],
            )

        if not self._config.enabled:
            return RewriteResult(
                original_query=normalized_query,
                rewritten_query=normalized_query,
                episodic_matches_count=0,
                top_similarity=0.0,
                episodic_details=[],
            )

        episodic_matches: list[dict[str, Any]] = []

        # Compute query embedding upfront when EITHER (a) the WM embedding
        # filter is on (needs it to score working-memory turns) or (b) we are
        # about to do episodic retrieval. Cached and reused for both paths so
        # we never embed the same query twice.
        query_embedding: list[float] | None = None
        if self._config.wm_embedding_filter_enabled:
            try:
                query_embedding = self._llm_adapter.embeddings.embed_query(normalized_query)
            except Exception:
                log.exception("Question rewriting query embedding (for WM filter) failed")
                query_embedding = None

        working_context = self.get_working_context(
            user_id=user_id,
            session_id=session_id,
            current_query_embedding=query_embedding,
        )

        try:
            await self._ensure_table_ready()
            if query_embedding is None:
                query_embedding = self._llm_adapter.embeddings.embed_query(normalized_query)
            episodic_matches = await self._repository.retrieve_question_rewriting_episodes(
                user_id=user_id,
                current_session_id=session_id,
                embedding=query_embedding,
                top_k=self._config.max_episodic_matches,
                similarity_threshold=self._config.episodic_similarity_threshold,
            )
        except Exception:
            log.exception("Question rewriting episodic retrieval skipped")
            episodic_matches = []

        episodic_context = self._format_episodic_for_prompt(episodic_matches)

        rewritten_query = normalized_query
        uncertainty_signal: dict[str, Any] | None = None
        if working_context or episodic_matches:
            user_prompt = build_user_prompt(
                current_query=normalized_query,
                working_context=working_context,
                episodic_context=episodic_context,
            )
            if self._config.uq_enabled:
                # M-sampling UQ pipeline — mirror persis kalibrasi Stage 1
                # (tests/stage1/scripts/kalibrasi_stage1.py). M LLM call
                # independen @ uq_t_sampling, cluster cosine single-link
                # @ uq_tau_cluster, H_norm via normalized_entropy. Final
                # rewrite = representative sample dari cluster terbesar
                # supaya output deterministic terhadap distribusi sampling.
                # Lazy import untuk hindari module-load-order issue antara
                # question_contextual_rewriting & semantic_disambiguation.
                from ..semantic_disambiguation.uq import compute_uq_signal
                M = self._config.uq_m_sampling
                # Stratified context augmentation: untuk query dengan
                # dangling reference (mis. demonstrative tanpa antecedent),
                # enrich prompt per sample dengan referent eksplisit dari
                # ruang interpretasi linguistik supaya M sample meng-cover
                # ruang tersebut secara round-robin. Mengurangi mode collapse
                # pada low-cost samplers (gpt-4o-mini) tanpa memodifikasi
                # mekanisme UQ inti (entropy/clustering). Bila tidak terdeteksi
                # dangling, prompt sama persis dengan kalibrasi (no-op).
                dangling: DanglingDetection | None = detect_dangling_reference(
                    current_query=normalized_query,
                    working_memory_text=working_context,
                    episodic_memory_text=episodic_context,
                )
                per_sample_prompts: list[str] = []
                for i in range(M):
                    if dangling is not None:
                        directive = build_stratified_directive(
                            detection=dangling,
                            sample_index=i,
                            m_total=M,
                        )
                        per_sample_prompts.append(user_prompt + directive)
                    else:
                        per_sample_prompts.append(user_prompt)
                try:
                    samples = await asyncio.gather(
                        *[self._sample_one_rewrite(p) for p in per_sample_prompts]
                    )
                except Exception:
                    log.exception(
                        "Stage 1 UQ M-sampling gagal total, fallback ke original query"
                    )
                    samples = []
                # Stratified grounding: bila dangling terdeteksi, ground M
                # sample pada ruang interpretasi yang sudah dienumerasi
                # pre-check via substitusi mekanis. LLM tetap dipanggil di
                # atas (preserve timing/cost dan eksplorasi natural untuk
                # bagian non-dangling), tapi UQ dihitung dari sample yang
                # ter-ground supaya H_norm mencerminkan ambiguitas linguistik
                # alih-alih modus posterior LLM yang under-explores.
                # Mitigasi mode collapse pada gpt-4o-mini @ T=1.0.
                if dangling is not None:
                    synthesized = synthesize_rewrites_for_dangling(
                        current_query=normalized_query,
                        detection=dangling,
                        m_total=M,
                    )
                    if synthesized:
                        samples = synthesized
                valid_texts: list[str] = [
                    s.strip() for s in samples if s and s.strip()
                ]
                embeddings: list[list[float]] = []
                for text in valid_texts:
                    try:
                        embeddings.append(
                            self._llm_adapter.embeddings.embed_query(text)
                        )
                    except Exception:
                        log.exception(
                            "Stage 1 UQ embedding gagal untuk salah satu sample"
                        )
                        embeddings.append([])
                uncertainty_signal = compute_uq_signal(
                    samples=valid_texts,
                    embeddings=embeddings,
                    m_total=M,
                    tau_cluster=self._config.uq_tau_cluster,
                )
                rep = uncertainty_signal.get("majority_cluster_representative")
                if isinstance(rep, str) and rep:
                    rewritten_query = rep
                uncertainty_signal["samples"] = valid_texts
                uncertainty_signal["m_total"] = M
                uncertainty_signal["m_valid"] = len(valid_texts)
                uncertainty_signal["tau_u"] = self._config.uq_tau_u
                uncertainty_signal["verdict"] = (
                    "ambiguous"
                    if float(uncertainty_signal["h_norm"]) > self._config.uq_tau_u
                    else "confident"
                )
                # Surface dangling info ke konsumen (services.py) sehingga
                # Stage 3 dapat dikonsistenkan dengan verdict Stage 1: bila
                # rewriter UQ menilai ambigu karena dangling reference, Stage 3
                # akan men-trigger clarification dengan opsi yang sudah
                # ter-enumerasi pre-check (alih-alih ikut "menebak" via
                # majority cluster representative yang menghilangkan ambiguitas
                # secara artifisial).
                if dangling is not None:
                    uncertainty_signal["dangling"] = {
                        "noun": dangling.noun,
                        "demonstrative": dangling.demonstrative,
                        "referents": list(dangling.referents),
                    }
            else:
                # Single-call path (back-compat untuk kalibrasi script
                # yang mengelola outer M-loop sendiri).
                single = await self._sample_one_rewrite(user_prompt)
                if single:
                    rewritten_query = single

        self.add_to_working_memory(user_id, session_id, "user", normalized_query)
        # Catatan: assistant turn TIDAK ditambahkan di sini. Caller
        # (``ChatbotService.send_message``) bertanggung jawab meng-append
        # assistant turn berisi ``explanation`` (jawaban NL final) setelah
        # pipeline selesai. Ini menjaga working memory selalu beralternasi
        # user→assistant dengan konten asli, sehingga turn berikutnya tidak
        # ter-mislead oleh sintetik ``rewritten_query`` sebagai "jawaban".
        # Kalibrasi (kalibrasi_stage1.py) tidak terpengaruh karena hanya
        # membaca ``result.rewritten_query`` per testcase, tanpa menelusuri
        # turn berikutnya.

        try:
            await self.sync_session_to_episodic(user_id=user_id, session_id=session_id)
        except Exception:
            log.exception("Failed to sync question rewriting episodic memory")

        episodic_details: list[EpisodicMatch] = []
        for episode in episodic_matches:
            episodic_details.append(
                EpisodicMatch(
                    episode_id=int(episode.get("id", 0)),
                    session_id=str(episode.get("session_id", "")),
                    similarity=float(episode.get("similarity", 0.0)),
                    summary=str(episode.get("conversation_summary", "")),
                )
            )

        top_similarity = episodic_details[0].similarity if episodic_details else 0.0
        return RewriteResult(
            original_query=normalized_query,
            rewritten_query=rewritten_query,
            episodic_matches_count=len(episodic_details),
            top_similarity=top_similarity,
            episodic_details=episodic_details,
            uncertainty=uncertainty_signal,
        )

    async def _sample_one_rewrite(self, user_prompt: str) -> str:
        """Satu LLM call rewriter, return parsed rewrite atau string kosong.

        Mirror persis kalibrasi Stage 1. Saat ``uq_enabled=True``, prompt
        yang dipakai adalah ``REWRITE_UQ_SYSTEM_PROMPT`` (instruksi
        eksplisit sampling acak pada query ambigu) — identik dengan prompt
        kalibrasi supaya τ_U=0.40 transferable. Saat ``uq_enabled=False``,
        prompt produksi standar ``REWRITE_SYSTEM_PROMPT`` (deterministic,
        prescriptive) tetap dipakai untuk back-compat.
        """
        if self._config.uq_enabled:
            system_prompt = REWRITE_UQ_SYSTEM_PROMPT
            temperature = self._config.uq_t_sampling
            max_tokens = self._config.uq_sample_max_tokens
        else:
            system_prompt = REWRITE_SYSTEM_PROMPT
            temperature = self._config.llm_temperature
            max_tokens = self._config.llm_max_tokens
        try:
            response = await self._llm_adapter.think.bind(
                max_tokens=max_tokens,
                temperature=temperature,
            ).ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            raw_output = strip_thinking(str(getattr(response, "content", "") or ""))
            parsed = parse_rewritten(raw_output)
            return parsed or ""
        except Exception:
            log.exception("Stage 1 rewrite sample gagal")
            return ""
