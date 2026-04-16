import re
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from app.core.llm import LLMAdapter
from app.core.logger import log

from ..repositories import ChatbotRepository
from .config import QuestionRewritingConfig
from .parsers import parse_rewritten, strip_thinking
from .prompts import REWRITE_SYSTEM_PROMPT, build_user_prompt
from .types import EpisodicMatch, RewriteResult


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


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

    def add_to_working_memory(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        key = self._session_key(user_id, session_id)
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
                }
            )

    def get_working_context(
        self,
        user_id: str,
        session_id: str,
        window: int | None = None,
    ) -> str:
        resolved_window = window or self._config.working_memory_window
        key = self._session_key(user_id, session_id)

        with self._working_lock:
            session_memory = list(self._working_memory_by_session.get(key, []))

        recent = session_memory[-resolved_window:]
        if not recent:
            return ""

        lines = [f"{item['role'].upper()}: {item['content']}" for item in recent]
        result = "\n".join(lines)
        if len(result) > self._config.max_working_snippet_chars:
            result = result[-self._config.max_working_snippet_chars :]
            first_break = result.find("\n")
            if first_break > 0:
                result = result[first_break + 1 :]

        return result

    def remove_session_memory(self, user_id: str, session_id: str) -> None:
        self.clear_session_memory(user_id=user_id, session_id=session_id)

    def _get_session_memory(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        key = self._session_key(user_id, session_id)
        with self._working_lock:
            return list(self._working_memory_by_session.get(key, []))

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

    def _build_episodic_summary(self, user_id: str, session_id: str) -> tuple[str, list[str], str, str]:
        session_memory = self._get_session_memory(user_id, session_id)

        user_queries = [item["content"] for item in session_memory if item["role"] == "user"]
        assistant_replies = [
            item["content"] for item in session_memory if item["role"] == "assistant"
        ]

        all_topics = " ".join(user_queries[:5])
        tags = self._tokenize(all_topics)[:8] or ["conversation"]

        n_turns = len(user_queries)
        rewritten_count = sum(
            1
            for original, rewritten in zip(user_queries, assistant_replies)
            if original.strip() != rewritten.strip()
        )

        highlighted_topics = [query.strip()[:80] for query in user_queries[:3] if query.strip()]
        topics_text = ", ".join(highlighted_topics) if highlighted_topics else "umum"

        summary = (
            f"Percakapan {n_turns} turn tentang: {topics_text}. "
            f"{rewritten_count} query di-rewrite."
        )
        worked = (
            "Menggabungkan konteks working memory untuk memperjelas referensi "
            "implisit dalam percakapan multi-turn."
        )
        avoid = "Menambahkan detail yang tidak muncul di working atau episodic context."
        return summary, tags, worked, avoid

    def _format_episodic_for_prompt(self, episodes: list[dict[str, Any]]) -> str:
        if not episodes:
            return ""

        blocks: list[str] = []
        total_chars = 0

        for index, episode in enumerate(episodes, start=1):
            tags = episode.get("context_tags") or []
            tags_text = ", ".join([str(tag) for tag in tags])
            message_count = int(episode.get("message_count") or 0)
            last_message_at = episode.get("last_message_at")
            last_message_at_text = str(last_message_at) if last_message_at else "-"
            recent_context = str(episode.get("conversation", ""))[:220]
            block = (
                f"Episode {index} (similarity={float(episode.get('similarity', 0.0)):.2f}):\n"
                f"- conversation_summary: {episode.get('conversation_summary', '')}\n"
                f"- session_size: {message_count} messages\n"
                f"- last_message_at: {last_message_at_text}\n"
                f"- recent_context: {recent_context}\n"
                f"- tags: {tags_text}\n"
                f"- what_worked: {episode.get('what_worked', '')}\n"
                f"- what_to_avoid: {episode.get('what_to_avoid', '')}"
            )
            if total_chars + len(block) > self._config.max_episodic_snippet_chars:
                break

            blocks.append(block)
            total_chars += len(block)

        return "\n\n".join(blocks)

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
        working_context = self.get_working_context(user_id=user_id, session_id=session_id)

        try:
            await self._ensure_table_ready()
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
        if working_context or episodic_matches:
            user_prompt = build_user_prompt(
                current_query=normalized_query,
                working_context=working_context,
                episodic_context=episodic_context,
            )
            try:
                response = await self._llm_adapter.think.bind(
                    max_tokens=self._config.llm_max_tokens
                ).ainvoke(
                    [
                        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                raw_output = strip_thinking(str(getattr(response, "content", "") or ""))
                parsed = parse_rewritten(raw_output)
                if parsed:
                    rewritten_query = parsed
            except Exception:
                log.exception("Question rewriting failed, fallback to original query")

        self.add_to_working_memory(user_id, session_id, "user", normalized_query)
        self.add_to_working_memory(user_id, session_id, "assistant", rewritten_query)

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
        )
