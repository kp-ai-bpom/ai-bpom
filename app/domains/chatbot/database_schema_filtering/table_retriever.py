import re

from app.core.llm import LLMAdapter

from ..repositories import ChatbotRepository
from .config import SemanticMemoryConfig
from .types import RetrievedTable

# Words that often appear as tokens inside table names but are too
# generic to use as evidence for relevance (would over-boost). Lowered.
_NAME_BOOST_STOPWORDS: frozenset[str] = frozenset(
    {"tm", "top", "th", "tr", "tx", "tb", "tabel", "data", "info", "view", "id"}
)


class TableRetriever:
    def __init__(
        self,
        llm_adapter: LLMAdapter,
        repository: ChatbotRepository,
        config: SemanticMemoryConfig,
    ):
        self._llm_adapter = llm_adapter
        self._repository = repository
        self._config = config

    async def retrieve(
        self,
        keywords: list[str],
        raw_query: str | None = None,
    ) -> dict[str, RetrievedTable]:
        """Retrieve relevant tables via vector similarity.

        Two embedding passes are performed:
        1. One embedding per extracted keyword (precise, narrow signal).
        2. One additional embedding of the full ``raw_query`` if provided
           (safety net — catches semantically relevant rows even when the
           keyword extractor misses an important term such as a column name
           that "looks like" a value, e.g. "pool" in "pool 1"). The cost is
           a single extra embedding call per request (~50ms) and dramatically
           improves recall on out-of-vocabulary or ambiguous terms.
        """
        embedding_targets: list[str] = list(keywords)
        if raw_query:
            stripped_query = raw_query.strip()
            if stripped_query and stripped_query.lower() not in {
                kw.strip().lower() for kw in keywords
            }:
                embedding_targets.append(stripped_query)

        if not embedding_targets:
            return {}

        try:
            table_available = await self._repository.is_vector_table_available(
                self._config.vector_table
            )
        except Exception:
            return {}

        if not table_available:
            return {}

        entity_map: dict[int, dict] = {}

        for target in embedding_targets:
            try:
                vector = self._llm_adapter.embeddings.embed_query(target)
            except Exception:
                continue

            rows = await self._repository.retrieve_entities_by_vector(
                vector_table=self._config.vector_table,
                vector=vector,
                top_k=self._config.top_n_per_keyword,
            )
            for row in rows:
                row_id = int(row.get("id") or 0)
                similarity = float(row.get("similarity") or 0.0)
                previous = entity_map.get(row_id)
                if previous is None or similarity > float(previous.get("similarity") or 0.0):
                    entity_map[row_id] = row

        if not entity_map:
            return {}

        table_scores: dict[str, list[float]] = {}
        column_scores: dict[str, dict[str, float]] = {}

        for entity in entity_map.values():
            schema_name = str(entity.get("schema_name") or "")
            table_name = str(entity.get("table_name") or "")
            if not schema_name or not table_name:
                continue

            table_key = f"{schema_name}.{table_name}"
            similarity = float(entity.get("similarity") or 0.0)
            entity_type = str(entity.get("entity_type") or "")
            weight = (
                self._config.table_weight
                if entity_type == "table"
                else self._config.column_weight
            )

            weighted_score = similarity * weight
            table_scores.setdefault(table_key, []).append(weighted_score)

            column_name = entity.get("column_name")
            if column_name:
                col_scores = column_scores.setdefault(table_key, {})
                col_name = str(column_name)
                if similarity > col_scores.get(col_name, 0.0):
                    col_scores[col_name] = similarity

        # Name-match boost: a table whose name contains an extracted
        # keyword as a whole token (split by `_`) is almost always
        # relevant for that keyword — even when its description
        # embedding ranks lower than other tables that happen to mention
        # the keyword incidentally. Inject these tables with a
        # guaranteed score so they survive the Top-N cap and reach
        # Stage 4. Concrete bug from a real trace: query "tampilkan
        # pegawai berpendidikan S2…" — `pegawai_tm` (literally the
        # table named in the query) was outranked by `riwayatjabatan_th`
        # and `sk_pegawai_v`, only entering Stage 4 via FK closure with
        # score=0.0. PK/FK + display cols are added later by the
        # ``auto_include_keys`` block, so name-match injection alone is
        # enough to make the table fully usable downstream.
        allowed = self._config.allowed_tables or {}
        keyword_tokens = {
            kw.lower().strip()
            for kw in keywords
            if kw and len(kw.strip()) >= 3 and kw.lower().strip() not in _NAME_BOOST_STOPWORDS
        }
        if allowed and keyword_tokens:
            # Boost score = JUST above retrieval_threshold (small
            # epsilon). Deliberately LOW so the trace clearly shows
            # the table entered via name-match, NOT via genuine
            # embedding similarity — earlier iteration used
            # threshold+0.4 (=0.8) which masqueraded as a high-
            # confidence retrieval score and was misleading. Survival
            # in Top-N is handled by ``max_retrieved_tables`` (set to
            # N=30 in current config), not by the boost magnitude.
            boost_floor = self._config.retrieval_threshold + 0.05
            for schema_name, table_names in allowed.items():
                for table_name in table_names:
                    table_tokens = {
                        tok
                        for tok in re.split(r"[_\.]+", table_name.lower())
                        if tok and tok not in _NAME_BOOST_STOPWORDS
                    }
                    if not (keyword_tokens & table_tokens):
                        continue
                    table_key = f"{schema_name}.{table_name}"
                    existing = table_scores.get(table_key) or []
                    current_max = max(existing) if existing else 0.0
                    if current_max < boost_floor:
                        table_scores.setdefault(table_key, []).append(boost_floor)

        ranked: list[tuple[str, RetrievedTable]] = []
        for table_key, scores in table_scores.items():
            max_score = max(scores)
            if max_score < self._config.retrieval_threshold:
                continue

            schema_name, table_name = table_key.split(".", 1)
            ranked.append(
                (
                    table_key,
                    RetrievedTable(
                        schema=schema_name,
                        table=table_name,
                        score=max_score,
                        column_scores=column_scores.get(table_key, {}),
                    ),
                )
            )

        ranked.sort(key=lambda item: item[1].score, reverse=True)
        limited = ranked[: self._config.max_retrieved_tables]
        result = {table_key: table for table_key, table in limited}

        # Auto-include PK/FK columns of retrieved tables so downstream
        # Stage 4 (SQL generation) never misses a join/identity key even
        # when vector search did not surface it. Injected with sim=1.0
        # so they pass any tau_column filter applied later. Only enabled
        # when the config flag is True (opt-in, calibration-driven).
        if getattr(self._config, "auto_include_keys", False) and result:
            try:
                key_map = await self._repository.get_table_key_columns(
                    self._config.allowed_tables
                )
            except Exception:
                key_map = {}
            from dataclasses import replace as _dc_replace
            for table_key, retrieved_table in list(result.items()):
                keys = key_map.get(table_key, set())
                if not keys:
                    continue
                col_scores = dict(retrieved_table.column_scores)
                changed = False
                for col_name in keys:
                    if col_scores.get(col_name, 0.0) < 1.0:
                        col_scores[col_name] = 1.0
                        changed = True
                if changed:
                    # RetrievedTable is a frozen dataclass; mutation via
                    # attribute assignment would raise FrozenInstanceError.
                    # Rebuild the entry instead.
                    result[table_key] = _dc_replace(
                        retrieved_table, column_scores=col_scores
                    )

            # Auto-include kolom display ("nama", "nip", "deskripsi",
            # "satker_nama", ...) untuk tabel master (_tm / _top / V_*).
            # Kolom-kolom ini hampir selalu muncul di SELECT GT tapi
            # embedding-nya generik sehingga cosine similarity-nya
            # konsisten di bawah threshold τ_column — gate paper 0.95
            # mustahil tercapai tanpa injeksi eksplisit. Marker score
            # 0.99 dipakai (lebih rendah dari 1.0 PK/FK) supaya bisa
            # dibedakan di log debug Stage 4 jika diperlukan.
            display_cols_map = getattr(self._config, "display_cols", None) or {}
            if display_cols_map:
                for table_key, retrieved_table in list(result.items()):
                    schema_name, table_name = table_key.split(".", 1)
                    # Hanya tabel master yang di-perluas: suffix _tm / _top
                    # (case-insensitif) atau prefix V_ untuk view master.
                    table_lower = table_name.lower()
                    is_master = (
                        table_lower.endswith("_tm")
                        or table_lower.endswith("_top")
                        or table_name.startswith("V_")
                    )
                    if not is_master:
                        continue
                    display_list = display_cols_map.get(table_key)
                    if not display_list:
                        continue
                    col_scores = dict(retrieved_table.column_scores)
                    changed = False
                    for col_name in display_list:
                        if col_scores.get(col_name, 0.0) < 0.99:
                            col_scores[col_name] = 0.99
                            changed = True
                    if changed:
                        result[table_key] = _dc_replace(
                            retrieved_table, column_scores=col_scores
                        )

        return result
