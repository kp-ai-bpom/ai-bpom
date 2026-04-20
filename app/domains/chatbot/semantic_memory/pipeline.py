import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.llm import LLMAdapter

from ..repositories import ChatbotRepository
from ..sql_generator import SQLGenerator, get_sql_generator_config
from .config import SemanticMemoryConfig
from .context_builder import ContextBuilder
from .keyword_extractor import KeywordExtractor
from .table_retriever import TableRetriever
from .types import PipelineResult, RetrievedTable


_COLUMN_ALIAS_STOPWORDS = {
    "atau",
    "dan",
    "yang",
    "dengan",
    "untuk",
    "pada",
    "di",
    "ke",
    "dari",
}

_BASE_KNOWLEDGE_CSV_PATH = Path(__file__).resolve().parents[4] / "data" / "base_knowledge.csv"


def _normalize_alias_key(identifier: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", identifier.strip().lower())


def _register_column_alias(
    aliases_by_table: dict[str, dict[str, set[str]]],
    table_key: str,
    alias: str,
    column_name: str,
) -> None:
    normalized_alias = _normalize_alias_key(alias)
    if not normalized_alias:
        return

    aliases_by_table.setdefault(table_key, {}).setdefault(normalized_alias, set()).add(
        column_name
    )


def _register_domain_aliases(
    aliases_by_table: dict[str, dict[str, set[str]]],
) -> None:
    # Known education aliases frequently used in user prompts.
    education_table = "siap.v_pendidikan_terakhir"
    for alias in {
        "nama_pt",
        "namapt",
        "perguruan_tinggi",
        "kampus",
        "universitas",
        "universitas_atau_sekolah",
        "almamater",
    }:
        _register_column_alias(aliases_by_table, education_table, alias, "namasekolah")

    for alias in {
        "nama_prodi",
        "namaprodi",
        "prodi",
        "jurusan",
        "program_studi",
    }:
        _register_column_alias(aliases_by_table, education_table, alias, "programstudi")

    # Common hallucinated person-name fields for pegawai data.
    pegawai_table = "public.pegawai_tm"
    for alias in {
        "first_name",
        "firstname",
        "last_name",
        "lastname",
        "full_name",
        "nama_lengkap",
    }:
        _register_column_alias(aliases_by_table, pegawai_table, alias, "nama")


@lru_cache(maxsize=1)
def _load_base_knowledge_column_aliases() -> dict[str, dict[str, set[str]]]:
    aliases_by_table: dict[str, dict[str, set[str]]] = {}

    if not _BASE_KNOWLEDGE_CSV_PATH.exists():
        _register_domain_aliases(aliases_by_table)
        return aliases_by_table

    try:
        with _BASE_KNOWLEDGE_CSV_PATH.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                entity_type = str(row.get("entity_type") or "").strip().lower()
                if entity_type != "column":
                    continue

                schema_name = str(row.get("schema_name") or "").strip()
                table_name = str(row.get("table_name") or "").strip()
                column_name = str(row.get("column_name") or "").strip()
                if not schema_name or not table_name or not column_name:
                    continue

                table_key = f"{schema_name.lower()}.{table_name.lower()}"

                # Always register the physical column name itself.
                _register_column_alias(
                    aliases_by_table,
                    table_key,
                    alias=column_name,
                    column_name=column_name,
                )

                alias_field = str(row.get("column_alias") or "")
                if alias_field:
                    for alias in re.split(r"[,;/|]", alias_field):
                        cleaned_alias = alias.strip()
                        if not cleaned_alias:
                            continue
                        _register_column_alias(
                            aliases_by_table,
                            table_key,
                            alias=cleaned_alias,
                            column_name=column_name,
                        )
    except Exception:
        # Keep pipeline resilient if CSV cannot be parsed.
        aliases_by_table = {}

    _register_domain_aliases(aliases_by_table)
    return aliases_by_table


_COLUMN_ALIAS_CANONICAL_MAP = {
    # Frequently produced aliases by LLM for education attributes.
    "namaprodi": "programstudi",
    "prodi": "programstudi",
    "programstudi": "programstudi",
    "namapt": "namasekolah",
    "perguruantinggi": "namasekolah",
    "universitasatausekolah": "namasekolah",
    "namasekolah": "namasekolah",
    # Common hallucinated person-name columns for pegawai table.
    "firstname": "nama",
    "lastname": "nama",
}

_COLUMN_TOKEN_EXPANSIONS = {
    "prodi": {"programstudi"},
    "namaprodi": {"programstudi"},
    "program": {"programstudi"},
    "studi": {"programstudi"},
    "universitas": {"namasekolah"},
    "kampus": {"namasekolah"},
    "sekolah": {"namasekolah"},
    "perguruan": {"namasekolah"},
    "tinggi": {"namasekolah"},
    "almamater": {"namasekolah"},
    "jurusan": {"programstudi"},
}


class SemanticMemoryPipeline:
    def __init__(
        self,
        llm_adapter: LLMAdapter,
        repository: ChatbotRepository,
        config: SemanticMemoryConfig,
    ):
        self._llm_adapter = llm_adapter
        self._repository = repository
        self._config = config
        self._keyword_extractor = KeywordExtractor(
            llm_adapter=llm_adapter,
            allowed_tables=config.allowed_tables,
            retries=config.keyword_retries,
        )
        self._table_retriever = TableRetriever(
            llm_adapter=llm_adapter,
            repository=repository,
            config=config,
        )
        self._context_builder = ContextBuilder(
            column_similarity_threshold=config.column_similarity_threshold,
            max_context_chars=config.max_context_chars,
        )
        sql_generator_config = get_sql_generator_config()
        self._sql_generator = SQLGenerator(
            llm_adapter=llm_adapter,
            allowed_tables=config.allowed_tables,
            config=sql_generator_config,
        )

    async def run(self, query: str) -> PipelineResult:
        schema_tables = await self._repository.load_schema(self._config.allowed_tables)
        if not schema_tables:
            raise RuntimeError("No schema loaded from database")

        keywords = await self._keyword_extractor.extract(query)
        predicted_tables = await self._table_retriever.retrieve(keywords)
        if not predicted_tables:
            predicted_tables = self._fallback_tables(
                keywords=keywords,
                schema_tables=schema_tables,
            )

        relevant_schema_tables = [
            table
            for table in schema_tables
            if f"{table['schema']}.{table['name']}" in predicted_tables
        ]

        table_descriptions: dict[str, str] = {}
        try:
            if await self._repository.is_vector_table_available(self._config.vector_table):
                table_descriptions = await self._repository.load_table_descriptions(
                    self._config.vector_table
                )
        except Exception:
            table_descriptions = {}

        samples = await self._repository.load_column_samples(
            schema_tables=relevant_schema_tables,
            n_samples=self._config.sample_rows_per_table,
        )

        context = self._context_builder.build(
            predicted_tables=predicted_tables,
            schema_tables=schema_tables,
            samples=samples,
            table_descriptions=table_descriptions,
        )

        sql, explanation = await self._sql_generator.generate(query, context)
        if not sql:
            raise RuntimeError(explanation or "Failed to generate SQL")

        corrected_sql = self._autocorrect_sql_identifiers(sql, schema_tables)
        if corrected_sql != sql:
            sql = corrected_sql

        validation_error = self._sql_generator.validate_sql_candidate(sql)
        if validation_error:
            raise RuntimeError(f"Generated SQL is invalid: {validation_error}")

        rows, execution_error = await self._repository.execute_sql(
            sql=sql,
            timeout_ms=self._config.sql_timeout_ms,
        )
        executed = execution_error is None
        if execution_error:
            explanation = f"{explanation} | SQL execution failed: {execution_error[:180]}"

        return PipelineResult(
            keywords=keywords,
            predicted_tables=predicted_tables,
            context=context,
            sql=sql,
            explanation=explanation,
            executed=executed,
            execution_error=execution_error,
            rows=rows,
        )

    def _fallback_tables(
        self,
        keywords: list[str],
        schema_tables: list[dict[str, Any]],
    ) -> dict[str, RetrievedTable]:
        scored: list[tuple[str, RetrievedTable]] = []

        lowered_keywords = [keyword.lower() for keyword in keywords if keyword.strip()]
        for table in schema_tables:
            table_schema = str(table["schema"])
            table_name = str(table["name"])
            table_key = f"{table_schema}.{table_name}"
            table_name_lower = table_name.lower()

            table_score = 0.0
            matched_columns: dict[str, float] = {}

            for keyword in lowered_keywords:
                if keyword in table_name_lower:
                    table_score += 1.0

                for column in table["columns"]:
                    column_name = str(column["name"])
                    if keyword in column_name.lower():
                        table_score += 0.8
                        matched_columns[column_name] = max(
                            matched_columns.get(column_name, 0.0),
                            0.8,
                        )

            if table_score > 0:
                scored.append(
                    (
                        table_key,
                        RetrievedTable(
                            schema=table_schema,
                            table=table_name,
                            score=table_score,
                            column_scores=matched_columns,
                        ),
                    )
                )

        if not scored:
            for table in schema_tables[: self._config.max_retrieved_tables]:
                table_schema = str(table["schema"])
                table_name = str(table["name"])
                table_key = f"{table_schema}.{table_name}"
                scored.append(
                    (
                        table_key,
                        RetrievedTable(
                            schema=table_schema,
                            table=table_name,
                            score=0.1,
                            column_scores={},
                        ),
                    )
                )

        scored.sort(key=lambda item: item[1].score, reverse=True)
        limited = scored[: self._config.max_retrieved_tables]
        return {table_key: table for table_key, table in limited}

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    @staticmethod
    def _normalize_identifier(identifier: str) -> str:
        return re.sub(r"_+", "", identifier.strip().lower())

    def _guess_column_by_tokens(
        self,
        raw_column_name: str,
        candidate_columns: list[str],
    ) -> str | None:
        # Heuristic fallback for semantic aliases generated by the model,
        # e.g. universitas_atau_sekolah -> namasekolah.
        normalized_raw = self._normalize_identifier(raw_column_name)

        mapped_target = _COLUMN_ALIAS_CANONICAL_MAP.get(normalized_raw)
        if mapped_target:
            mapped_candidates = [
                candidate
                for candidate in candidate_columns
                if self._normalize_identifier(candidate) == mapped_target
            ]
            if len(mapped_candidates) == 1:
                return mapped_candidates[0]

        raw_tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", raw_column_name.lower())
            if token and token not in _COLUMN_ALIAS_STOPWORDS and len(token) >= 4
        ]
        if not raw_tokens:
            return None

        expanded_tokens = set(raw_tokens)
        for token in list(raw_tokens):
            for probe, expansions in _COLUMN_TOKEN_EXPANSIONS.items():
                if probe == token or probe in token:
                    expanded_tokens.update(expansions)

        scores: dict[str, int] = {}
        for candidate_column in candidate_columns:
            candidate_key = self._normalize_identifier(candidate_column)
            if not candidate_key:
                continue

            score = 0
            for token in expanded_tokens:
                if token in candidate_key:
                    score += len(token)

            if score > 0:
                scores[candidate_column] = score

        if not scores:
            return None

        best_score = max(scores.values())
        best_candidates = [
            candidate for candidate, score in scores.items() if score == best_score
        ]
        if len(best_candidates) != 1:
            return None

        # Conservative thresholds to avoid accidental over-corrections.
        if best_score >= 8:
            return best_candidates[0]
        if len(scores) == 1 and best_score >= 6:
            return best_candidates[0]
        return None

    def _build_table_metadata(
        self,
        schema_tables: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        table_metadata: dict[str, dict[str, Any]] = {}
        base_knowledge_aliases = _load_base_knowledge_column_aliases()

        for table in schema_tables:
            schema_name = str(table.get("schema") or "").strip()
            table_name = str(table.get("name") or "").strip()
            if not schema_name or not table_name:
                continue

            columns_raw = [
                str(column.get("name") or "").strip()
                for column in table.get("columns", [])
                if str(column.get("name") or "").strip()
            ]
            columns_exact = set(columns_raw)
            columns_lower_map: dict[str, str] = {
                column_name.lower(): column_name for column_name in columns_raw
            }

            columns_normalized: dict[str, set[str]] = {}
            for column_name in columns_raw:
                normalized_key = self._normalize_identifier(column_name)
                if not normalized_key:
                    continue
                columns_normalized.setdefault(normalized_key, set()).add(column_name)

            table_key = f"{schema_name.lower()}.{table_name.lower()}"
            table_aliases_raw = base_knowledge_aliases.get(table_key, {})
            table_aliases: dict[str, set[str]] = {}
            for alias_key, candidate_columns in table_aliases_raw.items():
                valid_columns = {
                    candidate for candidate in candidate_columns if candidate in columns_exact
                }
                if valid_columns:
                    table_aliases[alias_key] = valid_columns

            table_name_canonical = (
                table_name
                if re.fullmatch(r"[a-z_][a-z0-9_]*", table_name)
                else self._quote_identifier(table_name)
            )
            canonical_ref = f"{schema_name}.{table_name_canonical}"

            table_metadata[table_key] = {
                "schema": schema_name,
                "table": table_name,
                "canonical_ref": canonical_ref,
                "columns_raw": columns_raw,
                "columns_exact": columns_exact,
                "columns_lower_map": columns_lower_map,
                "columns_normalized": columns_normalized,
                "column_aliases": table_aliases,
            }

        return table_metadata

    def _autocorrect_sql_identifiers(
        self,
        sql: str,
        schema_tables: list[dict[str, Any]],
    ) -> str:
        table_metadata = self._build_table_metadata(schema_tables)
        if not table_metadata:
            return sql

        corrected_sql = sql

        # Normalize table references so case-sensitive table names are properly quoted.
        for table_key in sorted(table_metadata.keys(), key=len, reverse=True):
            metadata = table_metadata[table_key]
            pattern = re.compile(
                rf"\b{re.escape(table_key)}\b",
                re.IGNORECASE,
            )
            corrected_sql = pattern.sub(str(metadata["canonical_ref"]), corrected_sql)

        alias_map: dict[str, dict[str, Any]] = {}
        table_alias_pattern = re.compile(
            r"\b(?:from|join)\s+"
            r"([A-Za-z_][A-Za-z0-9_]*\.(?:\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*))"
            r"(?:\s+(?:as\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
            re.IGNORECASE,
        )

        for match in table_alias_pattern.finditer(corrected_sql):
            table_ref = str(match.group(1) or "").strip()
            alias = str(match.group(2) or "").strip()
            if "." not in table_ref:
                continue

            schema_name, table_token = table_ref.split(".", 1)
            table_name = table_token.strip().strip('"')
            metadata = table_metadata.get(f"{schema_name.lower()}.{table_name.lower()}")
            if metadata is None:
                continue

            resolved_alias = alias or table_name
            alias_map[resolved_alias] = metadata
            alias_map[resolved_alias.lower()] = metadata

        if not alias_map:
            return corrected_sql

        column_ref_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")

        def _replace_column_ref(match: re.Match[str]) -> str:
            alias = str(match.group(1) or "")
            column_name = str(match.group(2) or "")

            metadata = alias_map.get(alias) or alias_map.get(alias.lower())
            if metadata is None:
                return match.group(0)

            columns_exact: set[str] = metadata["columns_exact"]
            if column_name in columns_exact:
                return match.group(0)

            columns_lower_map: dict[str, str] = metadata["columns_lower_map"]
            case_match = columns_lower_map.get(column_name.lower())
            if case_match and case_match != column_name:
                return f"{alias}.{case_match}"

            normalized_key = self._normalize_identifier(column_name)
            columns_normalized: dict[str, set[str]] = metadata["columns_normalized"]
            candidates = columns_normalized.get(normalized_key, set())
            if len(candidates) == 1:
                resolved_column = next(iter(candidates))
                if resolved_column != column_name:
                    return f"{alias}.{resolved_column}"

            column_aliases: dict[str, set[str]] = metadata.get("column_aliases", {})
            alias_candidates = column_aliases.get(normalized_key, set())
            if len(alias_candidates) == 1:
                resolved_column = next(iter(alias_candidates))
                if resolved_column != column_name:
                    return f"{alias}.{resolved_column}"

            token_match = self._guess_column_by_tokens(
                raw_column_name=column_name,
                candidate_columns=list(metadata.get("columns_raw", [])),
            )
            if token_match and token_match != column_name:
                return f"{alias}.{token_match}"

            return match.group(0)

        corrected_sql = column_ref_pattern.sub(_replace_column_ref, corrected_sql)
        return corrected_sql
