from typing import Any

from ..toon import TOON_NA, encode_table_with_max_chars

from .types import RetrievedTable


_SCHEMA_TABLE_FIELDS = (
    "schema",
    "table",
    "sql_ref",
    "score",
    "description",
)

_SCHEMA_COLUMN_FIELDS = (
    "schema",
    "table",
    "column",
    "type",
    "similarity",
    "samples",
)


class ContextBuilder:
    def __init__(self, column_similarity_threshold: float, max_context_chars: int):
        self._column_similarity_threshold = column_similarity_threshold
        self._max_context_chars = max_context_chars

    @staticmethod
    def _is_key_column(column_name: str) -> bool:
        lowered = column_name.lower()
        return (
            lowered == "id"
            or lowered.startswith("id")
            or lowered.endswith("_id")
            or lowered.endswith("_kode")
            or lowered == "nip"
            or lowered.endswith("_nip")
        )

    @staticmethod
    def _resolve_sql_ref(schema_name: str, table_name: str) -> str:
        if table_name != table_name.lower():
            return f'{schema_name}."{table_name}"'
        return f"{schema_name}.{table_name}"

    def build(
        self,
        predicted_tables: dict[str, RetrievedTable],
        schema_tables: list[dict[str, Any]],
        samples: dict[tuple[str, str, str], list[Any]] | None = None,
        table_descriptions: dict[str, str] | None = None,
    ) -> str:
        samples = samples or {}
        table_descriptions = table_descriptions or {}

        if not predicted_tables:
            return TOON_NA

        relevant_keys = set(predicted_tables.keys())
        selected_tables = [
            table
            for table in schema_tables
            if f"{table['schema']}.{table['name']}" in relevant_keys
        ]
        selected_tables.sort(
            key=lambda table: predicted_tables[f"{table['schema']}.{table['name']}"].score,
            reverse=True,
        )

        table_rows: list[dict[str, Any]] = []
        column_rows: list[dict[str, Any]] = []

        for table in selected_tables:
            table_schema = str(table["schema"])
            table_name = str(table["name"])
            table_key = f"{table_schema}.{table_name}"
            predicted = predicted_tables[table_key]
            score = predicted.score
            per_column_score = predicted.column_scores

            if score >= 0.6:
                max_data_columns = 6
            elif score >= 0.4:
                max_data_columns = 4
            else:
                max_data_columns = 3

            table_description = str(table_descriptions.get(table_key) or "")
            table_rows.append(
                {
                    "schema": table_schema,
                    "table": table_name,
                    "sql_ref": self._resolve_sql_ref(table_schema, table_name),
                    "score": f"{score:.3f}",
                    "description": table_description[:220],
                }
            )

            retrieved_columns: list[tuple[float, dict[str, Any]]] = []
            key_columns: list[tuple[float, dict[str, Any]]] = []

            for column in table["columns"]:
                column_name = str(column["name"])
                column_type = str(column["type"])
                col_similarity = float(per_column_score.get(column_name, 0.0))

                sample_key = (table_schema, table_name, column_name)
                sample_values = samples.get(sample_key, [])

                row = {
                    "schema": table_schema,
                    "table": table_name,
                    "column": column_name,
                    "type": column_type,
                    "similarity": f"{col_similarity:.2f}" if col_similarity > 0 else "",
                    "samples": [str(value) for value in sample_values[:3]],
                }

                if col_similarity >= self._column_similarity_threshold:
                    retrieved_columns.append((col_similarity, row))
                elif self._is_key_column(column_name):
                    key_columns.append((col_similarity, row))

            retrieved_columns.sort(key=lambda item: item[0], reverse=True)
            key_columns.sort(key=lambda item: item[0], reverse=True)

            selected_data_columns = [
                row for _, row in retrieved_columns[:max_data_columns]
            ]
            selected_column_names = {
                str(row["column"]) for row in selected_data_columns
            }

            selected_key_columns: list[dict[str, Any]] = []
            for _, row in key_columns:
                column_name = str(row["column"])
                if column_name in selected_column_names:
                    continue
                selected_column_names.add(column_name)
                selected_key_columns.append(row)
                if len(selected_key_columns) >= 5:
                    break

            column_rows.extend(selected_data_columns + selected_key_columns)

        table_block = encode_table_with_max_chars(
            name="schema_context_tables",
            rows=table_rows,
            fields=_SCHEMA_TABLE_FIELDS,
            max_chars=self._max_context_chars,
        )
        if table_block == TOON_NA:
            return TOON_NA

        remaining_chars = self._max_context_chars - len(table_block)
        if remaining_chars <= 2:
            return table_block

        column_block = encode_table_with_max_chars(
            name="schema_context_columns",
            rows=column_rows,
            fields=_SCHEMA_COLUMN_FIELDS,
            max_chars=remaining_chars - 2,
        )
        if column_block == TOON_NA:
            return table_block

        return f"{table_block}\n\n{column_block}"
