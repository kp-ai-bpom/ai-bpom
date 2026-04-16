from typing import Any

from .types import RetrievedTable


class ContextBuilder:
    def __init__(self, column_similarity_threshold: float, max_context_chars: int):
        self._column_similarity_threshold = column_similarity_threshold
        self._max_context_chars = max_context_chars

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
            return ""

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

        parts: list[str] = []
        total_chars = 0

        for table in selected_tables:
            table_key = f"{table['schema']}.{table['name']}"
            predicted = predicted_tables[table_key]
            score = predicted.score
            per_column_score = predicted.column_scores

            if score >= 0.6:
                max_data_columns = 6
            elif score >= 0.4:
                max_data_columns = 4
            else:
                max_data_columns = 3

            table_name = table["name"]
            if table_name != table_name.lower():
                sql_ref = f"{table['schema']}.\"{table_name}\""
            else:
                sql_ref = table_key

            header = f"\n## Table: {sql_ref} (score: {score:.3f})"
            parts.append(header)
            total_chars += len(header)

            table_description = table_descriptions.get(table_key)
            if table_description:
                line = f"Description: {table_description}"
                parts.append(line)
                total_chars += len(line)

            parts.append("Columns:")
            total_chars += len("Columns:")

            retrieved_columns: list[tuple[float, str]] = []
            key_columns: list[tuple[float, str]] = []

            for column in table["columns"]:
                column_name = str(column["name"])
                column_type = str(column["type"])
                col_similarity = float(per_column_score.get(column_name, 0.0))

                sample_key = (str(table["schema"]), str(table["name"]), column_name)
                sample_values = samples.get(sample_key, [])
                sample_str = ""
                if sample_values:
                    sample_preview = ", ".join(str(value) for value in sample_values[:3])
                    sample_str = f" -- samples: {sample_preview}"

                score_tag = f" [sim:{col_similarity:.2f}]" if col_similarity > 0 else ""
                line = f"  - {column_name} ({column_type}){score_tag}{sample_str}"

                lowered = column_name.lower()
                is_key_column = (
                    lowered == "id"
                    or lowered.startswith("id")
                    or lowered.endswith("_id")
                    or lowered.endswith("_kode")
                    or lowered == "nip"
                    or lowered.endswith("_nip")
                )

                if col_similarity >= self._column_similarity_threshold:
                    retrieved_columns.append((col_similarity, line))
                elif is_key_column:
                    key_columns.append((col_similarity, line))

            retrieved_columns.sort(key=lambda item: item[0], reverse=True)
            key_columns.sort(key=lambda item: item[0], reverse=True)

            selected_data_columns = [line for _, line in retrieved_columns[:max_data_columns]]
            selected_data_set = set(selected_data_columns)
            selected_key_columns = [
                line for _, line in key_columns if line not in selected_data_set
            ][:5]
            selected_lines = selected_data_columns + selected_key_columns

            skipped = max(0, len(table["columns"]) - len(selected_lines))

            for line in selected_lines:
                if total_chars + len(line) > self._max_context_chars:
                    parts.append("  ... (context truncated)")
                    return "\n".join(parts)
                parts.append(line)
                total_chars += len(line)

            if skipped > 0:
                parts.append(f"  ... (+{skipped} columns omitted)")

            if total_chars > self._max_context_chars:
                break

        return "\n".join(parts)
