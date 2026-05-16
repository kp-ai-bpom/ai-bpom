"""Execution validator: jalankan dry-run SQL dengan beban minimal."""

from __future__ import annotations

import re

from ..repositories import ChatbotRepository
from .config import SQLValidationConfig
from .types import ExecutionVerdict


_TRAILING_SEMICOLON_PATTERN = re.compile(r";+\s*$")


class ExecutionValidator:
    """Membungkus SQL kandidat dengan ``LIMIT 1`` sebelum dieksekusi.

    Tujuannya menangkap kesalahan sintaksis dan referensi (tabel/kolom yang
    tidak ada) dengan beban basis data minimal. Eksekusi dilakukan via
    ``ChatbotRepository.execute_sql`` yang sudah memasang ``statement_timeout``.
    """

    def __init__(
        self,
        repository: ChatbotRepository,
        config: SQLValidationConfig,
    ):
        self._repository = repository
        self._config = config

    @staticmethod
    def _strip_trailing_semicolon(sql: str) -> str:
        return _TRAILING_SEMICOLON_PATTERN.sub("", sql.strip()).strip()

    def _wrap_with_limit_one(self, sql: str) -> str:
        """Bungkus SQL agar paling banyak satu baris yang diambil.

        Menggunakan subquery ``SELECT 1 FROM (<sql>) AS _validation_dry LIMIT 1``
        agar aman terhadap query yang sudah memuat ``LIMIT`` atau ``ORDER BY``
        sendiri. Subquery juga aman untuk ``WITH ... SELECT`` di PostgreSQL 12+.
        """
        cleaned = self._strip_trailing_semicolon(sql)
        return f"SELECT 1 FROM ({cleaned}) AS _validation_dry LIMIT 1"

    async def validate(self, sql: str) -> ExecutionVerdict:
        if not sql or not sql.strip():
            return ExecutionVerdict(
                ok=False,
                error_message="SQL kosong; tidak dapat dieksekusi.",
            )

        wrapped_sql = self._wrap_with_limit_one(sql)
        rows, error_message = await self._repository.execute_sql(
            sql=wrapped_sql,
            timeout_ms=self._config.execution_timeout_ms,
        )

        if error_message is not None:
            normalized_error = error_message.strip()
            if len(normalized_error) > 500:
                normalized_error = normalized_error[:500].rstrip() + "..."
            return ExecutionVerdict(ok=False, error_message=normalized_error)

        rows_returned = len(rows) if rows else 0
        return ExecutionVerdict(ok=True, error_message=None, rows_returned=rows_returned)
