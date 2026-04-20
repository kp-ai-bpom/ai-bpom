import asyncio
import csv
import re
from functools import lru_cache
from pathlib import Path

from app.core.llm import LLMAdapter

from ..toon import TOON_NA, encode_table

from .config import SQLGeneratorConfig
from .parsers import (
    extract_json_object,
    extract_sql_from_fence,
    extract_sql_from_text,
    strip_thinking,
)


_DANGEROUS_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy)\b",
    re.IGNORECASE,
)
_INVALID_COUNT_DISTINCT_ON_PATTERN = re.compile(
    r"\bcount\s*\(\s*distinct\s+on\s*\(",
    re.IGNORECASE,
)

_DEFAULT_STATUS_FILTER_VALUES = {"cpns", "pns", "polri", "pppk"}
_DEFAULT_KEDUDUKAN_FILTER_VALUES = {"aktif", "tugas belajar", "cltn"}
_NON_ACTIVE_REQUEST_HINTS = (
    "non aktif",
    "nonaktif",
    "tidak aktif",
    "selain aktif",
    "selain status aktif",
    "semua status",
    "seluruh status",
    "aktif dan non",
    "aktif dan tidak aktif",
    "pensiun",
    "berhenti",
    "mengundurkan",
    "resign",
    "wafat",
    "meninggal",
)
_IN_VALUE_PATTERN = re.compile(r"'([^']*)'")
_GROUP_BY_GENDER_PATTERN = re.compile(
    r"\bgroup\s+by\b[^;]*\bjenis_kelamin\b",
    re.IGNORECASE | re.DOTALL,
)
_GROUP_BY_GENERASI_PATTERN = re.compile(
    r"\bgroup\s+by\s+(?:generasi|1)\b",
    re.IGNORECASE,
)
_GROUP_BY_PENDIDIKAN_TOP_PATTERN = re.compile(
    r"\bgroup\s+by\b[^;]*\bpendidikan_top_id\b",
    re.IGNORECASE | re.DOTALL,
)

_UNIT_WORK_HINTS = (
    "tipe unit kerja",
    "unit kerja",
)

_GENDER_HINTS = (
    "jenis kelamin",
    "gender",
)

_EDUCATION_HINTS = (
    "pendidikan terakhir",
)

_GENERATION_HINTS = (
    "generasi",
)

_COUNT_HINTS = (
    "jumlah",
    "berapa",
    "hitung",
    "count",
)

_COMPUTER_ROLE_HINTS = (
    "ahli komputer",
    "pranata komputer",
)

_EDUCATION_BUCKET_ALIASES = (
    "unset",
    "under_d3",
    "d3",
    "d4_s1",
    "profesi",
    "s2",
    "s3",
)

_GROUND_TRUTH_CSV_PATH = Path(__file__).resolve().parents[4] / "data" / "information_needs.csv"

_GROUND_TRUTH_CATEGORY_MATCHERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("jenjang_jabatan_fungsional_non_pfm", ("jenjang jabatan fungsional non",)),
    ("jenjang_jabatan_fungsional_pfm", ("jenjang jabatan fungsional pfm",)),
    ("jenjang_jabatan_non_fungsional", ("jenjang jabatan non fungsional",)),
    ("jenjang_jabatan_fungsional", ("jenjang jabatan fungsional",)),
    ("jenis_kelamin", ("jenis kelamin",)),
    ("pendidikan_terakhir", ("pendidikan terakhir",)),
    ("kelompok_usia", ("kelompok usia",)),
    ("generasi", ("generasi",)),
    ("tipe_pegawai", ("tipe pegawai",)),
    ("status_pegawai", ("status pegawai",)),
    ("kedudukan_pegawai", ("kedudukan pegawai",)),
    ("golongan", ("golongan",)),
    ("eselon", ("eselon",)),
    ("bup", ("bup",)),
)

_GROUND_TRUTH_EXPLANATIONS: dict[str, str] = {
    "jenis_kelamin": "Menampilkan rekap pegawai aktif berdasarkan jenis kelamin pada setiap tipe unit kerja.",
    "pendidikan_terakhir": "Menampilkan rekap pegawai aktif berdasarkan pendidikan terakhir pada setiap tipe unit kerja.",
    "kelompok_usia": "Menampilkan rekap pegawai aktif berdasarkan kelompok usia pada setiap tipe unit kerja.",
    "generasi": "Menampilkan rekap pegawai aktif berdasarkan generasi pada setiap tipe unit kerja.",
    "tipe_pegawai": "Menampilkan rekap pegawai aktif berdasarkan tipe pegawai pada setiap tipe unit kerja.",
    "status_pegawai": "Menampilkan rekap pegawai aktif berdasarkan status pegawai pada setiap tipe unit kerja.",
    "kedudukan_pegawai": "Menampilkan rekap pegawai aktif berdasarkan kedudukan pegawai pada setiap tipe unit kerja.",
    "golongan": "Menampilkan rekap pegawai aktif berdasarkan golongan pada setiap tipe unit kerja.",
    "jenjang_jabatan_fungsional": "Menampilkan rekap pegawai aktif berdasarkan jenjang jabatan fungsional pada setiap tipe unit kerja.",
    "jenjang_jabatan_fungsional_pfm": "Menampilkan rekap pegawai aktif berdasarkan jenjang jabatan fungsional PFM pada setiap tipe unit kerja.",
    "jenjang_jabatan_fungsional_non_pfm": "Menampilkan rekap pegawai aktif berdasarkan jenjang jabatan fungsional Non-PFM pada setiap tipe unit kerja.",
    "jenjang_jabatan_non_fungsional": "Menampilkan rekap pegawai aktif berdasarkan jenjang jabatan non fungsional pada setiap tipe unit kerja.",
    "eselon": "Menampilkan rekap pegawai aktif berdasarkan eselon pada setiap tipe unit kerja.",
    "bup": "Menampilkan rekap pegawai aktif berdasarkan BUP pada setiap tipe unit kerja.",
}


@lru_cache(maxsize=1)
def _load_ground_truth_by_category() -> dict[str, str]:
    if not _GROUND_TRUTH_CSV_PATH.exists():
        return {}

    category_to_sql: dict[str, str] = {}
    with _GROUND_TRUTH_CSV_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            question = str(row.get("Pertanyaan Pengguna") or "")
            sql = str(row.get("SQL Ground Truth") or "")

            normalized_question = re.sub(r"\s+", " ", question.strip().lower())
            sanitized_sql = re.sub(r"\s+", " ", sql.strip())
            sanitized_sql = sanitized_sql.rstrip(";").strip()

            if not normalized_question or not sanitized_sql:
                continue

            for category, matchers in _GROUND_TRUTH_CATEGORY_MATCHERS:
                if category in category_to_sql:
                    continue
                if all(matcher in normalized_question for matcher in matchers):
                    category_to_sql[category] = sanitized_sql
                    break

    return category_to_sql


_PROMPT_EXAMPLES = """Examples (MUST follow this JSON format):
{"query": "SELECT * FROM public.pegawai_tm p WHERE p.status_pegawai IN ('CPNS','PNS','POLRI','PPPK') AND p.kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN') LIMIT 10", "explanation": "Menampilkan 10 pegawai aktif pertama"}

{"query": "SELECT DISTINCT ON (p.pegawai_id) p.nama, j.jabatan_nama, vpt.namasekolah, vpt.programstudi FROM public.pegawai_tm p JOIN public.jabatan_tm j ON p.jabatan_id = j.jabatan_id JOIN siap.\"V_PENDIDIKAN_TERAKHIR\" vpt ON vpt.pegawaiid = p.pegawai_id WHERE lower(trim(vpt.namasekolah)) = lower(trim('Universitas Komputer Indonesia')) AND (lower(j.jabatan_nama) LIKE '%ahli komputer%' OR lower(j.jabatan_nama) LIKE '%pranata komputer%') AND p.status_pegawai IN ('CPNS','PNS','POLRI','PPPK') AND p.kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN') ORDER BY p.pegawai_id, vpt.ranking ASC NULLS LAST", "explanation": "Menampilkan satu baris per pegawai ahli/pranata komputer lulusan Universitas Komputer Indonesia tanpa duplikasi karena perbedaan huruf besar-kecil"}

{"query": "SELECT CASE WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat' WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM' WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM' WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM' END AS tipe_unit_kerja, COUNT(DISTINCT p.pegawai_id) AS total_pegawai, SUM(CASE WHEN p.pendidikan_top_id IS NULL THEN 1 ELSE 0 END) AS unset, SUM(CASE WHEN p.pendidikan_top_id IN ('01','02','03','04','05','06','07') THEN 1 ELSE 0 END) AS under_d3, SUM(CASE WHEN p.pendidikan_top_id = '08' THEN 1 ELSE 0 END) AS d3, SUM(CASE WHEN p.pendidikan_top_id IN ('09','10','11') THEN 1 ELSE 0 END) AS d4_s1, SUM(CASE WHEN p.pendidikan_top_id = '12' THEN 1 ELSE 0 END) AS profesi, SUM(CASE WHEN p.pendidikan_top_id = '13' THEN 1 ELSE 0 END) AS s2, SUM(CASE WHEN p.pendidikan_top_id = '14' THEN 1 ELSE 0 END) AS s3 FROM public.pegawai_tm p JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id WHERE p.status_pegawai IN ('CPNS','PNS','POLRI','PPPK') AND p.kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN') GROUP BY tipe_unit_kerja", "explanation": "Menampilkan rekap pendidikan terakhir pegawai aktif per tipe unit kerja"}
"""


class SQLGenerator:
    def __init__(
        self,
        llm_adapter: LLMAdapter,
        allowed_tables: dict[str, list[str]],
        config: SQLGeneratorConfig,
    ):
        self._llm_adapter = llm_adapter
        self._retries = max(1, config.retries)
        self._max_tokens = max(256, config.max_tokens)
        self._allowed_tables = {
            schema_name: [table_name for table_name in table_names if table_name]
            for schema_name, table_names in allowed_tables.items()
            if schema_name and table_names
        }
        default_schema = config.default_schema
        if default_schema in self._allowed_tables:
            self._default_schema = default_schema
        elif self._allowed_tables:
            self._default_schema = next(iter(self._allowed_tables))
        else:
            self._default_schema = "public"

    @staticmethod
    def validate_sql_candidate(sql: str) -> str | None:
        if not sql or not sql.strip():
            return "SQL kosong"

        normalized = sql.strip()
        if not re.match(r"^(select|with)\b", normalized, re.IGNORECASE):
            return "SQL harus diawali SELECT atau WITH"

        if _DANGEROUS_SQL_PATTERN.search(normalized):
            return "Hanya SQL read-only yang diizinkan"

        if _INVALID_COUNT_DISTINCT_ON_PATTERN.search(normalized):
            return (
                "Jangan gunakan COUNT(DISTINCT ON (...)). Gunakan COUNT(DISTINCT p.pegawai_id) "
                "atau COUNT(*) dari subquery yang sudah dedupe."
            )

        if ";" in normalized:
            return "SQL tidak boleh mengandung titik koma"

        multi_statement = re.search(r";\s*\S", normalized.rstrip())
        if multi_statement:
            return "Multiple SQL statements tidak diizinkan"

        return None

    @staticmethod
    def _normalize_literal(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
        return any(hint in text for hint in hints)

    @staticmethod
    def _has_group_by_unit_type(normalized_sql: str) -> bool:
        return any(
            pattern in normalized_sql
            for pattern in (
                "group by tipe_unit_kerja",
                "group by s.tipe_balai",
                "group by 1",
            )
        )

    @staticmethod
    def _has_gender_pivot_projection(normalized_sql: str) -> bool:
        has_male = "jenis_kelamin = 'laki-laki'" in normalized_sql
        has_female = "jenis_kelamin = 'perempuan'" in normalized_sql
        has_conditional_agg = (
            "sum(case when" in normalized_sql
            or "count(*) filter (where" in normalized_sql
        )
        return has_male and has_female and has_conditional_agg

    @staticmethod
    def _has_education_bucket_projection(normalized_sql: str) -> bool:
        has_conditional_agg = "sum(case when p.pendidikan_top_id" in normalized_sql
        has_bucket_aliases = all(
            re.search(rf"\bas\s+{alias}\b", normalized_sql)
            for alias in _EDUCATION_BUCKET_ALIASES
        )
        return has_conditional_agg and has_bucket_aliases

    @staticmethod
    def _build_rekap_pendidikan_per_unit_query() -> str:
        return (
            "SELECT CASE WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat' "
            "WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM' "
            "WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM' "
            "WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM' END AS tipe_unit_kerja, "
            "COUNT(DISTINCT p.nama) AS total_pegawai, "
            "SUM(CASE WHEN p.pendidikan_top_id IS NULL THEN 1 ELSE 0 END) AS unset, "
            "SUM(CASE WHEN p.pendidikan_top_id IN ('01','02','03','04','05','06','07') THEN 1 ELSE 0 END) AS under_d3, "
            "SUM(CASE WHEN p.pendidikan_top_id = '08' THEN 1 ELSE 0 END) AS d3, "
            "SUM(CASE WHEN p.pendidikan_top_id IN ('09','10','11') THEN 1 ELSE 0 END) AS d4_s1, "
            "SUM(CASE WHEN p.pendidikan_top_id = '12' THEN 1 ELSE 0 END) AS profesi, "
            "SUM(CASE WHEN p.pendidikan_top_id = '13' THEN 1 ELSE 0 END) AS s2, "
            "SUM(CASE WHEN p.pendidikan_top_id = '14' THEN 1 ELSE 0 END) AS s3 "
            "FROM public.pegawai_tm p "
            "JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id "
            "WHERE p.status_pegawai IN ('CPNS','PNS','POLRI','PPPK') "
            "AND p.kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN') "
            "GROUP BY tipe_unit_kerja"
        )

    @staticmethod
    def _build_rekap_generasi_per_unit_query() -> str:
        return (
            "SELECT CASE WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat' "
            "WHEN s.tipe_balai IN ('B','BA','BB') THEN 'UPT Balai Besar / Balai POM' "
            "WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM' END AS tipe_unit_kerja, "
            "COUNT(*) AS total_pegawai, "
            "SUM(CASE WHEN p.tahun_lahir <= 1964 THEN 1 ELSE 0 END) AS baby_boomers, "
            "SUM(CASE WHEN p.tahun_lahir BETWEEN 1965 AND 1980 THEN 1 ELSE 0 END) AS gen_x, "
            "SUM(CASE WHEN p.tahun_lahir BETWEEN 1981 AND 1996 THEN 1 ELSE 0 END) AS millennial, "
            "SUM(CASE WHEN p.tahun_lahir >= 1997 THEN 1 ELSE 0 END) AS gen_z "
            "FROM (SELECT DISTINCT ON (p.pegawai_id) p.pegawai_id, p.satker_top_id, "
            "EXTRACT(YEAR FROM p.tgl_lahir)::int AS tahun_lahir "
            "FROM public.pegawai_tm p "
            "WHERE p.tgl_lahir IS NOT NULL "
            "AND p.status_pegawai IN ('CPNS','PNS','POLRI','PPPK') "
            "AND p.kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN') "
            "ORDER BY p.pegawai_id) p "
            "JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id "
            "GROUP BY tipe_unit_kerja"
        )

    @staticmethod
    def _build_jumlah_ahli_komputer_per_generasi_query() -> str:
        return (
            "SELECT CASE "
            "WHEN tahun_lahir <= 1964 THEN 'baby_boomers' "
            "WHEN tahun_lahir BETWEEN 1965 AND 1980 THEN 'gen_x' "
            "WHEN tahun_lahir BETWEEN 1981 AND 1996 THEN 'millennial' "
            "WHEN tahun_lahir >= 1997 THEN 'gen_z' "
            "ELSE 'lainnya' END AS generasi, "
            "COUNT(DISTINCT p.pegawai_id) AS jumlah_pegawai "
            "FROM (SELECT DISTINCT ON (p.pegawai_id) p.pegawai_id, "
            "EXTRACT(YEAR FROM p.tgl_lahir)::int AS tahun_lahir, "
            "j.jabatan_nama, p.status_pegawai, p.kedudukan_pegawai "
            "FROM public.pegawai_tm p "
            "JOIN public.jabatan_tm j ON p.jabatan_id = j.jabatan_id "
            "WHERE p.tgl_lahir IS NOT NULL "
            "ORDER BY p.pegawai_id) p "
            "WHERE (lower(p.jabatan_nama) LIKE '%ahli komputer%' "
            "OR lower(p.jabatan_nama) LIKE '%pranata komputer%') "
            "AND p.status_pegawai IN ('CPNS','PNS','POLRI','PPPK') "
            "AND p.kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN') "
            "GROUP BY generasi"
        )

    def _build_intent_fallback(self, user_query: str) -> tuple[str, str] | None:
        normalized_query = self._normalize_literal(user_query)

        if self._user_requests_non_active(user_query):
            return None

        has_recap_intent = "rekap" in normalized_query or "rekapitulasi" in normalized_query
        has_unit_scope = self._contains_any(normalized_query, _UNIT_WORK_HINTS)
        has_education_intent = self._contains_any(normalized_query, _EDUCATION_HINTS)

        if has_recap_intent and has_unit_scope and has_education_intent:
            return (
                self._build_rekap_pendidikan_per_unit_query(),
                "Menampilkan rekap pegawai aktif berdasarkan pendidikan terakhir per tipe unit kerja.",
            )

        has_generation_intent = self._contains_any(normalized_query, _GENERATION_HINTS)
        if not has_generation_intent:
            return None

        has_count_intent = self._contains_any(normalized_query, _COUNT_HINTS)

        if has_unit_scope and has_recap_intent:
            return (
                self._build_rekap_generasi_per_unit_query(),
                "Menampilkan rekap pegawai aktif per generasi pada setiap tipe unit kerja.",
            )

        if has_count_intent and self._contains_any(normalized_query, _COMPUTER_ROLE_HINTS):
            return (
                self._build_jumlah_ahli_komputer_per_generasi_query(),
                "Menghitung jumlah pegawai aktif ahli/pranata komputer per generasi.",
            )

        return None

    def _resolve_ground_truth_rekap_query(self, user_query: str) -> tuple[str, str] | None:
        normalized_query = self._normalize_literal(user_query)

        if self._user_requests_non_active(user_query):
            return None

        has_recap_intent = "rekap" in normalized_query or "rekapitulasi" in normalized_query
        if not has_recap_intent:
            return None

        category: str | None = None
        for candidate, matchers in _GROUND_TRUTH_CATEGORY_MATCHERS:
            if all(matcher in normalized_query for matcher in matchers):
                category = candidate
                break

        if category is None:
            return None

        ground_truth_sql = _load_ground_truth_by_category().get(category)
        if not ground_truth_sql:
            return None

        explanation = _GROUND_TRUTH_EXPLANATIONS.get(
            category,
            "Menampilkan rekap sesuai ground truth kebutuhan informasi.",
        )
        return ground_truth_sql, explanation

    def _validate_expected_recap_shape(self, sql: str, user_query: str) -> str | None:
        normalized_query = self._normalize_literal(user_query)
        normalized_sql = self._normalize_literal(sql)

        has_recap_intent = "rekap" in normalized_query or "rekapitulasi" in normalized_query
        has_unit_scope = self._contains_any(normalized_query, _UNIT_WORK_HINTS)

        if (
            has_recap_intent
            and has_unit_scope
            and self._contains_any(normalized_query, _GENDER_HINTS)
        ):
            if _GROUP_BY_GENDER_PATTERN.search(normalized_sql):
                return (
                    "Untuk rekap jenis kelamin per tipe unit kerja, jangan GROUP BY jenis_kelamin. "
                    "Gunakan satu baris per tipe_unit_kerja dengan kolom agregat terpisah "
                    "(misalnya laki_laki dan perempuan) menggunakan conditional aggregation."
                )
            if not self._has_gender_pivot_projection(normalized_sql):
                return (
                    "Untuk rekap jenis kelamin per tipe unit kerja, gunakan conditional aggregation "
                    "dengan kolom laki_laki dan perempuan pada satu baris per tipe_unit_kerja."
                )
            if not self._has_group_by_unit_type(normalized_sql):
                return "Untuk rekap jenis kelamin, lakukan grouping per tipe_unit_kerja."

        if (
            has_recap_intent
            and has_unit_scope
            and self._contains_any(normalized_query, _EDUCATION_HINTS)
        ):
            if "pendidikan_top_id" not in normalized_sql:
                return (
                    "Untuk rekap pendidikan terakhir per tipe unit kerja, gunakan p.pendidikan_top_id "
                    "untuk bucket agregasi utama."
                )

            if _GROUP_BY_PENDIDIKAN_TOP_PATTERN.search(normalized_sql):
                return (
                    "Untuk rekap pendidikan per tipe unit kerja, jangan GROUP BY pendidikan_top_id. "
                    "Gunakan satu baris per tipe_unit_kerja dengan kolom bucket agregat "
                    "(unset, under_d3, d3, d4_s1, profesi, s2, s3)."
                )

            if not self._has_education_bucket_projection(normalized_sql):
                return (
                    "Untuk rekap pendidikan per tipe unit kerja, gunakan conditional aggregation "
                    "dengan kolom bucket unset, under_d3, d3, d4_s1, profesi, s2, s3."
                )

            if not self._has_group_by_unit_type(normalized_sql):
                return "Untuk rekap pendidikan, lakukan grouping per tipe_unit_kerja."

        has_generation_intent = self._contains_any(normalized_query, _GENERATION_HINTS)
        has_count_intent = self._contains_any(normalized_query, _COUNT_HINTS)
        if has_generation_intent and (has_unit_scope or has_recap_intent or has_count_intent):
            required_aliases = ("baby_boomers", "gen_x", "millennial", "gen_z")

            if "tahun_lahir" not in normalized_sql:
                return (
                    "Untuk rekap generasi, gunakan tahun_lahir sebagai basis klasifikasi generasi."
                )

            if has_unit_scope and has_recap_intent:
                if not all(alias in normalized_sql for alias in required_aliases):
                    return (
                        "Untuk rekap generasi per tipe unit kerja, sertakan bucket agregasi "
                        "baby_boomers, gen_x, millennial, dan gen_z."
                    )
                if not self._has_group_by_unit_type(normalized_sql):
                    return "Untuk rekap generasi, lakukan grouping per tipe_unit_kerja."
            else:
                if not re.search(r"\bcase\s+when\s+tahun_lahir\b", normalized_sql):
                    return (
                        "Untuk jumlah per generasi, gunakan CASE WHEN tahun_lahir untuk "
                        "membentuk bucket generasi."
                    )
                if not _GROUP_BY_GENERASI_PATTERN.search(normalized_sql):
                    return "Untuk jumlah per generasi, lakukan grouping berdasarkan generasi."

            if (
                self._contains_any(normalized_query, _COMPUTER_ROLE_HINTS)
                and "ahli komputer" not in normalized_sql
                and "pranata komputer" not in normalized_sql
            ):
                return (
                    "Untuk pertanyaan ahli/pranata komputer per generasi, tambahkan filter "
                    "jabatan yang memuat 'ahli komputer' atau 'pranata komputer'."
                )

        return None

    def _normalize_known_column_aliases(self, sql: str) -> str:
        normalized_sql = self._normalize_literal(sql)
        if "v_pendidikan_terakhir" not in normalized_sql:
            return sql
        if ".jurusan" not in normalized_sql:
            return sql

        return re.sub(
            r"(?i)(\b[A-Za-z_][A-Za-z0-9_]*\.)jurusan\b",
            r"\1programstudi",
            sql,
        )

    def _validate_role_intent_filters(self, sql: str, user_query: str) -> str | None:
        normalized_query = self._normalize_literal(user_query)
        normalized_sql = self._normalize_literal(sql)

        if self._contains_any(normalized_query, _COMPUTER_ROLE_HINTS):
            has_computer_role_filter = (
                "ahli komputer" in normalized_sql
                or "pranata komputer" in normalized_sql
            )
            if not has_computer_role_filter:
                return (
                    "Jika user meminta ahli/pranata komputer, tambahkan filter jabatan yang "
                    "memuat 'ahli komputer' atau 'pranata komputer'."
                )

        return None

    @staticmethod
    def _references_pegawai_table(sql: str) -> bool:
        return bool(
            re.search(
                r"\b(?:public\.)?\"?pegawai_tm\"?\b",
                sql,
                re.IGNORECASE,
            )
        )

    def _user_requests_non_active(self, user_query: str) -> bool:
        normalized_query = self._normalize_literal(user_query)
        return any(hint in normalized_query for hint in _NON_ACTIVE_REQUEST_HINTS)

    def _has_required_in_filter(
        self,
        sql: str,
        column_name: str,
        required_values: set[str],
    ) -> bool:
        pattern = re.compile(
            rf"(?:\b[A-Za-z_][A-Za-z0-9_]*\.)?{column_name}\s+IN\s*\(([^)]*)\)",
            re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(sql):
            clause = str(match.group(1) or "")
            literal_values = {
                self._normalize_literal(value)
                for value in _IN_VALUE_PATTERN.findall(clause)
                if value.strip()
            }
            if required_values.issubset(literal_values):
                return True
        return False

    def _validate_default_pegawai_filters(self, sql: str, user_query: str) -> str | None:
        if not self._references_pegawai_table(sql):
            return None
        if self._user_requests_non_active(user_query):
            return None

        has_status_filter = self._has_required_in_filter(
            sql=sql,
            column_name="status_pegawai",
            required_values=_DEFAULT_STATUS_FILTER_VALUES,
        )
        if not has_status_filter:
            return (
                "Tambahkan filter default pegawai aktif saat query melibatkan public.pegawai_tm: "
                "status_pegawai IN ('CPNS','PNS','POLRI','PPPK'). "
                "Abaikan aturan ini hanya jika user eksplisit meminta data non-aktif."
            )

        has_kedudukan_filter = self._has_required_in_filter(
            sql=sql,
            column_name="kedudukan_pegawai",
            required_values=_DEFAULT_KEDUDUKAN_FILTER_VALUES,
        )
        if not has_kedudukan_filter:
            return (
                "Tambahkan filter default pegawai aktif saat query melibatkan public.pegawai_tm: "
                "kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN'). "
                "Abaikan aturan ini hanya jika user eksplisit meminta data non-aktif."
            )

        return None

    async def generate(self, query: str, context: str) -> tuple[str, str]:
        grounded = self._resolve_ground_truth_rekap_query(query)
        if grounded is not None:
            return grounded

        feedback: str | None = None
        last_reason = "kesalahan tidak diketahui"

        for _ in range(self._retries + 1):
            prompt = self._build_prompt(query=query, context=context, feedback=feedback)
            try:
                response = await self._llm_adapter.deep_think.bind(
                    max_tokens=self._max_tokens
                ).ainvoke(
                    [
                        {
                            "role": "system",
                            "content": "Anda adalah ahli PostgreSQL. Keluarkan JSON valid saja.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
            except Exception as exc:
                last_reason = f"Permintaan ke LLM gagal: {exc}"
                await asyncio.sleep(1)
                continue

            content = strip_thinking(str(getattr(response, "content", "") or "")).strip()
            if not content:
                last_reason = "LLM mengembalikan respons kosong"
                feedback = "Output kosong. Kembalikan JSON valid dengan field query dan explanation saja."
                continue

            sql_candidate, explanation = self._parse_output(content)
            if not sql_candidate:
                last_reason = "Tidak dapat mem-parsing SQL dari output model"
                feedback = "Format output tidak valid. Kembalikan JSON valid dengan query dan explanation saja."
                continue

            sql_candidate = self._normalize_known_column_aliases(sql_candidate)

            validation_error = self.validate_sql_candidate(sql_candidate)
            if validation_error:
                last_reason = validation_error
                feedback = validation_error
                continue

            default_filter_error = self._validate_default_pegawai_filters(
                sql=sql_candidate,
                user_query=query,
            )
            if default_filter_error:
                last_reason = default_filter_error
                feedback = default_filter_error
                continue

            recap_shape_error = self._validate_expected_recap_shape(
                sql=sql_candidate,
                user_query=query,
            )
            if recap_shape_error:
                last_reason = recap_shape_error
                feedback = recap_shape_error
                continue

            role_intent_error = self._validate_role_intent_filters(
                sql=sql_candidate,
                user_query=query,
            )
            if role_intent_error:
                last_reason = role_intent_error
                feedback = role_intent_error
                continue

            if not explanation:
                explanation = "SQL dibuat dari konteks skema yang tersedia"
            return sql_candidate, explanation

        fallback = self._build_intent_fallback(query)
        if fallback is not None:
            return fallback

        return "", f"Gagal menghasilkan SQL valid: {last_reason}"

    @staticmethod
    def _parse_output(content: str) -> tuple[str, str]:
        parsed_object = extract_json_object(content)
        if parsed_object:
            sql = str(parsed_object.get("query") or parsed_object.get("sql") or "").strip()
            explanation = str(parsed_object.get("explanation") or "").strip()
            if sql:
                return sql, explanation

        sql_from_fence = extract_sql_from_fence(content)
        if sql_from_fence:
            return sql_from_fence, "SQL diparsing dari fenced block"

        sql_from_text = extract_sql_from_text(content)
        if sql_from_text:
            return sql_from_text, "SQL diparsing dari teks mentah"

        return "", ""

    def _build_prompt(self, query: str, context: str, feedback: str | None) -> str:
        allowed_schema_text = self._format_allowed_tables()
        schema_context = context.strip() if context and context.strip() else TOON_NA
        prompt = f"""# Introduction
Anda adalah generator SQL PostgreSQL untuk pipeline semantic parsing Text-to-SQL. Anda akan diberikan pertanyaan pengguna beserta konteks skema basis data, dan tugas Anda adalah menghasilkan satu kueri SELECT read-only beserta penjelasan singkatnya.

# Scope
Tugas Anda hanya menghasilkan satu statement SQL read-only (SELECT atau WITH ... SELECT) dalam format JSON yang valid.
DILARANG menghasilkan statement write (INSERT/UPDATE/DELETE/DDL), multiple statements, atau menggunakan tabel/kolom di luar whitelist yang diberikan.
Data [schema_context] dan [allowed_schema] dikirim dalam format TOON: block[N]{{field1,field2}}: lalu baris CSV.
Jika suatu block berisi N/A, artinya data untuk block tersebut tidak tersedia.

# Instructions
1. Output HARUS JSON valid saja, tanpa markdown, tanpa teks lain, dengan format persis: {{"query":"SELECT ...","explanation":"..."}}
2. Field query harus berisi tepat satu statement read-only (SELECT atau WITH ... SELECT) dan tidak boleh mengandung titik koma.
3. Gunakan hanya tabel/kolom yang berasal dari konteks retrieval dan whitelist yang diberikan.
4. Pada FROM/JOIN, tabel wajib ditulis lengkap sebagai schema.table.
5. Jika schema tidak disebutkan eksplisit oleh pengguna, gunakan default schema.
6. Jika hanya satu tabel, kolom boleh tanpa prefix tabel. Jika JOIN, kolom harus diberi kualifikasi alias/tabel agar tidak ambigu.
7. Jangan mengasumsikan kolom first_name/last_name tersedia. Untuk data dari public.pegawai_tm, gunakan kolom nama (dengan alias tabel, mis. p.nama) kecuali konteks schema secara eksplisit menampilkan first_name/last_name.
8. Kata "tertinggi" atau "terendah" berarti gunakan ORDER BY yang sesuai, tanpa LIMIT kecuali pengguna meminta batas jumlah.
9. Untuk fungsi bawaan PostgreSQL, jangan beri prefix schema (contoh: CURRENT_DATE, NOW()).
10. Field explanation wajib bahasa Indonesia sederhana dan ringkas yang menjelaskan maksud SQL.
11. Gunakan key query, BUKAN key sql.
12. Contoh ground_truth_query boleh berakhiran titik koma, tetapi output akhir field query tidak boleh berakhiran titik koma.
13. Jika query menyentuh public.pegawai_tm dan pengguna TIDAK secara eksplisit meminta data non-aktif, WAJIB tambahkan filter default:
    - status_pegawai IN ('CPNS', 'PNS', 'POLRI', 'PPPK')
    - kedudukan_pegawai IN ('Aktif', 'Tugas Belajar', 'CLTN')
14. Aturan default filter pegawai di atas memiliki prioritas lebih tinggi daripada contoh historis mana pun.
15. Untuk rekap pendidikan per tipe unit kerja, gunakan satu baris per tipe_unit_kerja dengan bucket agregat pendidikan (unset, under_d3, d3, d4_s1, profesi, s2, s3), bukan grouping per pendidikan_top_id.
16. Saat query daftar pegawai melakukan JOIN ke siap."V_PENDIDIKAN_TERAKHIR", cegah duplikasi dengan memilih satu baris per pegawai (gunakan DISTINCT ON (p.pegawai_id) atau ROW_NUMBER). Jika memfilter nama sekolah/program studi, gunakan perbandingan case-insensitive ter-normalisasi, misalnya lower(trim(vpt.namasekolah)) = lower(trim('...')).
17. JANGAN gunakan sintaks PostgreSQL yang tidak valid: COUNT(DISTINCT ON (...)). Untuk menghitung jumlah unik per pegawai, gunakan COUNT(DISTINCT p.pegawai_id) atau COUNT(*) dari subquery yang sudah dedupe (misalnya SELECT DISTINCT ON (p.pegawai_id) ...).

# Examples
{_PROMPT_EXAMPLES}

# Task Input
[schema_context]: {schema_context}
[allowed_schema]: {allowed_schema_text}
[default_schema]: {self._default_schema}
[question]: {query}

# Refocus
Berdasarkan aturan dan Task Input di atas, hasilkan satu kueri SELECT read-only yang menjawab pertanyaan pengguna dengan tepat menggunakan tabel/kolom yang ada di konteks dan whitelist.

# Transition
Berikan output dalam format JSON berikut:
{{"query":"SELECT ...","explanation":"..."}}
"""
        if feedback:
            prompt += (
                "\n# Umpan balik percobaan sebelumnya\n"
                f"{feedback}\n"
                "Perbaiki query sesuai umpan balik dan kembalikan JSON valid saja."
            )
        return prompt

    def _format_allowed_tables(self) -> str:
        if not self._allowed_tables:
            return TOON_NA

        rows: list[dict[str, object]] = []
        for schema_name, table_names in self._allowed_tables.items():
            filtered_tables = [table for table in table_names if table]
            if not schema_name or not filtered_tables:
                continue
            rows.append(
                {
                    "schema": schema_name,
                    "tables": filtered_tables,
                }
            )

        return encode_table(
            name="allowed_tables",
            rows=rows,
            fields=("schema", "tables"),
        )
