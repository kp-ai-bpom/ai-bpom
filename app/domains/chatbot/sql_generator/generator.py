import asyncio
import re

from app.core.llm import LLMAdapter

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


_PROMPT_EXAMPLES = """Examples (MUST follow this JSON format):
{"query": "SELECT * FROM public.pegawai_tm p WHERE p.status_pegawai IN ('CPNS','PNS','POLRI','PPPK') AND p.kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN') LIMIT 10", "explanation": "Menampilkan 10 pegawai aktif pertama"}

{"query": "SELECT * FROM siap.R_FUNGSI", "explanation": "Menampilkan semua data fungsi dari schema siap"}

{"query": "SELECT p.nama, f.nama_fungsi FROM public.pegawai_tm p JOIN siap.R_FUNGSI f ON p.fungsi_id = f.id WHERE p.status_pegawai IN ('CPNS','PNS','POLRI','PPPK') AND p.kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN')", "explanation": "Menampilkan pegawai aktif dengan nama fungsi"}

{"query": "SELECT DISTINCT ON (p.pegawai_id) p.nama, j.jabatan_nama, vpt.namasekolah, vpt.programstudi FROM public.pegawai_tm p JOIN public.jabatan_tm j ON p.jabatan_id = j.jabatan_id JOIN siap.\"V_PENDIDIKAN_TERAKHIR\" vpt ON vpt.pegawaiid = p.pegawai_id WHERE lower(trim(vpt.namasekolah)) = lower(trim('Universitas Komputer Indonesia')) AND (lower(j.jabatan_nama) LIKE '%ahli komputer%' OR lower(j.jabatan_nama) LIKE '%pranata komputer%') AND p.status_pegawai IN ('CPNS','PNS','POLRI','PPPK') AND p.kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN') ORDER BY p.pegawai_id, vpt.ranking ASC NULLS LAST", "explanation": "Menampilkan satu baris per pegawai ahli/pranata komputer lulusan Universitas Komputer Indonesia tanpa duplikasi karena perbedaan huruf besar-kecil"}

Contoh mapping pertanyaan ke query referensi:
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan jenis kelamin pada setiap tipe unit kerja",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(*) AS total_pegawai,\n  SUM(CASE WHEN p.jenis_kelamin = 'Laki-Laki' THEN 1 ELSE 0 END) AS laki_laki,\n  SUM(CASE WHEN p.jenis_kelamin = 'Perempuan' THEN 1 ELSE 0 END) AS perempuan\n FROM public.pegawai_tm p\n JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan pendidikan terakhir pada setiap tipe unit kerja",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(DISTINCT p.nama) AS total_pegawai,\n  SUM(CASE WHEN p.pendidikan_top_id IS NULL THEN 1 ELSE 0 END) as unset,\n  SUM(CASE WHEN p.pendidikan_top_id IN ('01','02','03','04','05','06','07') THEN 1 ELSE 0 END) as under_d3,\n  SUM(CASE WHEN p.pendidikan_top_id='08' THEN 1 ELSE 0 END) as d3,\n  SUM(CASE WHEN p.pendidikan_top_id IN ('09','10','11') THEN 1 ELSE 0 END) as d4_s1,\n  SUM(CASE WHEN p.pendidikan_top_id='12' THEN 1 ELSE 0 END) as profesi,\n  SUM(CASE WHEN p.pendidikan_top_id='13' THEN 1 ELSE 0 END) as s2,\n  SUM(CASE WHEN p.pendidikan_top_id='14' THEN 1 ELSE 0 END) as s3\n FROM public.pegawai_tm p\n JOIN public.\"SIAP_SATKER_TOP\" s\n  ON p.satker_top_id = s.satker_id\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan kelompok usia pada setiap tipe unit kerja",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(*) AS total_pegawai,\n  SUM(CASE WHEN usia <= 25 THEN 1 ELSE 0 END) AS usia_25,\n  SUM(CASE WHEN usia BETWEEN 26 AND 30 THEN 1 ELSE 0 END) AS usia_26_30,\n  SUM(CASE WHEN usia BETWEEN 31 AND 35 THEN 1 ELSE 0 END) AS usia_31_35,\n  SUM(CASE WHEN usia BETWEEN 36 AND 40 THEN 1 ELSE 0 END) AS usia_36_40,\n  SUM(CASE WHEN usia BETWEEN 41 AND 45 THEN 1 ELSE 0 END) AS usia_41_45,\n  SUM(CASE WHEN usia BETWEEN 46 AND 50 THEN 1 ELSE 0 END) AS usia_46_50,\n  SUM(CASE WHEN usia BETWEEN 51 AND 55 THEN 1 ELSE 0 END) AS usia_51_55,\n  SUM(CASE WHEN usia BETWEEN 56 AND 60 THEN 1 ELSE 0 END) AS usia_56_60,\n  SUM(CASE WHEN usia >= 61 THEN 1 ELSE 0 END) AS usia_61\n FROM (\n  SELECT DISTINCT ON (p.pegawai_id)\n  p.pegawai_id,\n  p.satker_top_id,\n  EXTRACT(YEAR FROM AGE(CURRENT_DATE, p.tgl_lahir))::int AS usia\n  FROM public.pegawai_tm p\n  WHERE p.tgl_lahir IS NOT NULL\n  ORDER BY p.pegawai_id\n ) p\n JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan generasi pada setiap tipe unit kerja",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(*) AS total_pegawai,\n  SUM(CASE WHEN tahun_lahir <= 1964 THEN 1 ELSE 0 END) AS baby_boomers,\n  SUM(CASE WHEN tahun_lahir BETWEEN 1965 AND 1980 THEN 1 ELSE 0 END) AS gen_x,\n  SUM(CASE WHEN tahun_lahir BETWEEN 1981 AND 1996 THEN 1 ELSE 0 END) AS millennial,\n  SUM(CASE WHEN tahun_lahir >= 1997 THEN 1 ELSE 0 END) AS gen_z\n FROM (\n  SELECT DISTINCT ON (p.pegawai_id)\n  p.pegawai_id,\n  p.satker_top_id,\n  EXTRACT(YEAR FROM p.tgl_lahir)::int AS tahun_lahir\n  FROM public.pegawai_tm p\n  WHERE p.tgl_lahir IS NOT NULL\n  ORDER BY p.pegawai_id\n ) p\n JOIN public.\"SIAP_SATKER_TOP\" s\n  ON p.satker_top_id = s.satker_id\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan tipe pegawai",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(DISTINCT p.pegawai_id) AS total_pegawai,\n  COUNT(*) FILTER (WHERE tp.deskripsi = 'Struktural') AS struktural,\n  COUNT(*) FILTER (WHERE tp.deskripsi = 'Pelaksana') AS pelaksana,\n  COUNT(*) FILTER (WHERE tp.deskripsi = 'Fungsional') AS fungsional\n FROM public.pegawai_tm p\n JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id\n JOIN public.tipepegawai_tm tp ON p.tipepegawai_id = tp.tipepegawai_id\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan status pegawai",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(DISTINCT p.pegawai_id) AS total_pegawai,\n  COUNT(*) FILTER (WHERE p.status_pegawai = 'CPNS') AS cpns,\n  COUNT(*) FILTER (WHERE p.status_pegawai = 'PNS') AS pns,\n  COUNT(*) FILTER (WHERE p.status_pegawai = 'PPPK') AS pppk\n FROM public.pegawai_tm p\n JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan kedudukan pegawai",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(*) AS total_pegawai,\n  COUNT(*) FILTER (WHERE kedudukan_pegawai = 'Aktif') AS aktif,\n  COUNT(*) FILTER (WHERE kedudukan_pegawai = 'Tugas Belajar') AS tugas_belajar,\n  COUNT(*) FILTER (WHERE kedudukan_pegawai = 'CLTN') AS cltn\n FROM public.pegawai_tm p\n JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan golongan",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(DISTINCT p.pegawai_id) AS total_pegawai,\n  COUNT(*) FILTER (WHERE sk.golongan = 'I') AS gol_i,\n  COUNT(*) FILTER (WHERE sk.golongan = 'II/a') AS ii_a,\n  COUNT(*) FILTER (WHERE sk.golongan = 'II/b') AS ii_b,\n  COUNT(*) FILTER (WHERE sk.golongan = 'II/c') AS ii_c,\n  COUNT(*) FILTER (WHERE sk.golongan = 'II/d') AS ii_d,\n  COUNT(*) FILTER (WHERE sk.golongan = 'III/a') AS iii_a,\n  COUNT(*) FILTER (WHERE sk.golongan = 'III/b') AS iii_b,\n  COUNT(*) FILTER (WHERE sk.golongan = 'III/c') AS iii_c,\n  COUNT(*) FILTER (WHERE sk.golongan = 'III/d') AS iii_d,\n  COUNT(*) FILTER (WHERE sk.golongan = 'IV/a') AS iv_a,\n  COUNT(*) FILTER (WHERE sk.golongan = 'IV/b') AS iv_b,\n  COUNT(*) FILTER (WHERE sk.golongan = 'IV/c') AS iv_c,\n  COUNT(*) FILTER (WHERE sk.golongan = 'IV/d') AS iv_d,\n  COUNT(*) FILTER (WHERE sk.golongan = 'IV/e') AS iv_e\n FROM public.pegawai_tm p\n JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id\n JOIN public.sk_pegawai_v sk ON p.pegawai_id = sk.pegawai_id\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan jenjang jabatan fungsional",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(DISTINCT p.pegawai_id) AS total_pegawai,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Terampil') AS terampil,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Mahir') AS mahir,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Penyelia') AS penyelia,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Pertama') AS ahli_pertama,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Muda') AS ahli_muda,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Madya') AS ahli_madya,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Utama') AS ahli_utama\n FROM public.pegawai_tm p\n JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id\n JOIN public.jabatan_tm j ON p.jabatan_id = j.jabatan_id\n WHERE j.jabatan_status = true AND j.jenis_jabatan = 'Tertentu'\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan jenjang jabatan fungsional PFM",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(DISTINCT p.pegawai_id) AS total_pegawai,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Terampil') AS terampil,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Mahir') AS mahir,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Penyelia') AS penyelia,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Pertama') AS ahli_pertama,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Muda') AS ahli_muda,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Madya') AS ahli_madya,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Utama') AS ahli_utama\n FROM public.pegawai_tm p\n JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id\n JOIN public.jabatan_tm j ON p.jabatan_id = j.jabatan_id\n WHERE j.jabatan_status = true AND j.jenis_jabatan = 'Tertentu'\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan jenjang jabatan fungsional Non-PFM",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(DISTINCT p.pegawai_id) AS total_pegawai,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Terampil') AS terampil,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Mahir') AS mahir,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Penyelia') AS penyelia,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Pertama') AS ahli_pertama,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Muda') AS ahli_muda,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Madya') AS ahli_madya,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Utama') AS ahli_utama\n FROM public.pegawai_tm p\n JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id\n JOIN public.jabatan_tm j ON p.jabatan_id = j.jabatan_id\n WHERE j.jabatan_status = true AND j.jenis_jabatan = 'Tertentu'\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan jenjang jabatan non fungsional",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(DISTINCT p.pegawai_id) AS total_pegawai,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Terampil') AS terampil,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Mahir') AS mahir,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Penyelia') AS penyelia,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Pertama') AS ahli_pertama,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Muda') AS ahli_muda,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Madya') AS ahli_madya,\n  COUNT(*) FILTER (WHERE j.jenjang_jabatan = 'Ahli Utama') AS ahli_utama\n FROM public.pegawai_tm p\n JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id\n JOIN public.jabatan_tm j ON p.jabatan_id = j.jabatan_id\n WHERE j.jabatan_status = true AND j.jenis_jabatan = 'Tertentu'\n GROUP BY tipe_unit_kerja"
}
{
    "question": "Tampilkan rekapitulasi jumlah pegawai berdasarkan eselon",
    "ground_truth_query": "SELECT\n  CASE\n  WHEN s.tipe_balai = 'P' THEN 'Unit Kerja Pusat'\n  WHEN s.tipe_balai = 'B' THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai IN ('BA','BB') THEN 'UPT Balai Besar / Balai POM'\n  WHEN s.tipe_balai = 'L' THEN 'UPT Loka POM'\n  END AS tipe_unit_kerja,\n  COUNT(DISTINCT p.pegawai_id) AS total_pegawai,\n  COUNT(*) FILTER (WHERE split_part(e.eselon_nama,'.',1) = 'I') AS eselon_i,\n  COUNT(*) FILTER (WHERE split_part(e.eselon_nama,'.',1) = 'II') AS eselon_ii,\n  COUNT(*) FILTER (WHERE split_part(e.eselon_nama,'.',1) = 'III') AS eselon_iii,\n  COUNT(*) FILTER (WHERE split_part(e.eselon_nama,'.',1) = 'IV') AS eselon_iv\n FROM public.pegawai_tm p\n JOIN public.\"SIAP_SATKER_TOP\" s ON p.satker_top_id = s.satker_id\n LEFT JOIN public.eselon_tm e ON p.eselon_id = e.eselon_id\n GROUP BY 1"
}
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

            if not explanation:
                explanation = "SQL dibuat dari konteks skema yang tersedia"
            return sql_candidate, explanation

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
        prompt = f"""Anda adalah generator SQL PostgreSQL untuk menjawab pertanyaan berbasis data.

# Konteks skema hasil retrieval
{context}

# Whitelist schema dan tabel yang boleh dipakai
{allowed_schema_text}

# Default schema
{self._default_schema}

# Pertanyaan pengguna
{query}

# Aturan wajib
1. Output HARUS JSON valid saja, tanpa markdown, tanpa teks lain, dengan format persis:
{{"query":"SELECT ...","explanation":"..."}}
2. Field query harus berisi tepat satu statement read-only (SELECT atau WITH ... SELECT) dan tidak boleh mengandung titik koma.
3. Gunakan hanya tabel/kolom yang berasal dari konteks retrieval dan whitelist di atas.
4. Pada FROM/JOIN, tabel wajib ditulis lengkap sebagai schema.table.
5. Jika schema tidak disebutkan eksplisit oleh pengguna, gunakan default schema.
6. Jika hanya satu tabel, kolom boleh tanpa prefix tabel. Jika JOIN, kolom harus diberi kualifikasi alias/tabel agar tidak ambigu.
7. Jangan mengasumsikan kolom first_name/last_name tersedia. Untuk data dari public.pegawai_tm, gunakan kolom nama (dengan alias tabel, mis. p.nama) kecuali konteks schema secara eksplisit menampilkan first_name/last_name.
8. Kata "tertinggi" atau "terendah" berarti gunakan ORDER BY yang sesuai, tanpa LIMIT kecuali pengguna meminta batas jumlah.
9. Untuk fungsi bawaan PostgreSQL, jangan beri prefix schema (contoh: CURRENT_DATE, NOW()).
10. explanation wajib bahasa Indonesia sederhana dan ringkas yang menjelaskan maksud SQL.
11. Gunakan key query, BUKAN key sql.
12. Contoh ground_truth_query boleh berakhiran titik koma, tetapi output akhir field query tidak boleh berakhiran titik koma.
13. Jika query menyentuh public.pegawai_tm dan pengguna TIDAK secara eksplisit meminta data non-aktif, WAJIB tambahkan filter default:
    - status_pegawai IN ('CPNS', 'PNS', 'POLRI', 'PPPK')
    - kedudukan_pegawai IN ('Aktif', 'Tugas Belajar', 'CLTN')
14. Aturan default filter pegawai di atas memiliki prioritas lebih tinggi daripada contoh historis mana pun.
15. Saat query daftar pegawai melakukan JOIN ke siap."V_PENDIDIKAN_TERAKHIR", cegah duplikasi dengan memilih satu baris per pegawai (gunakan DISTINCT ON (p.pegawai_id) atau ROW_NUMBER). Jika memfilter nama sekolah/program studi, gunakan perbandingan case-insensitive ter-normalisasi, misalnya lower(trim(vpt.namasekolah)) = lower(trim('...')).
16. JANGAN gunakan sintaks PostgreSQL yang tidak valid: COUNT(DISTINCT ON (...)). Untuk menghitung jumlah unik per pegawai, gunakan COUNT(DISTINCT p.pegawai_id) atau COUNT(*) dari subquery yang sudah dedupe (misalnya SELECT DISTINCT ON (p.pegawai_id) ...).

# Examples
{_PROMPT_EXAMPLES}

# Format jawaban
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
            return "- public: (tidak ada tabel whitelist)"

        lines: list[str] = []
        for schema_name, table_names in self._allowed_tables.items():
            lines.append(f"- {schema_name}: {', '.join(table_names)}")
        return "\n".join(lines)
