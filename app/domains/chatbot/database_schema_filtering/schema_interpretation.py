"""Stage 2 schema-interpretation sampler: catalog + prompt + parser.

Diberi [pertanyaan] pengguna dan [katalog_skema] (tabel + KOLOM konkret hasil
retrieval), LLM meng-generate SATU *interpretasi skema* — penjelasan singkat
bagaimana tiap konsep penting di pertanyaan dipetakan ke kolom/tabel konkret.
Dipanggil M kali @ temperature > 0 sehingga konsep yang benar-benar ambigu
menghasilkan pemetaan yang berbeda (→ ter-cluster terpisah di semantic entropy),
sementara konsep yang jelas konvergen ke satu cluster.

Mirror Stage 1 (``semantic_disambiguation``) Prinsip C: bila ada >1 pemetaan
yang sama-sama masuk akal, pilih SATU secara acak (uniform) — divergensi antar
panggilan adalah sinyal yang diukur Stage 2 UQ.

Catatan grounding (penting): interpretasi HARUS menyebut kolom konkret dari
[katalog_skema], bukan sekadar nama tabel. Tanpa kolom, istilah ambigu seperti
"senior" akan selalu kolaps ke ``senior → pegawai_tm`` di setiap sampel
(H_norm=0, ambiguitas tak terdeteksi). Katalog memuat kolom level-kolom dari
tabel kandidat teratas supaya alternatif pemetaan benar-benar terlihat LLM.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Batas katalog supaya prompt tetap ringkas namun memuat alternatif kolom yang
# cukup untuk memunculkan percabangan interpretasi.
_MAX_CATALOG_TABLES = 8
_MAX_COLS_PER_TABLE = 24


def format_schema_catalog(
    predicted_tables: dict[str, Any],
    schema_tables: list[dict[str, Any]],
    column_roles: dict[str, dict[str, str]] | None = None,
    *,
    max_tables: int = _MAX_CATALOG_TABLES,
    max_cols: int = _MAX_COLS_PER_TABLE,
) -> str:
    """Bangun katalog tabel+kolom ringkas dari hasil retrieval.

    Berbeda dari ``context`` retrieval (yang memfilter kolom berdasarkan
    similarity ke keyword, sehingga kolom alternatif untuk istilah ambigu
    sering hilang), katalog ini menampilkan kolom-kolom konkret tiap tabel
    kandidat teratas APA ADANYA. Tujuannya: memberi LLM cukup alternatif
    pemetaan agar interpretasi yang ambigu benar-benar bercabang.

    Mengembalikan string multi-baris ``- schema.table (skor X): col, col(PK),
    col(FK→...), ...``. String kosong bila tidak ada tabel.
    """
    if not predicted_tables or not schema_tables:
        return ""

    roles = column_roles or {}
    by_key: dict[str, dict[str, Any]] = {}
    for tbl in schema_tables:
        key = f"{tbl.get('schema')}.{tbl.get('name')}"
        by_key[key] = tbl

    def _score(key: str) -> float:
        rt = predicted_tables.get(key)
        return float(getattr(rt, "score", 0.0) or 0.0)

    ordered_keys = sorted(predicted_tables.keys(), key=_score, reverse=True)

    lines: list[str] = []
    for key in ordered_keys[:max_tables]:
        tbl = by_key.get(key)
        if not tbl:
            continue
        col_roles = roles.get(key, {})
        names: list[str] = []
        for column in tbl.get("columns", [])[:max_cols]:
            col_name = str(column.get("name") or "").strip()
            if not col_name:
                continue
            role = col_roles.get(col_name)
            names.append(f"{col_name}({role})" if role else col_name)
        if not names:
            continue
        lines.append(f"- {key} (skor {_score(key):.2f}): {', '.join(names)}")

    return "\n".join(lines)


def build_schema_interpretation_prompt(query: str, schema_catalog: str) -> str:
    catalog = (schema_catalog or "").strip()
    catalog_block = catalog[:6000] if catalog else "(tidak ada kandidat skema)"
    return f"""# Peran
Anda penginterpretasi skema untuk pipeline Text-to-SQL basis data kepegawaian BPOM. Diberi [pertanyaan] pengguna dan [katalog_skema] (tabel + kolom kandidat hasil retrieval), tugas Anda: tentukan bagaimana setiap konsep/istilah penting di [pertanyaan] dipetakan ke KOLOM konkret (bukan sekadar tabel) di [katalog_skema].

# Tujuan
Tahap ini BUKAN mencari jawaban paling benar. Tahap ini mengukur seberapa banyak pemetaan skema yang berbeda namun sama-sama masuk akal untuk pertanyaan ini. Karena itu, bila sebuah istilah bisa dipetakan ke beberapa kolom, JANGAN langsung memilih yang menurut Anda terbaik — perlakukan semua kandidat sebagai sama mungkin.

# Prinsip
1. Untuk SETIAP konsep/istilah penting di pertanyaan, petakan ke kolom konkret berformat ``tabel.kolom`` (sebut nama kolomnya, jangan berhenti di nama tabel).
2. Banyak istilah punya >1 kolom yang SAMA-SAMA masuk akal. Untuk istilah seperti itu, PILIH SATU secara acak (lempar koin) di antara semua kandidat yang masuk akal — JANGAN otomatis memilih yang paling umum/dominan/menurut Anda paling benar. Variasi antar panggilan justru SANGAT diharapkan dan menjadi sinyal deteksi ambiguitas.
3. Hanya gunakan tabel/kolom yang ADA di [katalog_skema]. Jangan mengarang nama kolom.
4. Tulis ``interpretasi`` sebagai 1-2 kalimat DESKRIPTIF dan mandiri: untuk tiap konsep, sebut makna yang dipilih DAN kolom konkretnya. Dua interpretasi yang memilih kolom berbeda harus berbunyi jelas berbeda.
5. PENTING: turunkan jawaban HANYA dari [pertanyaan] dan [katalog_skema] di bawah. JANGAN menyalin atau meniru kata/contoh apa pun dari instruksi ini — tidak ada contoh isi yang diberikan dengan sengaja agar Anda tidak terpaku pada satu interpretasi.

# Input
[pertanyaan]: {query}
[katalog_skema]:
{catalog_block}

# Output JSON (tanpa teks tambahan, isi sesuai pertanyaan & katalog di atas):
{{
  "penalaran": "<alasan singkat: konsep mana yang ambigu, kolom mana yang Anda pilih kali ini, dan kenapa>",
  "interpretasi": "<1-2 kalimat deskriptif yang menyebut tiap konsep penting beserta kolom konkret (tabel.kolom) yang Anda pilih>"
}}
"""


def _extract_json_object(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def parse_schema_interpretation(content: str) -> str:
    """Parse output sampler → string ``interpretasi``.

    Mengembalikan string kosong bila JSON invalid atau field hilang. Caller
    memperlakukan string kosong sebagai sample gagal (tidak di-embed, dihitung
    sebagai ``ERROR`` fingerprint di UQ).
    """
    parsed = _extract_json_object(content)
    if parsed is None:
        return ""
    return str(parsed.get("interpretasi") or "").strip()


__all__ = [
    "format_schema_catalog",
    "build_schema_interpretation_prompt",
    "parse_schema_interpretation",
]
