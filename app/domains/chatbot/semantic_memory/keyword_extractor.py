from app.core.llm import LLMAdapter

from .parsers import extract_json_array, strip_thinking, tokenize_keywords_fallback


_STOPWORDS = {
    "apa",
    "siapa",
    "berapa",
    "kapan",
    "dimana",
    "di",
    "ke",
    "dari",
    "yang",
    "dan",
    "atau",
    "dengan",
    "tampilkan",
    "buat",
    "semua",
    "itu",
    "ini",
    "jumlah",
    "list",
    "daftar",
    "pada",
    "untuk",
}


class KeywordExtractor:
    def __init__(
        self,
        llm_adapter: LLMAdapter,
        allowed_tables: dict[str, list[str]],
        retries: int,
    ):
        self._llm_adapter = llm_adapter
        self._allowed_tables = allowed_tables
        self._retries = max(0, retries)

    async def extract(self, query: str) -> list[str]:
        query = query.strip()
        if not query:
            return []

        schema_lines = []
        for schema_name, table_names in self._allowed_tables.items():
            table_hint = ", ".join(table_names)
            schema_lines.append(f"- {schema_name}: {table_hint}")
        schema_context = "\n".join(schema_lines) if schema_lines else "- public: (tidak tersedia)"

        system_prompt = f"""# Introduction
Anda akan diberikan pertanyaan pengguna yang dapat dijawab dengan melakukan kueri pada sistem basis data. Tujuan Anda adalah menganalisis pertanyaan untuk mengidentifikasi dan mengekstrak kata kunci (keywords) dan frasa kunci (keyphrases) yang dapat membantu menunjukkan bagian mana dari skema basis data yang harus digunakan.

# Scope
Fokuslah secara eksklusif pada entitas yang merepresentasikan skema basis data seperti nama tabel, nama kolom, atau konsep yang dapat dipetakan ke struktur data.
DILARANG KERAS berfokus pada nilai kolom spesifik (misalnya angka, nama orang, atau tanggal).

# Skema Tersedia
{schema_context}

# Instructions
1. Pahami maksud utama pertanyaan.
2. Ekstrak kata kunci (keywords) berupa istilah inti.
3. Ekstrak frasa kunci (keyphrases) berupa konsep penting.
4. Pastikan kata kunci berkaitan dengan struktur skema (tabel/kolom).
5. Batasi output antara 5-10 kata kunci/frasa kunci yang paling relevan.

# Examples
## Contoh 1
[question]: "Tampilkan pegawai perempuan di BBPOM Medan"
[output]: ["pegawai", "jenis kelamin", "unit kerja", "lokasi", "BBPOM"]

## Contoh 2
[question]: "Berapa jumlah pegawai yang akan pensiun tahun 2026?"
[output]: ["pegawai", "jumlah", "pensiun", "tahun", "tgl_pensiun"]

# Task Input
[question]: {query}

# Refocus
Identifikasi kata kunci dan frasa kunci yang dapat membantu pemilihan tabel dan kolom dalam basis data.

# Transition
Keluarkan output sebagai JSON array SAJA, tanpa teks lain.
[output]:
"""

        for _ in range(self._retries + 1):
            try:
                response = await self._llm_adapter.think.bind(max_tokens=1200).ainvoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": "Ekstrak keyword sesuai instruksi dan keluarkan JSON array saja.",
                        },
                    ]
                )
            except Exception:
                continue

            content = strip_thinking(str(getattr(response, "content", "") or ""))
            keywords = extract_json_array(content)
            if keywords:
                normalized = self._normalize_keywords(keywords)
                if normalized:
                    return normalized

        fallback = tokenize_keywords_fallback(
            query=query,
            stopwords=_STOPWORDS,
            max_keywords=6,
        )
        return self._normalize_keywords(fallback)

    @staticmethod
    def _normalize_keywords(keywords: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for keyword in keywords:
            value = keyword.strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)

        return normalized
