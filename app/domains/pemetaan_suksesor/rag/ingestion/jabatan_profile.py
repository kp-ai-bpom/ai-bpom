"""
Parse jabatan XLSX and categorise requirements into 14 structured fields.
Deterministic — no LLM required.
"""

import io
import re
import string

import pandas as pd


# ── Keyword mapping for 14 persyaratan fields ──────────────────────────────

PERSYARATAN_KEYWORDS: dict[str, list[str]] = {
    "status_kepegawaian": [
        "pegawai negeri sipil",
        "berstatus sebagai pns",
        "berstatus sebagai pegawai negeri",
    ],
    "usia_maksimal": ["usia paling tinggi", "batasan usia"],
    "pangkat_golongan": ["pangkat/golongan", "pangkat/golongan ruang", "golongan ruang"],
    "kualifikasi_pendidikan": ["kualifikasi pendidikan", "pendidikan paling rendah"],
    "pengalaman_jabatan_struktural": [
        "pengalaman jabatan",
        "jabatan administrator",
        "eselon",
        "jabatan fungsional ahli",
    ],
    "pelatihan_kepemimpinan": ["diklat kepemimpinan", "pelatihan kepemimpinan"],
    "penilaian_prestasi": ["penilaian prestasi", "prestasi kerja"],
    "hukuman_disiplin": ["hukuman disiplin"],
    "kesehatan": ["sehat jasmani", "kesehatan", "surat keterangan dokter"],
    "pelaporan_kekayaan_pajak": ["lhkpn", "lhkasn", "spt tahunan", "pelaporan kekayaan"],
    "integritas": ["pakta integritas", "rekam jejak", "moralitas yang baik"],
    "kompetensi": ["kompetensi manajerial", "kompetensi teknis", "kompetensi jabatan"],
    "kemampuan_bahasa": ["bahasa inggris", "berbahasa inggris"],
}


# ── Helper utilities ───────────────────────────────────────────────────────

def _get_col(row, idx: int) -> str:
    if idx >= len(row):
        return ""
    val = row[idx]
    return str(val).strip() if not bool(pd.isna(val)) else ""


def _is_reference_note(text: str) -> bool:
    if not text:
        return False
    if len(text) < 8:
        return True
    text_lower = text.lower()
    if text_lower.startswith("sesuai"):
        return True
    if "perbpom" in text_lower:
        return True
    if "mengacu" in text_lower:
        return True
    if "ikhtisar" in text_lower:
        return True
    if "selter" in text_lower:
        return True
    if re.search(r"\bSKJ\b", text):
        return True
    if re.search(r"\bSOTK\b", text):
        return True
    return False


def _detect_content_col(df) -> int:
    """Vote between col4 and col5 to find which holds actual content."""
    if len(df.columns) >= 7:
        return 5
    col4_scores = 0
    col5_scores = 0
    for _, row in df.iterrows():
        col3 = _get_col(row, 3)
        if col3.isdigit():
            c4 = _get_col(row, 4)
            c5 = _get_col(row, 5)
            if c4 and not _is_reference_note(c4):
                col4_scores += 1
            if c5 and not _is_reference_note(c5):
                col5_scores += 1
    return 5 if col5_scores > col4_scores else 4


def _read_item_content(row, content_col: int) -> str:
    primary = _get_col(row, content_col)
    if primary and not _is_reference_note(primary):
        return primary
    fallback_col = 4 if content_col == 5 else 5
    fallback = _get_col(row, fallback_col)
    if fallback and not _is_reference_note(fallback):
        return fallback
    return ""


def slugify(name: str) -> str:
    slug = name.lower().strip()
    for ch in [".", ",", "(", ")", "/"]:
        slug = slug.replace(ch, "")
    slug = slug.replace(" ", "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


# ── Parsing ────────────────────────────────────────────────────────────────

def _parse_dataframe(df) -> dict:
    content_col = _detect_content_col(df)
    data: dict = {
        "nama_jabatan": "",
        "atasan_langsung": "",
        "tugas": "",
        "fungsi": [],
        "persyaratan": [],
    }
    current_section = None

    for _, row in df.iterrows():
        col1 = _get_col(row, 1)
        col3 = _get_col(row, 3)

        if "Nama Jabatan" in col1:
            data["nama_jabatan"] = col3
        elif "Atasan Langsung" in col1:
            data["atasan_langsung"] = col3
        elif "Tugas" in col1:
            data["tugas"] = col3
        elif "Fungsi" in col1:
            current_section = "Fungsi"
            content = _read_item_content(row, content_col)
            if content and col3.isdigit():
                data["fungsi"].append(content.rstrip(";"))
        elif "PERSYARATAN" in col1:
            current_section = "Persyaratan"
            content = _read_item_content(row, content_col)
            if content and col3.isdigit():
                data["persyaratan"].append(content.rstrip(";"))
        elif current_section == "Fungsi" and col3.isdigit():
            content = _read_item_content(row, content_col)
            if content:
                data["fungsi"].append(content.rstrip(";"))
        elif current_section == "Persyaratan" and col3.isdigit():
            content = _read_item_content(row, content_col)
            if content:
                data["persyaratan"].append(content.rstrip(";"))
        elif current_section == "Persyaratan" and not col3:
            col4 = _get_col(row, 4)
            if col4 and len(col4) == 1 and col4.isalpha():
                sub_content = _read_item_content(row, content_col)
                if sub_content and data["persyaratan"]:
                    data["persyaratan"][-1] += f"; {col4}. {sub_content.rstrip(';')}"

    return data


def parse_xlsx_bytes_for_profile(data: bytes) -> dict:
    df = pd.read_excel(io.BytesIO(data), header=None)
    return _parse_dataframe(df)


# ── Categorisation ─────────────────────────────────────────────────────────

def categorize_persyaratan(items: list[str]) -> dict:
    """Map free-text persyaratan items to 14 structured fields via keyword matching."""
    result: dict = {
        "status_kepegawaian": "",
        "usia_maksimal": "",
        "pangkat_golongan": "",
        "kualifikasi_pendidikan": "",
        "pengalaman_jabatan_struktural": "",
        "pelatihan_kepemimpinan": "",
        "penilaian_prestasi": "",
        "hukuman_disiplin": "",
        "kesehatan": "",
        "pelaporan_kekayaan_pajak": "",
        "integritas": [],
        "kompetensi": "",
        "kemampuan_bahasa": "",
        "pengalaman_bidang_tugas": {
            "durasi_kumulatif": "",
            "kompetensi_spesifik": [],
        },
    }

    for item in items:
        text = item.strip().rstrip(";").rstrip(".")
        text_lower = text.lower()

        # Priority 1: pengalaman_bidang_tugas (most specific)
        if (
            "pengalaman jabatan dalam bidang tugas" in text_lower
            or "pengalaman jabatan yang terkait" in text_lower
        ):
            parts = re.split(r";\s*[a-z]\.\s", text.rstrip(";").rstrip("."))
            result["pengalaman_bidang_tugas"]["durasi_kumulatif"] = (
                parts[0].rstrip(";").rstrip(".").strip()
            )
            if len(parts) > 1:
                sub_items = re.findall(r"([a-z]\.\s[^;]+)", text.rstrip(";").rstrip("."))
                result["pengalaman_bidang_tugas"]["kompetensi_spesifik"] = [
                    s.lstrip(string.ascii_lowercase + ". ").strip() for s in sub_items
                ]
            continue

        # Priority 2: technical competency sub-items
        if "kompetensi teknis" in text_lower and (
            "skj" in text_lower or "pfm" in text_lower or "ahli madya" in text_lower
        ):
            result["pengalaman_bidang_tugas"]["kompetensi_spesifik"].append(
                text.rstrip(";").rstrip(".")
            )
            continue

        # Priority 3: generic keyword matching
        matched = False
        for field, keywords in PERSYARATAN_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                if field == "integritas":
                    result[field].append(text.rstrip(";").rstrip("."))
                else:
                    result[field] = text.rstrip(";").rstrip(".")
                matched = True
                break

        if matched:
            continue

        # Priority 4: loose fallback
        if "integritas" in text_lower or "moralitas" in text_lower:
            result["integritas"].append(text.rstrip(";").rstrip("."))
        elif "kompetensi" in text_lower:
            result["kompetensi"] = text.rstrip(";").rstrip(".")
        elif "bahasa" in text_lower or "inggris" in text_lower:
            result["kemampuan_bahasa"] = text.rstrip(";").rstrip(".")
        elif "pengalaman" in text_lower and "bidang tugas" in text_lower:
            result["pengalaman_bidang_tugas"]["durasi_kumulatif"] = text.rstrip(";").rstrip(".")
        # Priority 5: skip (already-merged sub-items)

    return result


# ── Build final profil jabatan JSON ────────────────────────────────────────

def build_profil_jabatan(file_bytes: bytes) -> dict:
    """Parse XLSX bytes and return a structured profil jabatan dict."""
    parsed = parse_xlsx_bytes_for_profile(file_bytes)

    fungsi = [f.rstrip(";").rstrip(".").strip() for f in parsed["fungsi"] if f.strip()]
    persyaratan = categorize_persyaratan(parsed["persyaratan"])

    return {
        "slug": slugify(parsed["nama_jabatan"]),
        "nama_jabatan": parsed["nama_jabatan"],
        "atasan_langsung": parsed["atasan_langsung"],
        "data": {
            "deskripsi_jabatan": {
                "nama_jabatan": parsed["nama_jabatan"],
                "atasan_langsung": parsed["atasan_langsung"],
                "tugas": parsed["tugas"].rstrip(";").rstrip(".").strip(),
                "fungsi": fungsi,
            },
            "persyaratan": persyaratan,
        },
    }
