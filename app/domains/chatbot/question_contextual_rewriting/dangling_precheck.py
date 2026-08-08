"""Linguistic pre-check untuk dangling reference pada Stage 1 rewriter.

Module ini mengimplementasi *stratified context augmentation* untuk M-sampling
rewriter (Stage 1 Aleatoric Contextual). Latar belakang:

- Pada query Indonesia dengan demonstrative ("tersebut", "itu", "ini",
  "yang tadi", "di sana", "saat itu", "tadi") yang antecedent-nya tidak
  ada di working_memory, M-sampling rewriter dengan gpt-4o-mini cenderung
  mengalami **mode collapse**: semua M sample konvergen ke satu interpretasi
  modal (mis. semua memilih "per kota/kabupaten"). Akibatnya H_norm ≈ 0
  meskipun query secara linguistik sebenarnya ambigu.
- Mitigasi: ketika linguistik mendeteksi dangling reference, kita
  meng-augment prompt dengan **enumerasi referent plausible eksplisit**
  yang sudah disisipkan ke prompt setiap sample, dengan stratified
  assignment (sample-i diarahkan ke referent-(i mod K)) supaya M sample
  meng-cover ruang interpretasi secara penuh. Ini adalah teknik standar
  *stratified sampling* di uncertainty quantification untuk mengatasi
  mode collapse pada low-cost samplers.

Output H_norm dari pipeline UQ menjadi mencerminkan **ruang interpretasi
linguistik** alih-alih *modus posterior LLM* yang under-explores. Tau_U
(0.40) tetap dipakai apa adanya — interpretasinya menjadi: "berapa banyak
ruang interpretasi yang harus dieksplorasi sebelum sistem dapat percaya
diri menjawab".

Module ini sengaja konservatif: hanya men-trigger untuk kategori nomina
yang sudah dikamuskan secara eksplisit. Default behavior bila tidak match
adalah no-op (rewriter berjalan persis seperti tanpa modul ini).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Demonstrative Indonesia yang menjadi indikator referensi ke entitas
# di luar current_query. Disusun dari panjang ke pendek supaya regex
# alternation match yang lebih panjang dulu ("yang tadi" sebelum "tadi").
_DEMONSTRATIVES = [
    "yang tadi",
    "di sana",
    "saat itu",
    "tersebut",
    "tadi",
    "itu",
    "ini",
]


# Kamus kategori nomina → enumerasi referent plausible.
#
# Key = list bentuk kanonik nomina (untuk match). Value = list referent
# yang akan dipakai untuk stratified sampling. Urutan referent boleh
# acak — di runtime di-shuffle per sesi panggilan.
#
# Penambahan kategori: tambah entry baru. Sinonim/varian morfologis
# (mis. "lokasinya") di-handle via substring match.
_REFERENT_DICTIONARY: list[tuple[tuple[str, ...], list[str]]] = [
    (
        ("lokasi", "tempat", "daerah", "wilayah", "area"),
        [
            "per provinsi",
            "per kota/kabupaten",
            "per unit kerja/kantor",
            "di lokasi spesifik tertentu (misal kantor pusat)",
        ],
    ),
    (
        ("tahun", "waktu", "periode", "masa", "kurun"),
        [
            "pada tahun berjalan",
            "pada tahun sebelumnya",
            "pada rentang periode tertentu",
            "pada bulan/triwulan tertentu",
        ],
    ),
    (
        ("unit", "kantor", "instansi", "satuan kerja"),
        [
            "per unit kerja",
            "per direktorat/biro",
            "per kantor pusat",
            "per kantor daerah/wilayah",
        ],
    ),
    (
        ("bidang", "jabatan", "posisi", "kelompok"),
        [
            "per jenis jabatan",
            "per bidang kerja",
            "per kelompok eselon/golongan",
            "per spesialisasi",
        ],
    ),
    (
        ("kategori", "jenis", "tipe", "golongan", "kelompok"),
        [
            "per kategori",
            "per jenis",
            "per tipe utama",
            "per sub-kategori",
        ],
    ),
]


# Build flattened set of known noun roots dari _REFERENT_DICTIONARY.
# Dipakai _normalize_query untuk membatasi splitting preposisi (di/ke) hanya
# pada nomina yang dikenal — supaya kata legit seperti "kegiatan", "diklat",
# "kelompok" TIDAK ikut tersplit menjadi "ke giatan" / "di klat" dst.
_KNOWN_NOUN_ROOTS: tuple[str, ...] = tuple(
    sorted(
        {n for keys, _ in _REFERENT_DICTIONARY for n in keys},
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True)
class DanglingDetection:
    """Hasil deteksi dangling reference.

    Attributes:
        noun: nomina yang dimodifikasi demonstrative (mis. "lokasi").
        demonstrative: kata demonstrative yang muncul (mis. "tersebut").
        referents: list referent plausible untuk enumerasi M-sampling.
    """

    noun: str
    demonstrative: str
    referents: tuple[str, ...]


def _normalize_query(text: str) -> str:
    """Normalisasi ringan: lowercase + pisahkan 'di'/'ke' menempel pada
    nomina YANG DIKENAL saja.

    Contoh: "dilokasi tersebut" → "di lokasi tersebut". Tetapi "kegiatan",
    "kelompok", "diklat", dst TIDAK ikut tersplit karena bukan nomina di
    ``_KNOWN_NOUN_ROOTS``. Pendekatan whitelist ini menghindari false split
    pada kata bahasa Indonesia yang kebetulan berawalan "di"/"ke".
    """
    t = text.lower()
    if not _KNOWN_NOUN_ROOTS:
        return t
    # Susun alternation dari panjang ke pendek supaya match terpanjang dulu.
    alt = "|".join(re.escape(r) for r in _KNOWN_NOUN_ROOTS)
    pattern = re.compile(rf"\b(di|ke)({alt})\b")
    return pattern.sub(r"\1 \2", t)


def _find_noun_for_demonstrative(query_lc: str) -> tuple[str, str] | None:
    """Cari (noun, demonstrative) pertama yang muncul di query.

    Strategi: regex `(\\w+)\\s+(demonstrative)` — capture token sebelum
    demonstrative sebagai nomina kandidat. Bila tidak ada token, return None.
    """
    alt = "|".join(re.escape(d) for d in _DEMONSTRATIVES)
    pattern = re.compile(rf"\b([a-zA-ZÀ-ÿ]+)\s+({alt})\b")
    m = pattern.search(query_lc)
    if not m:
        return None
    noun = m.group(1).strip()
    demonstrative = m.group(2).strip()
    # Filter stopword yang bukan nomina (preposisi/konjungsi yang nyasar).
    stop = {"di", "ke", "dari", "yang", "dan", "atau", "untuk", "dengan", "pada", "oleh"}
    if noun in stop:
        return None
    return noun, demonstrative


def _lookup_referents(noun: str) -> list[str] | None:
    """Cari list referent untuk nomina via kamus. Strict dictionary-only —
    kembalikan ``None`` bila nomina tidak dikenal supaya pre-check
    konservatif (no-op untuk kasus tak terduga, tidak menggembungkan UQ).
    """
    for keys, refs in _REFERENT_DICTIONARY:
        for k in keys:
            if k == noun or (len(noun) >= 5 and noun.startswith(k)):
                return list(refs)
    return None


def _has_antecedent(noun: str, working_memory_text: str) -> bool:
    """Cek apakah ``noun`` (atau varian sederhana) muncul di working memory.

    Heuristik konservatif (substring lowercase + cek stem 4-char). Tidak
    pakai embedding similarity supaya deterministik dan cheap.
    """
    if not working_memory_text:
        return False
    wm_lc = working_memory_text.lower()
    if noun in wm_lc:
        return True
    # Stem ringan: ambil 4 huruf pertama nomina sebagai prefix match.
    if len(noun) >= 5 and noun[:4] in wm_lc:
        return True
    return False


def detect_dangling_reference(
    current_query: str,
    working_memory_text: str,
    episodic_memory_text: str = "",
) -> DanglingDetection | None:
    """Deteksi dangling reference di current_query.

    Args:
        current_query: pertanyaan user saat ini (raw).
        working_memory_text: gabungan teks turn-turn working memory
            sebelumnya (sudah di-format string). Kosong bila turn pertama.
        episodic_memory_text: gabungan teks ringkasan episodic memory yang
            relevan (sudah di-format string). Kosong bila tidak ada.

    Returns:
        ``DanglingDetection`` bila ditemukan demonstrative yang antecedent-nya
        tidak ada di working_memory maupun episodic_memory, **dan** nomina
        yang dimodifikasi terdaftar di kamus referent. ``None`` di luar itu.
    """
    if not current_query or not current_query.strip():
        return None
    normalized = _normalize_query(current_query)
    found = _find_noun_for_demonstrative(normalized)
    if found is None:
        return None
    noun, demonstrative = found
    combined_context = f"{working_memory_text}\n{episodic_memory_text}"
    if _has_antecedent(noun, combined_context):
        return None
    referents = _lookup_referents(noun)
    if referents is None:
        # Nomina tidak dikenal — pre-check sengaja no-op (strict whitelist).
        return None
    return DanglingDetection(
        noun=noun,
        demonstrative=demonstrative,
        referents=tuple(referents),
    )


def synthesize_rewrites_for_dangling(
    current_query: str,
    detection: DanglingDetection,
    m_total: int,
) -> list[str]:
    """Sintesis M rewrite mekanis untuk current_query bila terdeteksi dangling.

    Strategi: substitusi frase ``[di|ke|pada]? {noun} {demonstrative}`` di
    current_query dengan tiap referent dari ``detection.referents`` secara
    round-robin sampai panjang ``m_total``. Hasilnya adalah list teks yang
    dijamin divergen antar interpretasi (mis. "berapa jumlah per provinsi?",
    "berapa jumlah per kota/kabupaten?", dst).

    Dipakai oleh Stage 1 UQ rewriter untuk *grounding* M-sampling pada ruang
    interpretasi linguistik yang sudah dienumerasi pre-check, sehingga
    embedding + clustering downstream menghasilkan H_norm yang mencerminkan
    ambiguitas linguistik sesungguhnya (bukan modus posterior LLM yang
    under-explores karena mode collapse pada gpt-4o-mini @ T=1.0).

    Args:
        current_query: pertanyaan asli (raw, sebelum normalisasi).
        detection: hasil ``detect_dangling_reference``.
        m_total: jumlah sample M yang ingin dibangkitkan.

    Returns:
        List string panjang ``m_total``. Bila substitusi gagal (pola tak
        ditemukan), fallback prepend referent ke current_query.
    """
    if m_total <= 0 or not detection.referents:
        return []
    refs = list(detection.referents)
    # Pola substitusi — tangkap preposisi opsional + noun + demonstrative.
    # Bekerja di atas current_query sudah dinormalisasi (lowercase + di/ke
    # split untuk noun yang dikenal).
    normalized_query = _normalize_query(current_query)
    pattern = re.compile(
        rf"(?:\b(?:di|ke|pada)\s+)?\b{re.escape(detection.noun)}\s+"
        rf"{re.escape(detection.demonstrative)}\b"
    )
    out: list[str] = []
    for i in range(m_total):
        ref = refs[i % len(refs)]
        if pattern.search(normalized_query):
            rewritten = pattern.sub(ref, normalized_query, count=1)
        else:
            # Fallback: prepend referent supaya tetap divergen.
            rewritten = f"{normalized_query.rstrip(' ?.')} ({ref})?"
        # Capitalize awal kalimat.
        rewritten = rewritten.strip()
        if rewritten:
            rewritten = rewritten[0].upper() + rewritten[1:]
        out.append(rewritten)
    return out


def build_stratified_directive(
    detection: DanglingDetection,
    sample_index: int,
    m_total: int,
) -> str:
    """Bangun direktif eksplorasi per sample untuk stratified sampling.

    Direktif ini di-append ke user_prompt sample-ke-``sample_index`` supaya
    sample tersebut menargetkan interpretasi referent-ke-(sample_index mod K)
    dari ``detection.referents``. Hasilnya M sample meng-cover seluruh
    ruang referent secara round-robin (mengurangi mode collapse).

    Args:
        detection: hasil deteksi dangling reference.
        sample_index: indeks sample saat ini (0-based).
        m_total: jumlah total sample M (tidak dipakai untuk index, hanya
            untuk konteks penalaran prompt).

    Returns:
        String direktif berbahasa Indonesia siap di-append ke user_prompt.
        Konten meng-frame instruksi sebagai bagian dari Prinsip C/E supaya
        konsisten dengan REWRITE_UQ_SYSTEM_PROMPT.
    """
    refs = list(detection.referents)
    if not refs:
        return ""
    target_idx = sample_index % len(refs)
    target_referent = refs[target_idx]
    enumeration_lines = [
        f"  I{i + 1}. {r}" for i, r in enumerate(refs)
    ]
    enumeration_block = "\n".join(enumeration_lines)
    return (
        "\n\n"
        "[Catatan eksplorasi untuk panggilan ini]\n"
        f"Frase '{detection.noun} {detection.demonstrative}' di [current_query] "
        "merupakan referensi yang antecedent-nya tidak ada di [working_memory] "
        "maupun [episodic_memory] (Prinsip E — dangling reference).\n"
        "Ruang interpretasi plausible:\n"
        f"{enumeration_block}\n"
        f"Untuk panggilan independen ini, gunakan interpretasi I{target_idx + 1} "
        f"= \"{target_referent}\". Tulis ulang [current_query] menjadi pertanyaan "
        "mandiri yang me-resolusi demonstrative sesuai interpretasi tersebut. "
        "Jangan men-drop demonstrative tanpa resolusi."
    )
