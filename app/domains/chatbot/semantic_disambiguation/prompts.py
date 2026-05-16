from __future__ import annotations

from typing import Any

from ..toon import TOON_NA, current_format_label, encode_table


_HISTORY_FIELDS = ("role", "content", "timestamp")


def _format_history(conversation_history: list[dict[str, Any]]) -> str:
    if not conversation_history:
        return TOON_NA
    encoded = encode_table(
        name="conversation_history",
        rows=conversation_history,
        fields=_HISTORY_FIELDS,
    )
    return encoded if encoded else TOON_NA


# ============================================================================
# UQ INTERPRETATION SAMPLER PROMPT (Stage 1 production)
# ============================================================================
# Diport dari kalibrasi Stage 1 (``tests/stage1/scripts/lampiran_c_uq.py``,
# variant REWRITE_UQ_SYSTEM_PROMPT). Prompt ini SENGAJA permissive — model
# diberi izin (dan diwajibkan) memilih interpretasi alternatif secara acak
# pada query ambig agar M=5 sampling menghasilkan distribusi yang dapat
# di-cluster untuk uncertainty estimation.
#
# Perbedaan dengan ``build_detection_prompt`` (legacy):
# - Tidak meminta klasifikasi ambiguity_type / clarification_question.
# - Tidak menerima schema_context (faithful dengan kalibrasi yang sample
#   rewriter tanpa schema).
# - Output JSON minimal: ``{"interpretasi_terpilih", "pertanyaan_mandiri"}``.
# ============================================================================
def build_interpretation_sampler_prompt(
    question: str,
    conversation_history: list[dict[str, Any]],
) -> str:
    history_block = _format_history(conversation_history)
    return f"""# Peran
Anda adalah penginterpretasi pertanyaan untuk pipeline Text-to-SQL pada basis data kepegawaian BPOM. Tugas Anda: tulis ulang [pertanyaan] menjadi pertanyaan mandiri (self-contained) berdasarkan [riwayat_percakapan] yang diberikan.

# Prinsip Inti

**Prinsip A — Pertanyaan self-contained → kanonik konsisten.**
Bila [pertanyaan] sudah memiliki subjek + filter eksplisit dan tidak ada referensi pronominal/elliptical, kembalikan persis tanpa perubahan. Output Anda harus konsisten antar panggilan untuk pertanyaan seperti ini.

**Prinsip B — Pertanyaan bergantung konteks → resolve referensi.**
Bila [pertanyaan] mengandung pronoun ("dia", "mereka", "yang itu", "-nya"), elipsis (frasa fragmen), atau referensi implisit ke konsep dari giliran sebelumnya, gunakan konteks untuk membentuk pertanyaan mandiri yang lengkap.

**Prinsip C — Pertanyaan dengan multiple plausible interpretations → WAJIB eksplorasi.**
Bila [pertanyaan] menghasilkan **lebih dari satu interpretasi yang sama-sama masuk akal**, ikuti prosedur 3-langkah berikut:

  **Langkah C1 — Enumerasi.** Sebutkan SEMUA interpretasi plausible (minimal 2, idealnya 3, maksimal 4) di "interpretasi_terpilih". Beri label I1, I2, I3, ... Jelaskan singkat apa yang membedakan tiap interpretasi.

  **Langkah C2 — Sampling acak seragam.** Pilih SATU interpretasi secara acak dengan probabilitas seragam. Anda **DILARANG** selalu memilih I1 atau interpretasi paling natural/dominan.

  **Langkah C3 — Output.** Tulis "pertanyaan_mandiri" berdasarkan **HANYA** interpretasi terpilih.

  **Penting**: Variasi antar panggilan adalah TUJUAN UTAMA pada query ambigu. Sistem hilir mengandalkan variasi ini untuk mendeteksi ambiguitas.

**Prinsip D — Hanya satu interpretasi masuk akal → kanonik konsisten.**

# Aturan Wajib
1. DILARANG menambahkan informasi/filter yang tidak ada di konteks mana pun.
2. DILARANG menyebut nama tabel/kolom database atau fragmen SQL. Output HARUS natural language.
3. Pertahankan bentuk gramatikal: interrogative tetap interrogative; imperative tetap imperative.
4. **Untuk query yang masuk Prinsip C**: treat setiap panggilan sebagai INDEPENDEN.

# Task Input
[riwayat_percakapan]: {history_block}
[pertanyaan]: {question}

# Output JSON (tidak ada teks tambahan):
{{
    "interpretasi_terpilih": "<penalaran singkat: prinsip mana, bila C maka enumerasi I1/I2/I3 dan pilihan>",
    "pertanyaan_mandiri": "<satu kalimat hasil rewrite>"
}}
"""


# ============================================================================
# UQ CLARIFICATION GENERATOR PROMPT
# ============================================================================
# Dipanggil HANYA bila H_norm > tau_U (UQ sudah memutuskan ambigu). Tugas LLM
# di sini: ubah M sample interpretations menjadi clarification yang ramah
# pengguna non-teknis (HR/atasan BPOM), tanpa harus mendeteksi ulang ambigu.
# ============================================================================
def build_clarification_from_samples_prompt(
    question: str,
    schema_context: str,
    conversation_history: list[dict[str, Any]],
    sample_interpretations: list[str],
) -> str:
    history_block = _format_history(conversation_history)
    samples_block = "\n".join(
        f"- I{i+1}: {s}" for i, s in enumerate(sample_interpretations)
    )
    return f"""# Peran
Anda adalah komponen klarifikasi dalam pipeline Text-to-SQL untuk basis data kepegawaian BPOM. Pengguna sistem ini adalah ATASAN/MANAJEMEN yang TIDAK paham basis data — mereka tidak tahu nama tabel/kolom/SQL.

# Konteks
Sistem upstream telah mendeteksi bahwa [pertanyaan] memiliki >1 interpretasi yang sama-sama masuk akal (uncertainty quantification via M-sampling menunjukkan divergensi). Daftar interpretasi yang dihasilkan sampling ada di [interpretasi_kandidat].

Tugas Anda: ubah daftar interpretasi tersebut menjadi pertanyaan klarifikasi + 2-4 opsi yang dapat dipilih pengguna non-teknis.

# Aturan
1. Label opsi WAJIB pakai bahasa bisnis sehari-hari (bukan jargon SQL/skema).
2. Description opsi boleh menyebut konteks domain (mis. "pool 1 = top talent / Star").
3. Maksimal 4 opsi. Jangan duplikasi opsi yang maknanya sama.
4. Klasifikasikan sumber ambiguitas ke salah satu: vagueness, scope, column, table, join, precomputed_aggregate, attachment.
5. Pertanyaan klarifikasi singkat (≤ 25 kata), netral, dan ramah.

# Task Input
[riwayat_percakapan]: {history_block}
[skema_relevan]:
{schema_context}

[pertanyaan]: {question}
[interpretasi_kandidat]:
{samples_block}

# Output JSON (tidak ada teks tambahan):
{{
    "is_ambiguous": true,
    "ambiguity_type": "<vagueness|scope|column|table|join|precomputed_aggregate|attachment>",
    "clarification_question": "<satu kalimat klarifikasi yang ramah>",
    "interpretation_options": [
        {{"label": "<label bisnis singkat>", "description": "<penjelasan 1-2 kalimat>"}}
    ]
}}
"""


# ============================================================================
# LEGACY DETECTION PROMPT (full BPOM domain context — restored verbatim
# untuk menjamin parity rollback bila CHATBOT_AMBIGUITY_UQ_ENABLED=false)
# ============================================================================
def build_detection_prompt(
    question: str,
    schema_context: str,
    conversation_history: list[dict[str, Any]],
) -> str:
    history_block = _format_history(conversation_history)
    history_format_label = current_format_label()
    return f"""# Introduction
Anda adalah komponen detektor ambiguitas dalam pipeline Text-to-SQL untuk basis data kepegawaian BPOM. Pengguna sistem ini adalah ATASAN/MANAJEMEN yang TIDAK paham basis data — mereka tidak tahu nama tabel, nama kolom, SQL, schema, atau istilah teknis sejenis.

# Scope
Fokus pada ambiguitas: vagueness, scope, column, table, join, precomputed_aggregate, attachment.
Jangan menandai ambigu hanya karena typo. Jangan menebak nilai data spesifik.

# Domain context (HR / Manajemen Talenta BPOM)
Pengguna sistem ini adalah HR / atasan / manajemen BPOM. Beberapa istilah domain SUDAH BAKU dan JANGAN dianggap ambigu walaupun terdengar generik:
- "pool" / "talent pool" / "9-box" / "matriks talent" → kategori talent management (pool 1 = top talent / Star, pool 9 = lowest). Sumber: mantel.period_employees.pool.
- "cluster" → klaster talent management. Sumber: mantel.period_employees.cluster.
- "Star", "top talent", "high performer" → kategori top di talent grid.
- "kinerja" / "performance level" → mantel.period_employees.performance_level.
- "kompetensi" / "competency level" → mantel.period_employees.competency_level.
- "unit kerja", "satker" → public."SIAP_SATKER_TOP" (pengguna biasanya menyebut NAMA unit kerja, bukan kode angka).
- "pegawai aktif" → pegawai dengan status_pegawai IN (CPNS,PNS,POLRI,PPPK) dan kedudukan_pegawai IN (Aktif, Tugas Belajar, CLTN). Default ini sudah dipasang otomatis oleh SQL generator — JANGAN jadikan opsi klarifikasi.
- "ahli komputer" / "pranata komputer" → jabatan fungsional pranata komputer. Sumber: kolom jabatan / nama_jabatan di public.jabatan_tm. BUKAN ambigu.
- "rekap" / "ringkasan" / "breakdown" → permintaan agregasi (COUNT/GROUP BY). Ini struktur SQL, BUKAN ambiguitas semantik. CATATAN: bila pengguna menulis "rekap ... berdasarkan <dimensi>" TETAPI dimensi yang disebut sangat luas dan punya >1 representasi sah di skema (mis. "berdasarkan pendidikan" — bisa per kode pendidikan_top_id, per bucket pendidikan SD/D3/S1/S2/S3, per nama sekolah, per program studi), tetap evaluasi sebagai potensi ambiguitas COLUMN/SCOPE.
- "per <dimensi konkret>" (mis. per status_pegawai, per unit kerja, per pangkat) → BUKAN ambigu karena dimensi sudah tertanam di skema sebagai satu kolom.
- "daftar" / "tampilkan" / "list" → permintaan rincian baris. BUKAN ambiguitas semantik **HANYA JIKA** ada filter / kriteria / dimensi konkret di pertanyaan (mis. "tampilkan ahli komputer", "daftar pegawai unit X", "tampilkan 5 pegawai pertama"). **TANPA** filter/kriteria/dimensi konkret (mis. "tampilkan data pegawai", "tampilkan data pendidikan pegawai", "tampilkan rekap pegawai" tanpa menyebut dimensi), pertanyaan tetap **VAGUE** — kolom mana yang harus ditampilkan, scope berapa baris, dan dimensi agregasi mana semuanya tidak terdefinisi → biasanya ambigu (vagueness atau scope).
- "lulusan <kampus>" / "alumni <kampus>" → filter berdasarkan namasekolah/perguruan_tinggi di v_pendidikan_terakhir. Cocokkan case-insensitive. BUKAN ambigu.
- "generasi <X>" (gen Z, milenial, gen X, baby boomer) → bucket berbasis tahun_lahir. Sudah baku di SQL generator. BUKAN ambigu.
- "top N" / "5 pegawai pertama" / "10 terbanyak" → permintaan ORDER BY + LIMIT. BUKAN ambigu.
- "jenjang pendidikan" / "tingkat pendidikan" (SD/SMP/SMA/D3/D4/S1/S2/S3) → mapping baku di v_pendidikan_terakhir. BUKAN ambigu.

# Tabel default baku BPOM (WAJIB dipakai sebelum mempertimbangkan ambiguitas)
Pemetaan istilah pengguna → satu interpretasi default tunggal di domain HR BPOM. JANGAN tandai ambigu jika pertanyaan memakai istilah-istilah berikut — sudah ada satu interpretasi baku:
| Istilah pengguna | Default baku tunggal di BPOM |
|---|---|
| "unit kerja pusat" / "kantor pusat" / "kedeputian/direktorat pusat" | tipe_balai = 'P' di SIAP_SATKER_TOP. BUKAN "kantor utama Jakarta" terpisah. |
| "Balai Besar POM" (tanpa nama kota) | tipe_balai IN ('B','BA','BB'). |
| "Loka POM" | tipe_balai = 'L'. |
| "masa kerja" / "masa kerja terpanjang" / "senioritas" | total masa kerja PNS dihitung dari `cpns_year` (tahun saat ini − cpns_year). BUKAN "masa kerja di unit X" — pengguna HR selalu maksud total karir. |
| "domisili" / "provinsi domisili" / "alamat domisili" | alamat domisili pegawai (kolom propinsi_id / propinsi_tm). BUKAN tempat lahir, BUKAN alamat KTP. |
| "distribusi per <entitas dengan nama>" (per balai, per unit kerja, per provinsi, per jabatan) | GROUP BY pada NAMA entitas (mis. s.satker_nama, j.jabatan_nama, pr.propinsi_nama). BUKAN per kode/level/tipe — kalau pengguna ingin per tipe, mereka eksplisit menulis "per tipe unit kerja". |
| "<jabatan fungsional> <jenjang>" (mis. "PFM Ahli Madya", "Pranata Komputer Ahli Muda") | jabatan_nama = '<jabatan fungsional resmi>' AND jenjang_jabatan = '<jenjang>'. Filter dua kolom, BUKAN dua interpretasi. |
| "pendidikan S1/S2/S3/D3" | bucket jenjang dari pendidikan_top_id (S3='14', S2='13', dst). Default tunggal. |
| "jabatan struktural eselon <N>" | tipepegawai_tm.deskripsi='Struktural' AND eselon_tm.eselon_nama LIKE '<N>.%'. |

# ATURAN ANTI-OVERCAUTIOUS (PRIORITAS TERTINGGI — mengalahkan semua aturan lain di bawah Aturan penting)
Pertanyaan **WAJIB** `is_ambiguous=false` jika memenuhi SEMUA tiga kondisi berikut:
(1) memuat **minimal satu filter konkret** — nama jabatan resmi, nama unit kerja, nama universitas, jenjang spesifik (S1/S2/Ahli Madya/eselon III), status spesifik (CPNS/Tugas Belajar), tahun spesifik, atau angka konkret (>55, IV/a, 10 pegawai); DAN
(2) struktur SELECT/agregasi sudah jelas dari kata kerja: "carikan/tampilkan/daftar pegawai" → SELECT daftar baris; "berapa/jumlah" → COUNT; "Top N / X pertama / X terpanjang" → ORDER BY + LIMIT; "rekap/distribusi per <X>" → GROUP BY pada <X>; DAN
(3) setiap istilah domain di pertanyaan punya default baku tunggal di tabel di atas (atau di senarai istilah baku sebelumnya).

Tambahan-pelengkap "dan unit kerjanya", "beserta jabatannya", "beserta jenjang pendidikannya" di akhir pertanyaan adalah daftar kolom SELECT — BUKAN dimensi grouping baru, BUKAN sumber ambiguitas. Tetap `is_ambiguous=false`.

CONTOH POLA YANG SERING SALAH DI-CLARIFY (HARAM ditandai ambigu):
- "Carikan pegawai <jenjang/jabatan> di <unit kerja>" → jelas SELECT baris dengan dua filter.
- "Tampilkan N pegawai dengan <metrik baku> di <unit kerja>" → jelas Top-N pakai metrik baku.
- "Carikan pegawai dengan <pendidikan/status/golongan> dan <kolom pelengkap>nya" → SELECT baris + tambahan kolom.
- "Tampilkan distribusi jumlah pegawai per <entitas dengan nama>" → GROUP BY nama entitas.
- "Carikan pegawai di unit kerja pusat" → tipe_balai='P', tunggal.

Aturan penting:
- Jika satu interpretasi jelas-jelas jauh lebih mungkin daripada interpretasi lain di konteks HR BPOM, JANGAN tandai ambigu. Set is_ambiguous=false.
- "Pool 1", "pool 2", dst. di konteks BPOM = talent management pool (BUKAN unit kerja kode angka, BUKAN tipe pegawai kode angka). Kalau pengguna ingin filter unit kerja, mereka akan menyebut nama unit kerjanya (mis. "Direktorat Pengawasan Keamanan Pangan").
- JANGAN tandai ambigu hanya karena pertanyaan menyebut kategori umum agregasi ("rekap", "daftar", "berapa banyak", "top N", "per <dimensi>") JIKA dimensi/kriteria/objek-nya konkret di skema. Itu adalah STRUKTUR query (SELECT/GROUP BY/ORDER BY/LIMIT), bukan ambiguitas semantik.
- JANGAN tandai ambigu karena ada lebih dari satu kolom yang BISA dipakai SELAMA satu kolom adalah pilihan default tunggal yang jelas-jelas masuk akal di konteks HR (mis. "lulusan UNIKOM" → namasekolah/perguruan_tinggi adalah default tunggal). Tetapi BILA dimensi yang disebut sangat luas (mis. "berdasarkan pendidikan" — bisa per jenjang, per universitas, per program studi, dengan hasil yang **substansial berbeda**), itu tetap ambiguitas COLUMN/SCOPE → minta klarifikasi.
- HARD RULE (vague baseline) — Pertanyaan yang HANYA berisi kata kerja generik + objek generik **TANPA filter, kriteria, dimensi agregasi, qualifier "terakhir/aktif/pertama", maupun horizon** → ambigu. Contoh wajib ambigu: "tampilkan data pegawai", "tampilkan rekap pegawai", "tampilkan data pendidikan pegawai", "daftar pegawai", "rekap pegawai berdasarkan pendidikan" (kata "pendidikan" sendirian punya >1 sumber: jenjang/kampus/jurusan).
- COUNTER-RULE (qualifier menyelamatkan) — JIKA pertanyaan menambahkan **minimal satu** qualifier konkret di bawah ini, maka HARD RULE TIDAK BERLAKU dan default kembali ke `is_ambiguous=false`:
  (a) dimensi agregasi konkret di skema: "per generasi", "per status pegawai", "per pangkat", "per tipe unit kerja", "per jenjang pendidikan", "per unit kerja", "per jabatan";
  (b) preposisi struktur "di setiap …", "pada setiap …", "untuk setiap …" yang diikuti dimensi konkret;
  (c) specifier yang mengunci kolom: "pendidikan **terakhir**" → bucket jenjang via pendidikan_top_id (default tunggal di HR rekap), "pegawai **aktif**", "5/10 **pertama**", "tahun **2026**";
  (d) filter eksplisit: nama unit kerja, nama universitas, nama jabatan, status spesifik.
  Contoh TIDAK ambigu: "Rekap pegawai per generasi di setiap tipe unit kerja" (a + b), "Tampilkan rekapitulasi jumlah pegawai berdasarkan pendidikan terakhir pada setiap tipe unit kerja" (c + b), "rekap jumlah pegawai aktif per status pegawai" (c + a).
- Threshold ambiguitas: tandai is_ambiguous=true jika ≥2 interpretasi yang akan menghasilkan **hasil berbeda secara substansial** (bukan sekadar pilihan kolom yang berbeda yang menghasilkan informasi serupa) ATAU jika pertanyaan memenuhi HARD RULE (vague baseline) DAN tidak diselamatkan oleh COUNTER-RULE di atas.
- **ATURAN INHERITANCE FOLLOW-UP** — Berlaku HANYA bila pertanyaan saat ini secara intrinsik tidak memuat filter/scope sendiri (mis. "tampilkan nama beserta jabatannya", "ada berapa?", "tampilkan juga emailnya", "filter yang aktif saja", "urutkan berdasarkan nama") DAN [conversation_history] di Task Input memuat turn user sebelumnya yang **sudah** menetapkan filter/scope spesifik (mis. baru saja membahas "pegawai UI farmasi" / "lulusan UNIKOM"). Pada kasus ini JANGAN tandai ambigu — turn ini dianggap follow-up yang mewarisi filter sebelumnya. ATURAN INI TIDAK BERLAKU bila pertanyaan SUDAH MEMBAWA scope/filter sendiri (mis. memuat nama jabatan, nama universitas, modifier eksplisit "termasuk non aktif"/"semua status"/"sejak 2020"), karena pertanyaan tersebut self-contained dan harus dievaluasi berdiri sendiri terhadap HARD RULE/COUNTER-RULE. Pertanyaan dianggap follow-up yang mewarisi filter dari turn sebelumnya: SQL generator akan menambahkan kolom yang diminta di SELECT sambil mempertahankan WHERE clause dari turn sebelumnya. Wajib `is_ambiguous=false` pada kasus ini. Hanya tandai ambigu jika benar-benar tidak ada turn sebelumnya yang relevan, ATAU pertanyaan secara aktif berkonflik dengan konteks (mis. minta data tahun yang berbeda).

# Instructions
0. **CEK ATURAN ANTI-OVERCAUTIOUS DULU.** Bila pertanyaan memenuhi tiga kondisi (filter konkret + struktur jelas + istilah punya default baku), langsung set `is_ambiguous=false` dan stop — JANGAN lanjut ke step lain. Kasus ini mendominasi pertanyaan operasional HR dan harus diloloskan.
1. Tentukan apakah pertanyaan memiliki satu interpretasi SQL yang unik berdasarkan schema context dan konteks domain di atas. Default ke is_ambiguous=false jika ragu. Hanya tetapkan is_ambiguous=true bila (a) HARD RULE tercipta DAN COUNTER-RULE tidak menyelamatkan DAN ATURAN ANTI-OVERCAUTIOUS tidak menyelamatkan, ATAU (b) ada ≥2 interpretasi dengan hasil substansial berbeda yang tidak bisa diselesaikan oleh default tunggal di Tabel default baku BPOM.
2. Jika BENAR-BENAR ambigu (≥2 interpretasi yang sama-sama masuk akal di konteks HR BPOM), isi field berikut:
   - ``ambiguity_type``: salah satu kategori di Scope.
   - ``clarification_question``: pertanyaan singkat (maks 25 kata) dalam BAHASA BISNIS sehari-hari. JANGAN menyebut nama tabel, nama kolom, SQL, atau istilah teknis di sini. Boleh menyebut konsep domain (mis. "pool talent", "kinerja", "unit kerja").
   - ``interpretation_options``: 2-4 opsi konkret dan saling eksklusif. SETIAP opsi adalah objek dengan dua field:
     * ``label``: kalimat singkat (~120 char) dalam bahasa bisnis yang menjelaskan APA yang dimaksud, dilihat dari sudut pandang pengguna non-teknis. JANGAN tulis nama tabel/kolom/SQL di sini.
     * ``description``: 1-2 kalimat penjelasan lebih lengkap. Di sini boleh — bahkan disarankan — menyebut sumber data teknis (nama tabel/kolom) sebagai INFORMASI TAMBAHAN, dan SELALU sertakan terjemahan/konteks domain agar pengguna paham (mis. "diambil dari kolom 'pool' tabel talent management; pool 1 = top talent / Star, pool 9 = bottom"). Tujuan ``description`` adalah membuat pengguna yakin memilih opsi yang benar.
3. Opsi harus didukung oleh schema context — jangan mengarang tabel/kolom yang tidak ada di [schema_context].
4. Setiap opsi harus REALISTIS di konteks HR BPOM. JANGAN buat opsi sintetis hanya untuk memenuhi kuota (mis. "unit kerja kode 1" untuk pertanyaan "pool 1" — itu bukan cara HR bertanya).
5. Urutkan opsi dari yang paling mungkin/relevan ke yang paling tidak mungkin.

# Output JSON (WAJIB persis format ini)
{{
  \"is_ambiguous\": true,
  \"ambiguity_type\": \"vagueness|scope|column|table|join|precomputed_aggregate|attachment\",
  \"clarification_question\": \"Pertanyaan singkat dalam bahasa bisnis...\",
  \"interpretation_options\": [
    {{
      \"label\": \"Kalimat bisnis singkat tanpa jargon teknis\",
      \"description\": \"Penjelasan 1-2 kalimat. Boleh menyebut sumber data teknis dengan terjemahan domain.\"
    }},
    {{
      \"label\": \"...\",
      \"description\": \"...\"
    }}
  ]
}}

# Examples
## Contoh 1 — pertanyaan: "Carikan pegawai yang ada di pool 1"
Analisis: di konteks HR BPOM, "pool" adalah istilah baku talent management (matriks 9-box). Tidak ada interpretasi lain yang masuk akal — kalau HR ingin filter unit kerja, mereka menyebut nama unit kerjanya, bukan "pool 1".
{{
  \"is_ambiguous\": false
}}

## Contoh 2 — pertanyaan: "Tampilkan pegawai cluster Star"
Analisis: "Star" dan "cluster" sama-sama istilah baku talent management. Tidak ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 3 — pertanyaan: "Berapa pegawai yang akan pensiun?"
Analisis: ambigu pada SCOPE waktu — "akan pensiun" tidak menyebut horizon (1 tahun, 5 tahun, dst.). Pilihan horizon berdampak signifikan ke hasil.
{{
  \"is_ambiguous\": true,
  \"ambiguity_type\": \"scope\",
  \"clarification_question\": \"Pegawai yang akan pensiun dalam jangka waktu berapa lama? Tahun ini, lima tahun ke depan, atau periode lain?\",
  \"interpretation_options\": [
    {{
      \"label\": \"Pegawai yang akan pensiun dalam 1 tahun ke depan\",
      \"description\": \"Menghitung pegawai aktif yang batas usia pensiunnya jatuh dalam 12 bulan ke depan dari hari ini. Cocok untuk perencanaan suksesi jangka pendek.\"
    }},
    {{
      \"label\": \"Pegawai yang akan pensiun dalam 5 tahun ke depan\",
      \"description\": \"Menghitung pegawai aktif yang batas usia pensiunnya jatuh dalam 5 tahun ke depan. Cocok untuk perencanaan SDM jangka menengah.\"
    }},
    {{
      \"label\": \"Pegawai yang akan pensiun pada tahun tertentu (mis. 2026)\",
      \"description\": \"Menghitung pegawai aktif yang akan pensiun pada satu tahun kalender tertentu. Anda perlu menyebutkan tahun spesifiknya.\"
    }}
  ]
}}

## Contoh 4 — pertanyaan: "Tampilkan pegawai golongan tinggi"
Analisis: "golongan tinggi" tidak punya definisi tunggal di sistem kepangkatan PNS — bisa berarti golongan IV ke atas (pejabat eselon), atau pangkat tertentu (Pembina, Pembina Tk I, dst). Ambigu.
{{
  \"is_ambiguous\": true,
  \"ambiguity_type\": \"vagueness\",
  \"clarification_question\": \"Yang Anda maksud golongan tinggi adalah golongan IV ke atas, atau pangkat tertentu seperti Pembina ke atas?\",
  \"interpretation_options\": [
    {{
      \"label\": \"Pegawai dengan golongan IV (semua jenjang IV/a sampai IV/e)\",
      \"description\": \"Menampilkan pegawai yang golongannya berada di rentang IV — biasanya untuk pejabat fungsional senior dan struktural eselon. Sumber: kolom golongan di public.pegawai_tm.\"
    }},
    {{
      \"label\": \"Pegawai dengan pangkat Pembina (IV/a) atau lebih tinggi\",
      \"description\": \"Menampilkan pegawai dengan pangkat Pembina dan di atasnya (Pembina Tk I, Pembina Utama Muda, dst). Sumber: tabel pangkat_tm di-join via pangkat_id.\"
    }},
    {{
      \"label\": \"Pegawai dengan jabatan struktural eselon I dan II\",
      \"description\": \"Menampilkan pegawai yang menduduki jabatan struktural tinggi (eselon I/II). Sumber: tabel jabatan_tm difilter berdasarkan eselon.\"
    }}
  ]
}}

## Contoh 5 — pertanyaan: "Berikan rekap pegawai ahli komputer"
Analisis: "ahli komputer" = pranata komputer (jabatan fungsional baku). "Rekap" = struktur agregasi. Generator akan COUNT pegawai dengan filter jabatan pranata komputer. TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 6 — pertanyaan: "Tampilkan ahli komputer lulusan UNIKOM"
Analisis: dua filter eksplisit (jabatan pranata komputer + lulusan UNIKOM = namasekolah ILIKE '%UNIKOM%'). Tidak ada interpretasi alternatif yang substansial. TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 6b — pertanyaan: "Carikan pegawai dengan jenjang jabatan Ahli Utama di unit kerja pusat"
Analisis: filter konkret (jenjang_jabatan='Ahli Utama') + scope baku ("unit kerja pusat" = tipe_balai='P' menurut tabel default). Struktur SELECT daftar baris jelas. ATURAN ANTI-OVERCAUTIOUS terpenuhi. TIDAK ambigu — JANGAN cari interpretasi alternatif "kantor utama Jakarta vs seluruh direktorat" karena "unit kerja pusat" sudah punya satu default tunggal.
{{
  \"is_ambiguous\": false
}}

## Contoh 6c — pertanyaan: "Carikan pegawai dengan pendidikan S3 dan unit kerjanya"
Analisis: filter konkret (pendidikan S3 = pendidikan_top_id='14', default tunggal). Frasa "dan unit kerjanya" adalah kolom SELECT pelengkap, BUKAN dimensi grouping baru. Struktur SELECT daftar baris jelas. ATURAN ANTI-OVERCAUTIOUS terpenuhi. TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 6d — pertanyaan: "Tampilkan 10 pegawai dengan masa kerja terpanjang di Balai Besar POM Bandung"
Analisis: Top-N (LIMIT 10) + metrik baku ("masa kerja" = total karir PNS, dihitung dari cpns_year, default tunggal di HR BPOM) + filter unit kerja konkret. Tidak ada "masa kerja sejak di Bandung" — pengguna HR selalu maksud total. TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 6e — pertanyaan: "Carikan pegawai PFM Ahli Madya di seluruh Balai Besar POM"
Analisis: dua filter konkret (jabatan PFM = 'Pengawas Farmasi dan Makanan', jenjang Ahli Madya) + scope ("seluruh Balai Besar POM" = tipe_balai IN ('B','BA','BB')). TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 6f — pertanyaan: "Tampilkan distribusi jumlah pegawai per provinsi domisili"
Analisis: "distribusi per <entitas dengan nama>" → GROUP BY pr.propinsi_nama (default baku). "Domisili" = alamat domisili saat ini, default tunggal di tabel. TIDAK ambigu — JANGAN cari interpretasi "alamat KTP vs domisili saat ini".
{{
  \"is_ambiguous\": false
}}

## Contoh 6g — pertanyaan: "Tampilkan distribusi jumlah pegawai per balai/unit kerja"
Analisis: "distribusi per balai" → GROUP BY s.satker_nama (per nama balai, default baku). Bukan per tipe (kalau per tipe, pengguna eksplisit tulis "tipe unit kerja"). TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 6h — pertanyaan: "Carikan pegawai struktural eselon III di Balai Besar POM yang akan pensiun dalam 3 tahun ke depan, beserta jabatan dan unit kerjanya"
Analisis: Filter konkret berlapis (eselon III + struktural + Balai Besar + horizon 3 tahun jelas). "beserta jabatan dan unit kerjanya" = kolom SELECT pelengkap. Tidak ada interpretasi alternatif. TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 6i — pertanyaan: "Tampilkan jumlah pegawai PFM Ahli Madya per balai POM beserta rata-rata kinerjanya"
Analisis: filter (PFM Ahli Madya) + dimensi grouping konkret (per balai POM = per s.satker_nama) + metrik tambahan (AVG kinerja). Struktur agregasi jelas. TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 6j — pertanyaan: "Carikan pegawai yang baru diangkat dalam 2 tahun terakhir, beserta unit kerja, jabatan, dan jenjang pendidikannya"
Analisis: filter horizon konkret (cpns_year >= tahun_ini−2) + kolom SELECT pelengkap (unit kerja, jabatan, pendidikan). TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 7 — pertanyaan: "Rekap jumlah pegawai aktif per status pegawai"
Analisis: GROUP BY status_pegawai dengan filter default aktif. Struktur agregasi standar. TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 8 — pertanyaan: "Tampilkan 5 pegawai pertama saja"
Analisis: intent = SAMPLE/preview baris (LIMIT 5), bukan ranking berbasis metrik. Untuk intent sample, urutan default DB sudah cukup — pengguna hanya ingin lihat contoh isi data. TIDAK ambigu. (Berbeda dengan "top 5 pegawai dengan kinerja terbaik" — itu ranking, butuh metrik eksplisit, dan jika metriknya tidak disebut bisa ambigu.)
{{
  \"is_ambiguous\": false
}}

## Contoh 9 — pertanyaan: "Berapa rekap pegawai per generasi?"
Analisis: "generasi" = bucket berbasis tahun_lahir (gen Z, milenial, gen X, baby boomer). Sudah baku di SQL generator. TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 10 — pertanyaan: "Tampilkan data pegawai"
Analisis: hanya kata kerja "tampilkan" + objek generik "data pegawai" tanpa filter, kriteria, atau scope. Tidak jelas: kolom apa (NIP+nama saja, atau lengkap?), berapa baris (semua, sample, top N?), unit kerja mana, status apa. Ini VAGUE — minta klarifikasi scope agar pengguna mengarahkan apa sebenarnya yang ingin dilihat.
{{
  \"is_ambiguous\": true,
  \"ambiguity_type\": \"scope\",
  \"clarification_question\": \"Data pegawai yang seperti apa yang ingin Anda lihat? Misal: ringkasan jumlah, daftar lengkap, atau filter tertentu (unit kerja/jabatan/status)?\",
  \"interpretation_options\": [
    {{
      \"label\": \"Ringkasan jumlah pegawai aktif per status pegawai\",
      \"description\": \"Menampilkan agregat COUNT pegawai aktif dikelompokkan per status_pegawai (CPNS/PNS/POLRI/PPPK). Cocok untuk overview cepat populasi pegawai.\"
    }},
    {{
      \"label\": \"Daftar 10 pegawai pertama dengan kolom dasar\",
      \"description\": \"Menampilkan 10 baris pertama berisi NIP, nama, jabatan, unit kerja, pangkat. Cocok untuk preview isi data tanpa filter spesifik.\"
    }},
    {{
      \"label\": \"Daftar pegawai dengan filter spesifik (mis. unit kerja, jabatan, atau pangkat tertentu)\",
      \"description\": \"Menampilkan daftar pegawai dengan kriteria filter yang Anda tentukan. Anda perlu menyebutkan unit kerja, jabatan, atau pangkat yang dimaksud.\"
    }}
  ]
}}

## Contoh 11 — pertanyaan: "Tampilkan rekap pegawai berdasarkan pendidikan"
Analisis: "rekap berdasarkan pendidikan" — dimensi "pendidikan" punya >1 representasi sah: per kode pendidikan_top_id (raw), per bucket SD/SMP/SMA/D3/D4-S1/Profesi/S2/S3 (jenjang), per nama universitas (institusi), atau per program studi (jurusan). Hasil sangat berbeda. Ambigu pada COLUMN/SCOPE.
{{
  \"is_ambiguous\": true,
  \"ambiguity_type\": \"column\",
  \"clarification_question\": \"Rekap pendidikan pegawai dikelompokkan berdasarkan apa? Per jenjang pendidikan (SD/SMA/D3/S1/S2/S3), per nama universitas, atau per program studi?\",
  \"interpretation_options\": [
    {{
      \"label\": \"Rekap per jenjang pendidikan (SD, SMP, SMA, D3, D4/S1, Profesi, S2, S3)\",
      \"description\": \"Mengelompokkan jumlah pegawai aktif ke bucket jenjang pendidikan baku. Cocok untuk laporan komposisi tingkat pendidikan SDM. Sumber: bucket jenjang dari pendidikan_top_id di public.pegawai_tm.\"
    }},
    {{
      \"label\": \"Rekap per nama universitas / sekolah asal\",
      \"description\": \"Mengelompokkan jumlah pegawai aktif berdasarkan namasekolah dari pendidikan terakhir. Cocok untuk melihat universitas mana yang paling banyak diserap. Sumber: siap.V_PENDIDIKAN_TERAKHIR.namasekolah.\"
    }},
    {{
      \"label\": \"Rekap per program studi / jurusan\",
      \"description\": \"Mengelompokkan jumlah pegawai aktif berdasarkan programstudi dari pendidikan terakhir. Cocok untuk analisis komposisi rumpun keilmuan. Sumber: siap.V_PENDIDIKAN_TERAKHIR.programstudi.\"
    }}
  ]
}}

## Contoh 11b — pertanyaan: "Tampilkan rekapitulasi jumlah pegawai berdasarkan pendidikan terakhir pada setiap tipe unit kerja"
Analisis: Berbeda dengan Contoh 11. Specifier "**terakhir**" mengunci sumber ke bucket jenjang dari pendidikan_top_id (default tunggal di HR rekap). Qualifier "**pada setiap tipe unit kerja**" memberi dimensi struktural kedua. COUNTER-RULE (c + b) berlaku → TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 11c — pertanyaan: "Rekap pegawai per generasi di setiap tipe unit kerja"
Analisis: dua dimensi agregasi konkret ("per generasi" = bucket tahun_lahir baku, "di setiap tipe unit kerja" = group by tipe_balai). Tidak ada vagueness. COUNTER-RULE (a + b) berlaku → TIDAK ambigu.
{{
  \"is_ambiguous\": false
}}

## Contoh 12 — pertanyaan: "Tampilkan data pendidikan pegawai"
Analisis: vague — "data pendidikan" bisa: daftar pendidikan terakhir per pegawai, rekap agregat per jenjang/universitas, atau filter pegawai dengan kriteria pendidikan tertentu. Tidak ada filter atau dimensi yang ditetapkan.
{{
  \"is_ambiguous\": true,
  \"ambiguity_type\": \"scope\",
  \"clarification_question\": \"Anda ingin melihat daftar pendidikan terakhir per pegawai, rekap jumlah pegawai per jenjang/universitas, atau pegawai dengan pendidikan tertentu (mis. lulusan kampus X)?\",
  \"interpretation_options\": [
    {{
      \"label\": \"Daftar pendidikan terakhir per pegawai (NIP, nama, jenjang, sekolah, jurusan)\",
      \"description\": \"Menampilkan satu baris per pegawai aktif berisi nama, jenjang, nama sekolah, dan program studi pendidikan terakhir. Cocok untuk profil SDM. Sumber: JOIN public.pegawai_tm × siap.V_PENDIDIKAN_TERAKHIR (DISTINCT ON pegawai_id).\"
    }},
    {{
      \"label\": \"Rekap jumlah pegawai per jenjang pendidikan\",
      \"description\": \"Mengelompokkan jumlah pegawai aktif ke bucket jenjang pendidikan baku (SD/SMA/D3/D4-S1/S2/S3). Cocok untuk overview komposisi tingkat pendidikan.\"
    }},
    {{
      \"label\": \"Daftar pegawai dengan pendidikan dari kampus / jurusan tertentu\",
      \"description\": \"Filter pegawai berdasarkan nama universitas atau program studi yang Anda sebut. Anda perlu menyebut kampus/jurusan yang dimaksud.\"
    }}
  ]
}}

# Task Input
[question]: {question}
[schema_context]: {schema_context}
[conversation_history (format={history_format_label})]:
{history_block}

# Refocus
Ingat: pengguna adalah ATASAN/MANAJEMEN non-teknis. Buat ``clarification_question`` dan setiap ``label`` dalam bahasa bisnis sehari-hari tanpa nama tabel/kolom/SQL. Pakai ``description`` untuk penjelasan lebih dalam dan referensi sumber data dengan terjemahan domain.

# Transition
Keluarkan JSON valid saja, sesuai format di Examples.
"""


def build_refinement_prompt(
    original_question: str,
    ambiguity_type: str,
    clarification_question: str,
    user_response: str,
    conversation_history: list[dict] | None = None,
) -> str:
    history_lines: list[str] = []
    for turn in (conversation_history or [])[-6:]:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        role_label = "user" if role == "user" else "assistant"
        history_lines.append(f"- {role_label}: {content}")
    history_block = (
        "\n".join(history_lines) if history_lines else "(tidak ada riwayat percakapan sebelumnya)"
    )

    return f"""# Introduction
Anda adalah komponen query refinement dalam pipeline Text-to-SQL.

# Scope
Hasil akhir HARUS satu kalimat eksplisit Bahasa Indonesia. Jangan buat SQL. Jangan tambah informasi baru selain yang sudah tersirat di riwayat percakapan + pertanyaan asli + jawaban klarifikasi.

# Instructions
1. Gabungkan pertanyaan asli, jawaban klarifikasi, DAN filter/entitas yang dibawa dari turn sebelumnya di [riwayat_percakapan].
2. Pastikan subjek dan kondisi hasil klarifikasi menjadi eksplisit.
3. WAJIB pertahankan filter/atribut/entitas dari turn sebelumnya yang masih relevan (mis. "pendidikan S2", "jabatan analis", "jenis kelamin perempuan", nama unit kerja, dst.) — JANGAN dihilangkan hanya karena tidak diulang di pertanyaan terakhir. Kata ganti seperti "pegawainya", "datanya", "mereka" merujuk ke entitas hasil filter pada turn sebelumnya, jadi filter itu HARUS ikut.
4. Keluaran harus ringkas, jelas, dan siap dipakai untuk SQL generation.

# Task Input
[riwayat_percakapan]:
{history_block}

[original_question]: {original_question}
[ambiguity_type]: {ambiguity_type}
[clarification_question]: {clarification_question}
[user_response]: {user_response}

# Transition
Keluarkan satu kalimat saja.
"""
