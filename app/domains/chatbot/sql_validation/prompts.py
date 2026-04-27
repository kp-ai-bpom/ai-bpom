"""Prompt builders untuk Refiner dan Semantic Judge."""

from __future__ import annotations


_REFINER_SYSTEM = (
    "Anda adalah revisor SQL PostgreSQL untuk basis data kepegawaian BPOM. "
    "Tugas Anda memperbaiki SQL kandidat agar sesuai pertanyaan pengguna dan "
    "skema yang diberikan. Keluarkan JSON valid saja, tanpa penjelasan di luar JSON."
)

_JUDGE_SYSTEM = (
    "Anda adalah penilai (judge) SQL PostgreSQL untuk basis data kepegawaian BPOM. "
    "Tugas Anda menilai apakah SQL kandidat menjawab pertanyaan pengguna dengan "
    "tepat berdasarkan skema yang diberikan. Keluarkan JSON valid saja."
)


def refiner_system_message() -> str:
    return _REFINER_SYSTEM


def judge_system_message() -> str:
    return _JUDGE_SYSTEM


def build_refiner_prompt(
    *,
    question: str,
    schema_context: str,
    broken_sql: str,
    feedback: str,
    trigger: str,
) -> str:
    """Bangun prompt revisi SQL.

    ``trigger`` membantu refiner menyesuaikan fokus revisi: ``execution_error``
    mengarah ke perbaikan sintaksis/referensi; ``semantic_partial`` mengarah ke
    perbaikan kesesuaian dengan pertanyaan (mis. filter atau agregasi yang
    kurang tepat).
    """
    if trigger == "execution_error":
        focus_block = (
            "Fokus revisi: SQL gagal dieksekusi. Perbaiki kesalahan sintaksis, "
            "nama tabel/kolom yang salah, atau referensi yang tidak ada. "
            "Pertahankan maksud asli pertanyaan."
        )
    else:
        focus_block = (
            "Fokus revisi: SQL bisa dieksekusi tetapi belum sepenuhnya tepat "
            "menjawab pertanyaan. Perbaiki klausa SELECT/WHERE/GROUP BY/JOIN "
            "agar selaras dengan maksud pertanyaan."
        )

    return f"""# Pertanyaan pengguna
{question.strip()}

# Skema yang relevan
{schema_context.strip()}

# SQL kandidat sebelumnya
```sql
{broken_sql.strip()}
```

# Umpan balik validasi
{feedback.strip() or "(tidak ada umpan balik tambahan)"}

# Tugas
{focus_block}

# Format output (WAJIB JSON valid, tanpa teks lain)
{{
  "query": "<SQL PostgreSQL hasil revisi, satu baris atau multi-baris>",
  "explanation": "<penjelasan singkat satu kalimat tentang perubahan yang dilakukan>"
}}
"""


def build_judge_prompt(
    *,
    question: str,
    schema_context: str,
    sql: str,
) -> str:
    """Bangun prompt penilaian rubric 5 dimensi.

    Definisi tiap dimensi diambil dari Bab 2.9 (hybrid Huyen LLM-as-a-Judge
    untuk Faithfulness/Relevance + Spider Component Matching untuk
    SELECT/WHERE/GROUP BY).
    """
    return f"""# Pertanyaan pengguna
{question.strip()}

# Skema yang relevan
{schema_context.strip()}

# SQL kandidat
```sql
{sql.strip()}
```

# Tugas
Nilai SQL kandidat di atas pada lima dimensi berikut. Untuk tiap dimensi, beri
label salah satu dari ``PASS``, ``PARTIAL``, atau ``FAIL`` beserta alasan singkat
(maksimum 25 kata).

## ATURAN ANTI-HALUSINASI (BACA INI LEBIH DULU)
Empat kesalahan umum penilai sebelumnya yang HARUS Anda hindari:

**A. JANGAN tertukar TEXT name vs ID.** Konvensi nama kolom skema BPOM:
   - Kolom **ID** berakhiran ``_id`` (mis. ``pendidikan_top_id``,
     ``jabatan_id``, ``satker_top_id``, ``pangkat_id``).
   - Kolom **TEXT name** TIDAK berakhiran ``_id`` (mis. ``v.pendidikan``,
     ``j.jabatan_nama``, ``s.namasatker``, ``s.satker_nama``,
     ``pk.pangkat_nama``, ``vpt.namasekolah``, ``vpt.programstudi``,
     ``p.nama``, ``p.nama_lengkap_gelar``, ``pe.full_name``,
     ``pe.position_name``).
   - Bila SQL memuat ``v.pendidikan AS pendidikan_terakhir`` itu **TEXT name**,
     bukan ID. Jangan beri PARTIAL/FAIL dengan alasan "via ID, bukan nama"
     atau "hanya agregasi berdasarkan ID/kelompok pendidikan".

**B. Conditional aggregation = pola STANDAR untuk rekap pendidikan/generasi/
jenjang per tipe unit kerja.** Bila SQL memakai
``SUM(CASE WHEN p.pendidikan_top_id IN ('09','10','11') THEN 1 ELSE 0 END)
AS d4_s1`` (atau analog untuk generasi: ``SUM(CASE WHEN tahun_lahir BETWEEN
1981 AND 1996 THEN 1 ELSE 0 END) AS millennial``), itu adalah **jawaban
yang benar dan presentasional** untuk pertanyaan rekap. Alias kategori
(``unset``, ``under_d3``, ``d3``, ``d4_s1``, ``profesi``, ``s2``, ``s3``,
``baby_boomers``, ``gen_x``, ``millennial``, ``gen_z``) sudah self-explanatory.
Beri **PASS** pada ``relevance`` DAN ``select`` untuk pola ini — JANGAN
labelkan PARTIAL dengan alasan "via ID, bukan nama" atau "tidak menampilkan
nama pendidikan per kategori".

**C. Filter default pegawai aktif boleh berada di lokasi mana pun yang setara
secara semantik.** Sebelum labelkan ``where: FAIL`` "filter default pegawai
aktif tidak dipasang", scan **SELURUH teks SQL** dan terima penempatan
filter di SEMUA lokasi berikut sebagai SAH:
   1. Outer ``WHERE`` clause.
   2. Inner subquery / derived table ``WHERE``.
   3. CTE (``WITH x AS (... WHERE ...)``) ``WHERE``.
   4. ``JOIN ... ON ... AND p.status_pegawai IN (...)`` — termasuk JOIN ON
      yang menggabungkan tabel ``pegawai_tm`` ke tabel lain.
   5. Kombinasi: satu filter di subquery WHERE, satu lagi di outer JOIN ON
      (semantik tetap memfilter pegawai aktif walaupun penempatan terpisah).
Cari literal ``status_pegawai`` dan ``kedudukan_pegawai`` di mana pun.
Bila kedua kolom ditemukan dengan nilai ``IN ('CPNS','PNS','POLRI','PPPK')``
dan ``IN ('Aktif','Tugas Belajar','CLTN')`` di lokasi mana pun (boleh berbeda
scope), semantiknya sudah memfilter pegawai aktif → ``where: PASS``.

CATATAN: Penempatan filter di JOIN ON atau di subquery yang tidak meng-expose
kolom adalah masalah **STRUKTURAL generator**, bukan masalah semantik filter.
Bila SQL sudah lulus eksekusi (Anda dipanggil setelah validator eksekusi PASS),
artinya semantiknya sah. Tugas Anda menilai apakah filter ada — bukan menilai
gaya penempatan. Beri PASS dan biarkan reviewer manusia / refiner generator
yang menangani gaya struktural.

**D. Validator eksekusi sudah membuktikan keberadaan tabel/kolom secara fisik.**
Anda dipanggil sesudah eksekusi PASS. JANGAN ragu kolom dengan alasan
"kolom X kemungkinan salah penamaan" atau "kolom Y tidak terkonfirmasi di
skema". Variant nama kolom seperti ``p.nama_lengkap_gelar``, ``p.nama``,
``pe.full_name``, ``vpt.namasekolah`` semua TERBUKTI ada bila SQL lulus
eksekusi.

Pelanggaran A/B sangat merugikan karena memicu refiner mengubah SQL yang
sudah benar menjadi salah. Bila ragu, default ke PASS.


1. **faithfulness** — apakah SQL menggunakan entitas (tabel/kolom) yang masuk akal
   di domain HR BPOM, BUKAN entitas yang dikarang sepenuhnya (mis. tabel "products",
   kolom "harga_jual", atau modul yang jelas-jelas di luar HR).
   ATURAN UTAMA: **Default label = PASS.** Validator eksekusi (Tahap 5 sebelum Anda)
   sudah menjalankan SQL terhadap database asli; bila SQL berhasil eksekusi, semua
   tabel/kolom secara fisik PASTI ada. Anda hanya dipanggil setelah eksekusi PASS,
   sehingga keberadaan kolom BUKAN domain Anda. Tugas Anda di sini sempit:
   mendeteksi entitas yang JELAS-JELAS di luar domain HR (mis. tabel inventaris,
   tabel transaksi, tabel produk).
   PROTOKOL VERIFIKASI WAJIB sebelum memberi label PARTIAL/FAIL pada faithfulness:
   (a) Periksa apakah ada nama tabel di SQL yang tidak ada hubungannya sama sekali
       dengan domain HR/kepegawaian (mis. `produk_tm`, `transaksi_jual`, `inventory`).
       Jika SEMUA tabel yang dipakai SQL berada di domain HR (pegawai_tm, jabatan_tm,
       pangkat_tm, V_PENDIDIKAN_TERAKHIR, SIAP_SATKER_TOP, mantel.*, dll.) → PASS.
   (b) Untuk klaim "kolom X tidak ada di tabel Y": JANGAN buat klaim ini. Validator
       eksekusi sudah memvalidasi keberadaan kolom secara fisik. Jika SQL eksekusi
       sukses, semua referensi `alias.kolom` dijamin ada. Bila Anda hanya tidak
       MELIHAT kolom di [Skema yang relevan], itu artinya schema retrieval tidak
       menarik blok lengkap (bukan halusinasi generator) → tetap PASS.
   (c) FAIL hanya bila SQL menyebut nama tabel atau JOIN ke tabel yang jelas-jelas
       fiktif di domain HR (mis. `karyawan_inventory`, `gaji_produk`). Bila ragu
       sedikit pun, label PASS. False-positive faithfulness merugikan pipeline lebih
       besar daripada false-negative.
2. **relevance** — apakah SQL menjawab pertanyaan yang diajukan, bukan menjawab
   pertanyaan lain yang mirip. ATURAN PROTOKOL VERIFIKASI (sama seperti
   faithfulness): JANGAN menurunkan label hanya karena tabel/kolom tidak terlihat
   di blok ``[Skema yang relevan]``. Validator eksekusi sudah membuktikan tabel/
   kolom secara fisik ada di database; ketidakmunculan blok di sini hanya berarti
   schema retrieval tidak menarik blok lengkap. Jadi jangan beri PARTIAL/FAIL
   dengan alasan "tabel X tidak ada di skema" atau "isi kolom Y mungkin tidak
   benar karena tabel tidak terkonfirmasi". Beri PARTIAL/FAIL hanya bila SQL
   memang menjawab pertanyaan yang berbeda dari yang diajukan user.
   PENTING: jangan menuduh SQL "memakai ID bukan nama" tanpa memverifikasi
   nama kolom. Kolom **TEXT name** di skema BPOM sangat banyak dan TIDAK
   diakhiri ``_id``: ``v.pendidikan``, ``j.jabatan_nama``, ``s.namasatker``,
   ``pk.pangkat_nama``, ``vpt.namasekolah``, ``vpt.programstudi``,
   ``vpt.jurusan``, ``p.nama``, ``p.nama_lengkap_gelar``, dll. Kolom **ID**
   biasanya berakhiran ``_id`` (``pendidikan_top_id``, ``jabatan_id``,
   ``satker_top_id``, ``pangkat_id``). Bila SELECT memuat ``v.pendidikan AS
   pendidikan_terakhir`` (TEXT name) dengan ``GROUP BY ... v.pendidikan``,
   itu adalah rekap "berdasarkan nama pendidikan terakhir" — JANGAN
   labelkan PARTIAL dengan alasan "via ID, bukan nama".
3. **select** — apakah daftar kolom pada klausa SELECT memuat informasi inti
   yang diminta pengguna. Default label = PASS. Hanya beri PARTIAL bila ada
   informasi inti yang DIMINTA pengguna tapi TIDAK ADA di SELECT (mis. user
   minta "nama beserta jabatannya" tapi SQL hanya menampilkan nama).
   **PROTOKOL KEPERCAYAAN KOLOM**: Bila ``validation_status`` execution
   sudah PASS (Anda dipanggil sesudahnya), berarti SEMUA kolom di SQL —
   termasuk varian penamaan domain seperti ``p.nama_lengkap_gelar``,
   ``p.nama``, ``p.full_name``, ``vpt.namasekolah``, ``j.jabatan_nama``,
   ``pe.full_name``, ``pe.position_name``, ``pe.work_unit_name``, dll. —
   sudah terbukti ada di tabel masing-masing. JANGAN beri PARTIAL/FAIL
   dengan alasan "kolom X kemungkinan salah penamaan", "kolom nama tidak
   jelas tersedia (p.nama)", atau "ragu kolom Y benar-benar ada". Bila
   SQL pakai ``p.nama_lengkap_gelar AS nama``, terima saja sebagai kolom
   nama lengkap dengan gelar (formal full-name dengan academic title).
   JANGAN
   beri PARTIAL hanya karena ada kolom tambahan yang dipakai SQL untuk
   transparansi/disambiguasi (mis. ikut menampilkan vpt.jurusan, vpt.programstudi,
   p.status_pegawai, p.kedudukan_pegawai padahal user tidak minta eksplisit) —
   itu adalah praktik baik untuk auditability dan TIDAK mengurangi keakuratan
   jawaban. JANGAN juga menurunkan label hanya karena tabel asal kolom tidak
   muncul di blok ``[Skema yang relevan]`` (mis. "jabatan_nama dari jabatan_tm
   yang tidak terkonfirmasi di skema") — validator eksekusi sudah membuktikan
   tabel/kolom secara fisik ada. JANGAN pula menurunkan label hanya karena
   SQL menggunakan **conditional aggregation** (``SUM(CASE WHEN
   pendidikan_top_id IN ('09','10','11') THEN 1 ELSE 0 END) AS d4_s1``)
   alih-alih JOIN ke tabel referensi nama pendidikan: pola ini adalah
   praktik baku untuk rekap pendidikan dan **alias kolom** (``d3``,
   ``d4_s1``, ``profesi``, ``s2``, ``s3``, ``under_d3``, dst.) sudah
   menjadi label kategori yang informatif untuk presentasi tabel.
   Selama alias kolom self-explanatory di domain HR (singkatan jenjang
   pendidikan/generasi/tipe unit kerja), jangan beri PARTIAL hanya karena
   "tidak menyebut nama pendidikan/jenjang per kategori". JANGAN pula
   tertukar antara kolom **kode/ID** (``pendidikan_top_id``,
   ``jabatan_id``, ``satker_top_id``, ``pangkat_id``, dll. — biasanya
   pendek/numerik/alfanumerik) dan kolom **nama** (``v.pendidikan``,
   ``j.jabatan_nama``, ``s.namasatker``, ``pk.pangkat_nama``,
   ``vpt.namasekolah``, dll. — TEXT bebas). SQL yang menampilkan
   ``v.pendidikan AS pendidikan_terakhir`` (TEXT name) sudah memenuhi
   permintaan "nama pendidikan terakhir"; jangan bilang "memakai ID bukan
   nama" hanya karena belum melihat skemanya. Hanya beri FAIL bila SELECT
   sama sekali tidak menjawab pertanyaan (mis. user minta jumlah, SQL
   hanya menampilkan list nama).
4. **where** — apakah klausa WHERE memfilter sesuai kondisi yang diminta.
   PENTING: filter default "pegawai aktif" (`status_pegawai IN ('CPNS','PNS','POLRI','PPPK')`
   dan `kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN')`) sudah menjadi
   tanggung jawab SQL generator (ada validator otomatis di sisi generator yang
   memaksa retry bila hilang). Jika SQL akhir yang Anda nilai TIDAK memuat
   kedua filter tersebut padahal pertanyaan TIDAK eksplisit minta data non-aktif,
   beri ``FAIL`` pada dimensi where dengan reason "filter default pegawai aktif
   tidak dipasang" — ini akan memicu refiner untuk memperbaikinya. Bila
   pengguna eksplisit minta "termasuk non aktif/semua status/pensiun/berhenti",
   absennya filter default justru benar → PASS.
   PENTING: cek keberadaan filter default pada **SELURUH teks SQL**, termasuk
   di dalam **subquery / CTE / derived table / inline view**. Pola yang sangat
   umum adalah inner subquery memfilter pegawai aktif lebih dulu (mis.
   ``FROM (SELECT DISTINCT ON (p.pegawai_id) ... FROM public.pegawai_tm p
   WHERE p.tgl_lahir IS NOT NULL AND p.status_pegawai IN ('CPNS','PNS','POLRI','PPPK')
   AND p.kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN') ...) p JOIN ...``)
   sebelum di-JOIN/agregasi di outer query. Selama kedua filter
   ``status_pegawai IN ('CPNS','PNS','POLRI','PPPK')`` dan
   ``kedudukan_pegawai IN ('Aktif','Tugas Belajar','CLTN')`` muncul **di mana
   pun** dalam teks SQL (outer WHERE, subquery, atau CTE) **dan** subquery
   tersebut adalah satu-satunya jalur yang memasok baris pegawai untuk
   agregasi outer, semantiknya tetap memfilter pegawai aktif → PASS pada
   dimensi where. JANGAN beri FAIL hanya karena outer WHERE kosong/lemah
   tanpa membaca isi subquery.
5. **group_by** — apakah klausa GROUP BY (jika ada) sesuai dengan tingkat
   agregasi yang diminta; bila pertanyaan tidak meminta agregasi dan SQL tidak
   memakainya, beri ``PASS``.

Beri pula label agregat ``overall`` dengan kebijakan: ada satu ``FAIL`` →
``FAIL``; tidak ada ``FAIL`` tetapi ada ``PARTIAL`` → ``PARTIAL``; semuanya
``PASS`` → ``PASS``.

# Format output (WAJIB JSON valid, tanpa teks lain)
{{
  "rubric": {{
    "faithfulness": {{"label": "PASS|PARTIAL|FAIL", "reason": "..."}},
    "relevance":    {{"label": "PASS|PARTIAL|FAIL", "reason": "..."}},
    "select":       {{"label": "PASS|PARTIAL|FAIL", "reason": "..."}},
    "where":        {{"label": "PASS|PARTIAL|FAIL", "reason": "..."}},
    "group_by":     {{"label": "PASS|PARTIAL|FAIL", "reason": "..."}}
  }},
  "overall": "PASS|PARTIAL|FAIL",
  "summary": "<satu kalimat ringkasan, maksimum 30 kata>"
}}
"""
