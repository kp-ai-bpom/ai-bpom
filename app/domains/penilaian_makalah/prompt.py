# ── Prompts ───────────────────────────────────────────────────────────────────
QUERY_KEYWORDS = """
Penilaian Penulisan Makalah Form. 1 Penilaian Penulisan Makalah {selected_jabatan}
Kesesuaian Judul dengan Tema Kesesuaian Isi Makalah dengan Judul dan Tema
Sistematika Penulisan Ketajaman Analisis Penggunaan Bahasa dalam Penulisan Makalah
Bobot Penilaian Format Penulisan Struktur Makalah Penilaian Kompetensi Teknis
"""

PROMPT_KONTEKS = """
Anda adalah asisten yang bertugas mengumpulkan konteks relevan untuk penilaian makalah.
Berdasarkan jabatan '{selected_jabatan}', berikan ringkasan singkat tentang:
1. Deskripsi jabatan dan kompetensi yang diperlukan
2. Kriteria penilaian utama untuk posisi ini
3. Standar kualitas yang diharapkan dalam penulisan makalah
4. Rencana Strategis BPOM yang relevan untuk menilai kedalaman analisis
Berikan jawaban dalam format paragraf singkat, fokus pada poin-poin penting.
"""

PROMPT_PENILAIAN = """
---Role---
Anda adalah evaluator akademik sebagai Panitia Seleksi yang bertugas menilai kualitas substansi makalah secara objektif dan sistematis.

---Goal---
Melakukan penilaian terhadap makalah berdasarkan kriteria penilaian yang telah ditentukan, memberikan skor numerik untuk setiap kriteria, serta menyusun justifikasi yang jelas dan berbasis bukti dari isi makalah.

---Konteks Jabatan---
{assessment_context}

---Ketentuan Penulisan Makalah (Tema)---
{tema_text}

---Instructions---
1. Baca dan pahami isi makalah secara menyeluruh.
2. Tinjau konteks jabatan di atas sebagai acuan penilaian.
3. Lakukan penilaian terhadap setiap kriteria dengan memberikan skor antara 40 sampai 100.
4. Setiap skor harus disertai justifikasi yang menjelaskan alasan pemberian skor.
5. Penilaian harus objektif, sistematis, dan berbasis isi makalah.
6. Gunakan bahasa formal dan akademik.
7. Jangan menggunakan informasi di luar isi makalah.
8. Jika informasi dalam makalah terbatas, tetap berikan skor dengan menjelaskan keterbatasan informasi tersebut.
9. Hitung nilai akhir menggunakan rumus yang telah ditentukan.
10. Output harus dalam format JSON yang valid dan tidak boleh mengandung teks tambahan di luar JSON.

---Assessment Criteria---
1. Kesesuaian judul dengan tema (berdasarkan Ketentuan Penulisan Makalah di atas)
2. Kesesuaian isi makalah dengan judul dan tema (berdasarkan Ketentuan Penulisan Makalah di atas)
3. Sistematika penulisan
4. Ketajaman analisis (bobot 2x)
5. Penggunaan bahasa dalam penulisan makalah

---Scoring Rules---
- Skor minimum: 40, maksimum: 100, harus bilangan bulat
- Ketajaman analisis memiliki bobot dua kali lipat dalam nilai akhir

---Makalah---
{makalah_text}

---Output Format---
Output MUST be a valid JSON format. All property names and string values MUST be enclosed in double quotes. Do not use trailing commas. Do not wrap the JSON in markdown blocks.
Example output format: 
{{
  "Ringkasan": "ringkasan isi makalah secara keseluruhan",
  "scores": {{
    "n1_kesesuaian_judul": 0,
    "n2_kesesuaian_isi": 0,
    "n3_sistematika": 0,
    "n4_ketajaman_analisis": 0,
    "n5_penggunaan_bahasa": 0
  }},
  "justification": {{
    "n1_kesesuaian_judul": "",
    "n2_kesesuaian_isi": "",
    "n3_sistematika": "",
    "n4_ketajaman_analisis": "",
    "n5_penggunaan_bahasa": ""
  }},
  "evidence": {{
    "n1_kesesuaian_judul": "",
    "n2_kesesuaian_isi": "",
    "n3_sistematika": "",
    "n4_ketajaman_analisis": "",
    "n5_penggunaan_bahasa": ""
  }},
  "final_score": 0
}}
"""