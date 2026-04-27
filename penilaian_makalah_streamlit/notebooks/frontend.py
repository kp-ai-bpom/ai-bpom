import streamlit as st
import requests
import pandas as pd
import time
import os
import json
from pathlib import Path

# Base URL API Service kita
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080/api/penilaian-makalah")

# ══════════════════════════════════════════════════════════════════════════════
# API CLIENT LAYER
# ══════════════════════════════════════════════════════════════════════════════

def api_get_jabatan():
    try:
        r = requests.get(f"{API_BASE_URL}/jabatan")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Gagal mengambil daftar jabatan: {e}")
        return []

def api_get_tema():
    try:
        r = requests.get(f"{API_BASE_URL}/tema")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Gagal mengambil daftar tema: {e}")
        return []

def api_get_makalah_detailed():
    try:
        r = requests.get(f"{API_BASE_URL}/makalah")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Gagal mengambil daftar makalah: {e}")
        return []

def api_get_makalah_text(filename):
    try:
        r = requests.get(f"{API_BASE_URL}/makalah/{filename}/text")
        if r.status_code == 200:
            return r.json().get("text", "")
        return f"Gagal: {r.text}"
    except Exception as e:
        return f"Error koneksi: {e}"

def api_get_history(limit=100):
    try:
        r = requests.get(f"{API_BASE_URL}/history", params={"limit": limit})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Gagal mengambil histori: {e}")
        return []

def api_upload_file(kategori, uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        r = requests.post(f"{API_BASE_URL}/upload/{kategori}", files=files)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Gagal mengunggah {kategori}: {e}")
        return None

def api_ingest():
    try:
        r = requests.post(f"{API_BASE_URL}/ingest")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Gagal memproses ingestion: {e}")
        return None

def api_evaluate(jabatan, filename_makalah, filename_tema, query_mode):
    payload = {
        "jabatan": jabatan,
        "filename_makalah": filename_makalah,
        "filename_tema": filename_tema,
        "query_mode": query_mode
    }
    try:
        r = requests.post(f"{API_BASE_URL}/evaluate", json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Gagal memproses evaluasi: {e}")
        return None

def compute_final_score(scores: dict) -> float:
    SCORE_WEIGHTS = {
        "n1_kesesuaian_judul":   1,
        "n2_kesesuaian_isi":     1,
        "n3_sistematika":        1,
        "n4_ketajaman_analisis": 2,
        "n5_penggunaan_bahasa":  1,
    }
    total_weight = sum(SCORE_WEIGHTS.values())
    weighted_sum = sum(scores.get(k, 0) * w for k, w in SCORE_WEIGHTS.items())
    return round(weighted_sum / total_weight, 1)

# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

SCORE_LABELS = {
    "n1_kesesuaian_judul":   "Kesesuaian Judul dengan Tema",
    "n2_kesesuaian_isi":     "Kesesuaian Isi dengan Judul & Tema",
    "n3_sistematika":        "Sistematika Penulisan",
    "n4_ketajaman_analisis": "Ketajaman Analisis",
    "n5_penggunaan_bahasa":  "Penggunaan Bahasa",
}

SCORE_WEIGHTS = {
    "n1_kesesuaian_judul":   1,
    "n2_kesesuaian_isi":     1,
    "n3_sistematika":        1,
    "n4_ketajaman_analisis": 2,
    "n5_penggunaan_bahasa":  1,
}

def score_color(score: float) -> str:
    if score >= 85:   return "#22c55e"
    elif score >= 70: return "#f59e0b"
    elif score >= 55: return "#f97316"
    else:             return "#ef4444"

def render_score_card(result: dict):
    scores    = result.get("scores", {})
    final     = result.get("final_score", compute_final_score(scores))
    justif    = result.get("justification", {})
    evidence  = result.get("evidence", {})
    ringkasan = result.get("ringkasan", result.get("Ringkasan", ""))
    
    color     = score_color(final)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{color}22,{color}11);
      border:2px solid {color};border-radius:20px;padding:32px;text-align:center;margin-bottom:24px;">
      <div style="font-size:13px;color:#94a3b8;font-weight:600;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;">NILAI AKHIR</div>
      <div style="font-size:80px;font-weight:800;color:{color};line-height:1;">{final}</div>
      <div style="font-size:12px;color:#64748b;margin-top:4px;">dari 100</div>
    </div>
    """, unsafe_allow_html=True)

    if ringkasan:
        st.markdown("**📄 Ringkasan Makalah**")
        st.info(ringkasan)

    st.markdown("**Rincian Skor per Kriteria**")
    for key, label in SCORE_LABELS.items():
        score  = scores.get(key, 0)
        weight = SCORE_WEIGHTS[key]
        sc     = score_color(score)
        with st.expander(f"{label}  —  **{score}**"):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown(f"""
                <div style="background:{sc}22;border:2px solid {sc};border-radius:12px;
                  padding:16px;text-align:center;">
                  <div style="font-size:36px;font-weight:800;color:{sc};">{score}</div>
                  <div style="font-size:11px;color:#64748b;">bobot ×{weight}</div>
                </div>""", unsafe_allow_html=True)
                st.progress(max(0, min(100, int(score))) / 100)
            with c2:
                j = justif.get(key, "")
                e = evidence.get(key, "")
                if j:
                    st.markdown(f"**📝 Justifikasi:**\n{j}")
                if e:
                    st.markdown(f"**🔎 Bukti:**\n_{e}_")

# ══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Penilaian Makalah — AI BPOM",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.score-card { border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 24px; }
.score-num { font-size: 72px; font-weight: 800; line-height: 1; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "📝 Penilaian Makalah"
if "eval_results_batch" not in st.session_state:
    st.session_state.eval_results_batch = {}
if "loaded_papers_names" not in st.session_state:
    st.session_state.loaded_papers_names = []

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
      <div style="font-size:32px;">📋</div>
      <div style="font-size:18px; font-weight:800; color:#e2e8f0;">Penilaian Makalah</div>
      <div style="font-size:11px; color:#00008B; letter-spacing:1px;">API Client Mode · BPOM</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    nav_options = ["📝 Penilaian Makalah", "📊 Riwayat Penilaian", "⚙️ Docs & Settings"]
    st.session_state.nav_page = st.radio("Navigasi", nav_options,
                                         index=nav_options.index(st.session_state.nav_page),
                                         label_visibility="collapsed")
    
    st.divider()
    query_mode = st.selectbox("Query Mode", ["hybrid", "mix", "local", "global", "naive"], index=0, help="Mode query LightRAG Backend")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PENILAIAN MAKALAH
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.nav_page == "📝 Penilaian Makalah":
    st.markdown("## 📝 Penilaian Makalah")

    jabatans = api_get_jabatan()
    temas = api_get_tema()

    with st.container(border=True):
        st.markdown("""<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
          <div style="background:#3b82f6;color:white;width:28px;height:28px;border-radius:50%;
            display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;">1</div>
          <div style="font-size:16px;font-weight:700;color:#090088;">Pilih Jabatan & Tema Penulisan Makalah</div>
        </div>""", unsafe_allow_html=True)
        
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            jabatan = st.selectbox("Pilih Jabatan Target", options=["Pilih Jabatan"] + jabatans)
        with col_j2:
            tema_file = st.selectbox("Pilih Ketentuan Tema", options=["Pilih Tema"] + temas)

    st.divider()

    with st.container(border=True):
        st.markdown("""<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
          <div style="background:#3b82f6;color:white;width:28px;height:28px;border-radius:50%;
            display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;">2</div>
          <div style="font-size:16px;font-weight:700;color:#090088;">Pilih Makalah yang Akan Dinilai</div>
        </div>""", unsafe_allow_html=True)

        src_tab1, src_tab2 = st.tabs(["📂 Pilih dari MinIO", "⬆️ Upload File Baru"])

        with src_tab1:
            minio_files_detail = api_get_makalah_detailed()
            if minio_files_detail:
                df_minio = pd.DataFrame(minio_files_detail)
                df_minio.insert(0, "Pilih", False)
                
                st.markdown(f"**{len(minio_files_detail)} file tersedia di MinIO**")
                st.markdown("Centang file yang ingin dinilai pada tabel di bawah:")

                edited_df = st.data_editor(
                    df_minio,
                    column_config={
                        "Pilih": st.column_config.CheckboxColumn("Pilih", default=False)
                    },
                    disabled=["Nama File", "Ukuran", "Terakhir Diubah"],
                    hide_index=True,
                    use_container_width=True,
                )

                sel_keys = edited_df[edited_df["Pilih"]]["Nama File"].tolist()

                if sel_keys:
                    if st.button(f"📥 Konfirmasi {len(sel_keys)} File Terpilih", type="secondary", use_container_width=True):
                        st.session_state.loaded_papers_names = list(set(st.session_state.loaded_papers_names + sel_keys))
                        st.toast(f"✅ Berhasil memilih {len(sel_keys)} file.")
            else:
                st.info("Belum ada file di MinIO bucket makalah.")

        with src_tab2:
            uploaded_files = st.file_uploader(
                "Upload makalah (PDF / DOCX / TXT) — bisa multiple",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                if st.button("Unggah File & Tambahkan ke Antrean"):
                    with st.spinner("Mengunggah file ke backend..."):
                        for uf in uploaded_files:
                            res = api_upload_file("makalah", uf)
                            if res:
                                st.session_state.loaded_papers_names.append(uf.name)
                                st.toast(f"✅ Berhasil unggah: {uf.name}")
                        st.session_state.loaded_papers_names = list(set(st.session_state.loaded_papers_names))

        if st.session_state.loaded_papers_names:
            st.markdown(f"**{len(st.session_state.loaded_papers_names)} makalah siap dinilai:**")
            for fn in st.session_state.loaded_papers_names:
                st.markdown(f"- `{fn}`")
            if st.button("🗑️ Bersihkan Daftar Makalah", use_container_width=True):
                st.session_state.loaded_papers_names = []
                st.session_state.eval_results_batch = {}
                st.rerun()

    st.divider()

    with st.container(border=True):
        st.markdown("""<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
          <div style="background:#3b82f6;color:white;width:28px;height:28px;border-radius:50%;
            display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;">3</div>
          <div style="font-size:16px;font-weight:700;color:#090088;">Mulai Penilaian</div>
        </div>""", unsafe_allow_html=True)

        can_eval = bool(jabatan != "Pilih Jabatan" and tema_file != "Pilih Tema" and st.session_state.loaded_papers_names)

        if not can_eval:
            st.info("ℹ️ Pastikan Anda sudah memilih Jabatan, Tema, dan minimal 1 Makalah.")

        if st.button("🚀 Mulai Penilaian Otomatis", type="primary", disabled=not can_eval, use_container_width=True):
            results = {}
            with st.status("Memproses Penilaian...", expanded=True) as status_box:
                total = len(st.session_state.loaded_papers_names)
                for idx, fname in enumerate(st.session_state.loaded_papers_names, 1):
                    st.write(f"⏳ Menilai: {fname} ({idx}/{total})...")
                    start_time = time.time()
                    eval_res = api_evaluate(jabatan, fname, tema_file, query_mode)
                    
                    if eval_res:
                        results[fname] = eval_res
                    else:
                        st.write(f"❌ Gagal menilai {fname}")
                
                status_box.update(label=f"✅ Penilaian selesai! {len(results)} makalah telah dinilai.", state="complete", expanded=False)
                st.session_state.eval_results_batch = results

    batch = st.session_state.get("eval_results_batch", {})
    if batch:
        st.divider()
        st.markdown("## 📊 Hasil Penilaian")

        # Summary table
        rows = []
        for fn, r in batch.items():
            sc = r.get("scores", {})
            rows.append({
                "Makalah": fn,
                "Judul": sc.get("n1_kesesuaian_judul", None),
                "Isi": sc.get("n2_kesesuaian_isi", None),
                "Sistematika": sc.get("n3_sistematika", None),
                "Analisis": sc.get("n4_ketajaman_analisis", None),
                "Bahasa": sc.get("n5_penggunaan_bahasa", None),
                "Nilai Akhir": r.get("final_score", 0),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Detail per makalah
        st.markdown("### 🔍 Detail per Makalah")
        
        if "loaded_papers_text" not in st.session_state:
            st.session_state.loaded_papers_text = {}
            
        doc_tabs = st.tabs([f"📄 {fn}" for fn in batch.keys()])
        for tab, (fn, r) in zip(doc_tabs, batch.items()):
            with tab:
                col_left, col_right = st.columns([1, 1.2])
                
                with col_left:
                    st.markdown("#### 📄 Teks Makalah")
                    if fn not in st.session_state.loaded_papers_text:
                        with st.spinner("Memuat teks makalah..."):
                            st.session_state.loaded_papers_text[fn] = api_get_makalah_text(fn)
                    
                    paper_text = st.session_state.loaded_papers_text[fn]
                    st.markdown(
                        f'<div style="height:800px;overflow-y:auto;padding:16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;font-size:14px;white-space:pre-wrap;">{paper_text}</div>',
                        unsafe_allow_html=True
                    )
                
                with col_right:
                    render_score_card(r)
                    st.divider()
                    col_dl, col_js = st.columns(2)
                    with col_dl:
                        st.download_button(
                            "⬇️ Download JSON",
                            data=json.dumps(r, ensure_ascii=False, indent=2),
                            file_name=f"penilaian_{Path(fn).stem}.json",
                            mime="application/json",
                            use_container_width=True,
                            key=f"dl_batch_{fn}",
                        )
                    with col_js:
                        with st.expander("🔧 Raw JSON"):
                            st.json(r)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RIWAYAT PENILAIAN
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "📊 Riwayat Penilaian":
    st.markdown("## 📊 Riwayat Penilaian Makalah")

    jabatans = api_get_jabatan()

    with st.container():
        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1, 1, 1])
        with col_f1:
            jabatan_opts = ["Semua"] + jabatans
            filter_jabatan = st.selectbox("Filter Jabatan", jabatan_opts, key="hist_jabatan")
        with col_f2:
            filter_limit = st.number_input("Maks. Data", min_value=10, max_value=500, value=100, step=10)
        with col_f3:
            filter_score_min = st.number_input("Nilai Min.", min_value=0, max_value=100, value=0)
        with col_f4:
            st.write("")
            st.write("")
            refresh_btn = st.button("🔄 Muat Riwayat", use_container_width=True)

    history = api_get_history(int(filter_limit))
    
    # Client-side filtering because backend get_history doesn't support filter parameters yet.
    if filter_jabatan != "Semua":
        history = [h for h in history if h.get("jabatan") == filter_jabatan]
    if filter_score_min > 0:
        history = [h for h in history if h.get("final_score", 0) >= filter_score_min]

    if not history:
        st.info("Belum ada riwayat penilaian atau tidak ada yang cocok dengan filter.")
    else:
        # Summary metrics
        scores_list = [h.get("final_score", 0) for h in history if h.get("final_score") is not None]
        if scores_list:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Evaluasi", len(history))
            m2.metric("Rata-rata Nilai", f"{sum(scores_list)/len(scores_list):.1f}")
            m3.metric("Nilai Tertinggi", f"{max(scores_list):.1f}")
            m4.metric("Nilai Terendah", f"{min(scores_list):.1f}")

        st.divider()

        # Table
        table_data = []
        for h in history:
            table_data.append({
                "ID": h.get("id"),
                "Makalah": h.get("paper_filename"),
                "Jabatan": h.get("jabatan"),
                "Nilai Akhir": h.get("final_score"),
                "Mode": h.get("query_mode"),
                "Tanggal": str(h.get("created_at"))[:16] if h.get("created_at") else "-",
            })
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, height=280, hide_index=True)

        # Download all
        all_export = json.dumps({"total": len(history), "data": history}, ensure_ascii=False, indent=2)
        st.download_button("📥 Export Semua Riwayat (JSON)", data=all_export,
                           file_name=f"riwayat_penilaian_{len(history)}data.json",
                           mime="application/json")

        st.divider()
        st.markdown("### 🔍 Detail per Evaluasi")

        for h in history:
            final = h.get("final_score", 0) or 0
            color = score_color(final)
            date_str = str(h.get("created_at"))[:10] if h.get("created_at") else "-"
            with st.expander(f"[{final:.1f}]  {h.get('paper_filename')}  ·  {h.get('jabatan')}  ·  {date_str}"):
                scores = h.get("scores") or {}
                justif = h.get("justification") or {}

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,{color}22,{color}11);
                  border:1px solid {color};border-radius:12px;padding:16px;text-align:center;margin-bottom:16px;">
                  <div style="font-size:11px;color:#94a3b8;letter-spacing:2px;">NILAI AKHIR</div>
                  <div style="font-size:48px;font-weight:800;color:{color};line-height:1.1;">{final:.1f}</div>
                </div>""", unsafe_allow_html=True)

                ringkasan = h.get("ringkasan", "")
                if ringkasan:
                    st.markdown(f"**Ringkasan:** {ringkasan[:300]}{'...' if len(ringkasan) > 300 else ''}")

                st.markdown("**Skor per Kriteria:**")
                for key, label in SCORE_LABELS.items():
                    score = scores.get(key, "-")
                    j = justif.get(key, "")
                    w = SCORE_WEIGHTS.get(key, 1)
                    st.markdown(f"- {'⭐ ' if w > 1 else ''}**{label}**: `{score}`")
                    if j:
                        st.caption(f"  → {j[:150]}")

                st.download_button(
                    "⬇️ Download JSON",
                    data=json.dumps(h, ensure_ascii=False, indent=2),
                    file_name=f"eval_{h.get('id')}_{h.get('paper_filename')}.json",
                    mime="application/json",
                    key=f"dl_hist_{h.get('id')}",
                )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: DOCS & SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "⚙️ Docs & Settings":
    st.markdown("## ⚙️ Dokumen & Ingestion (Via API)")
    
    st.info("💡 Halaman ini mengirimkan file ke API Backend untuk diunggah ke MinIO dan diproses (Ingest) oleh LightRAG.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Upload Ketentuan Tema")
        uploaded_tema = st.file_uploader("Upload Ketentuan Tema (PDF/DOCX)", type=["pdf", "docx"], key="tema")
        if uploaded_tema and st.button("Upload Tema ke API"):
            if api_upload_file("tema", uploaded_tema):
                st.success("Tema berhasil diunggah via API Backend!")
                
    with col2:
        st.markdown("### Upload SKJ")
        uploaded_skj = st.file_uploader("Upload SKJ (JSON/PDF/DOCX)", type=["json", "pdf", "docx"], key="skj")
        if uploaded_skj and st.button("Upload SKJ ke API"):
            if api_upload_file("skj", uploaded_skj):
                st.success("SKJ berhasil diunggah via API Backend!")
                
    st.divider()
    st.markdown("### Memicu LightRAG Ingestion")
    st.write("Klik tombol di bawah ini untuk memerintahkan Backend API mengeksekusi (ingest) semua dokumen SKJ yang ada di MinIO ke dalam LightRAG. Backend akan menggunakan delta ingestion secara pintar.")
    
    if st.button("🚀 Panggil API Ingestion", type="primary"):
        with st.spinner("Menunggu Backend API menyelesaikan Ingestion..."):
            res = api_ingest()
            if res:
                st.success("Ingestion di sisi Backend selesai!")
                st.json(res)
