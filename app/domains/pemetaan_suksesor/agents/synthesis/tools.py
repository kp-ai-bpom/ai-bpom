"""Synthesis Agent deterministic scoring tools.

Adapted from hybrid-rag for the ai-bpom in-memory API architecture.
The tool receives analysis output and blueprint context as JSON strings
instead of file paths, since data flows in-memory between agents.
"""

from __future__ import annotations

import json
import re

from strands import tool


# ---------------------------------------------------------------------------
# Helper: safe numeric parsing
# ---------------------------------------------------------------------------


def _parse_pct(val: str | int | float | None) -> float:
    """Parse a percentage string like '33%' or '80%' to a float."""
    if val is None:
        return 0.0
    s = str(val).strip().replace(",", ".")
    m = re.search(r"([\d.]+)", s)
    return float(m.group(1)) if m else 0.0


def _parse_int(val: str | int | float | None) -> int:
    """Parse an integer from various formats like '3', '3 hop', '2 hop'."""
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else 0


def _parse_float(val: str | int | float | None) -> float:
    """Parse a float from strings like '5 tahun', '7.64', '63%'."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", ".")
    m = re.search(r"([\d.]+)", s)
    return float(m.group(1)) if m else 0.0


def _safe_get(data: dict, *keys, default=None):
    """Safely traverse nested dicts."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current


# ---------------------------------------------------------------------------
# Sub-function 1: _flatten_candidates
# ---------------------------------------------------------------------------


def _flatten_candidates(
    analysis_data: dict,
    input_candidates: dict | None = None,
    planner_candidates: dict | None = None,
) -> list[dict]:
    """Extract flat per-candidate data from nested Analysis output structure.

    Looks for candidate_data in order of priority:
    1. candidate_data field inside each analysis candidate
    2. Matching NIP in input_candidates dict
    3. Matching NIP in planner_candidates dict
    4. Matching NIP in xai_blueprint inside the analysis data
    5. Whatever fields are available in the analysis candidate itself
    """
    report = analysis_data.get("xai_justification_report", analysis_data)
    candidates_raw = _safe_get(report, "candidates", default=[])

    # Blueprint from the analysis data itself
    blueprint = analysis_data.get("xai_blueprint", {})
    blueprint_candidates = {
        c.get("nip"): c
        for c in _safe_get(blueprint, "candidates", default=[])
        if isinstance(c, dict)
    }

    # Use pre-loaded input candidates if provided, otherwise try from analysis_data
    if input_candidates is None:
        input_data = analysis_data.get("input", {})
        input_candidates = {
            c.get("nip"): c
            for c in (
                input_data.get("candidates", []) if isinstance(input_data, dict) else []
            )
            if isinstance(c, dict)
        }

    # Use pre-loaded planner candidates if provided
    if planner_candidates is None:
        planner_candidates = {}

    results: list[dict] = []
    for cand in candidates_raw:
        nip = cand.get("nip", "")
        nama = cand.get("nama", "")

        # --- final_justification ---
        fj = cand.get("final_justification", {})
        skor_domain_fit = float(fj.get("skor_domain_fit", 0))
        rekomendasi_sistem = fj.get("rekomendasi_sistem", "")

        # --- mining_results.explainability_metrics ---
        em = _safe_get(cand, "mining_results", "explainability_metrics", default={})
        fo = em.get("functional_overlap_score", {})
        functional_overlap_pct = _parse_pct(fo.get("overlap_percentage", "0%"))

        lowr = em.get("level_of_work_ratio", {})
        level_of_work_strategis = str(lowr.get("strategis_manajerial", "0%"))
        level_of_work_operasional = str(lowr.get("operasional_teknis", "0%"))

        ceg = em.get("cumulative_experience_gap", {})
        experience_gap_status = ceg.get("status", "")
        experience_gap_years = _parse_float(
            ceg.get("bidang_tugas_terkait_target", ceg.get("bidang_pengawasan_intern", "0"))
        )
        experience_gap_keterangan = ceg.get("keterangan", "")

        sp = em.get("structural_proximity", {})
        hop_distance = _parse_int(sp.get("hop_distance", "0"))
        career_trajectory = sp.get("career_trajectory", "")
        career_trajectory_keterangan = sp.get("keterangan", "")

        # --- candidate_data (priority: analysis → input → planner → blueprint → fallback) ---
        cd = cand.get("candidate_data", {})
        if not cd or not isinstance(cd, dict) or len(cd) < 3:
            if nip in input_candidates:
                cd = input_candidates[nip]
        if (not cd or not isinstance(cd, dict) or len(cd) < 3) and nip in planner_candidates:
            pc = planner_candidates[nip]
            if isinstance(pc, dict):
                cd_from_planner = pc.get("candidate_data", {})
                if cd_from_planner and isinstance(cd_from_planner, dict):
                    cd = cd_from_planner
        if not cd or not isinstance(cd, dict) or len(cd) < 3:
            if nip in blueprint_candidates:
                bc = blueprint_candidates[nip]
                cd_from_bp = bc.get("candidate_data", {})
                if cd_from_bp and isinstance(cd_from_bp, dict):
                    cd = cd_from_bp
        if not cd or not isinstance(cd, dict):
            cd = cand

        nama_lengkap = cd.get("nama_lengkap", nama)
        jabatan_saat_ini = cd.get("jabatan_nama", cd.get("jabatan_terakhir", ""))
        diklat_pim_level = cd.get("diklat_pim_level")
        nilai_kinerja = float(cd.get("nilai_kinerja", 0))
        nilai_kinerja_label = str(cd.get("nilai_kinerja_label", ""))
        nilai_potensi = float(cd.get("nilai_potensi", 0))
        nilai_mansoskul = float(cd.get("nilai_mansoskul", 0))
        pengalaman_struktural_tahun = _parse_float(
            cd.get("pengalaman_struktural_tahun", "0")
        )
        masa_kerja_tahun = float(cd.get("masa_kerja", cd.get("masa_kerja_total_tahun", 0)))
        riwayat_pendidikan = cd.get("riwayat_pendidikan", [])

        # alternatif_jabatan
        alt_list = cand.get("alternatif_jabatan", [])
        best_alt = None
        if alt_list:
            best_alt = {
                "nama_jabatan": alt_list[0].get("nama_jabatan", ""),
                "skor_kesesuaian": float(alt_list[0].get("skor_kesesuaian", 0)),
            }

        # sumber_data (for SG-VECTOR-RAG-ERROR detection)
        sumber_data = fj.get("sumber_data", [])

        results.append({
            "nip": nip,
            "nama": nama,
            "nama_lengkap": nama_lengkap,
            "jabatan_saat_ini": jabatan_saat_ini,
            "skor_domain_fit": skor_domain_fit,
            "functional_overlap_pct": functional_overlap_pct,
            "level_of_work_strategis": level_of_work_strategis,
            "level_of_work_operasional": level_of_work_operasional,
            "experience_gap_status": experience_gap_status,
            "experience_gap_years": experience_gap_years,
            "experience_gap_keterangan": experience_gap_keterangan,
            "hop_distance": hop_distance,
            "career_trajectory": career_trajectory,
            "career_trajectory_keterangan": career_trajectory_keterangan,
            "diklat_pim_level": diklat_pim_level,
            "nilai_kinerja": nilai_kinerja,
            "nilai_kinerja_label": nilai_kinerja_label,
            "nilai_potensi": nilai_potensi,
            "nilai_mansoskul": nilai_mansoskul,
            "pengalaman_struktural_tahun": pengalaman_struktural_tahun,
            "masa_kerja_tahun": masa_kerja_tahun,
            "riwayat_pendidikan": riwayat_pendidikan,
            "rekomendasi_sistem": rekomendasi_sistem,
            "sumber_data": sumber_data,
            "alternatif_jabatan_terbaik": best_alt,
            "_raw_candidate": cand,
        })

    return results


# ---------------------------------------------------------------------------
# Sub-function 2: _compute_skor_rekam_jejak
# ---------------------------------------------------------------------------


def _compute_skor_rekam_jejak(candidate: dict, aturan_penilaian: dict) -> dict:
    """Compute skor_rekam_jejak per candidate using the rubrik.

    Returns a dict matching SkorRekamJejakDetail structure.
    """

    # --- Extract bobot from aturan_penilaian ---
    rekam_jejak_kriteria = _safe_get(
        aturan_penilaian,
        "seleksi_terbuka_jpt", "rekam_jejak", "kriteria",
        default=[],
    )

    # Build bobot lookup by nama (case-insensitive)
    bobot_lookup: dict[str, str] = {}
    for k in rekam_jejak_kriteria:
        nama = k.get("nama", "").lower().strip()
        bobot = k.get("bobot", "0%")
        bobot_lookup[nama] = bobot

    # Default bobot if aturan_penilaian is missing
    default_bobot = {
        "jabatan": "5%",
        "pendidikan": "19%",
        "pelatihan": "19%",
        "disiplin": "19%",
        "skp": "19%",
        "integritas": "5%",
    }

    def _get_bobot(key_variants: list[str]) -> str:
        """Find bobot by trying multiple name variants."""
        for variant in key_variants:
            normalized = variant.lower().strip()
            normalized_alt = normalized.replace("°", "o")
            if normalized in bobot_lookup:
                return bobot_lookup[normalized]
            if normalized_alt in bobot_lookup:
                return bobot_lookup[normalized_alt]
        for variant in key_variants:
            normalized = variant.lower().strip()
            if normalized in default_bobot:
                return default_bobot[normalized]
            match = re.search(r"\(([^)]+)\)", normalized)
            if match:
                inner = match.group(1).strip()
                if inner in default_bobot:
                    return default_bobot[inner]
        return "0%"

    # --- Jabatan (5%) ---
    jabatan_str = str(candidate.get("jabatan_saat_ini", ""))
    raw = candidate.get("_raw_candidate", {})
    candidate_data = raw.get("candidate_data", {})
    current_eselon = candidate_data.get("current_eselon_id")

    skala_jabatan = 1
    basis_jabatan = jabatan_str if jabatan_str else "Tidak teridentifikasi"

    if current_eselon:
        eselon_str = str(current_eselon)
        if eselon_str in ("21", "21a"):
            skala_jabatan = 4
            basis_jabatan = f"Eselon II ({jabatan_str})"
        elif eselon_str in ("31", "32", "33", "34"):
            skala_jabatan = 3
            basis_jabatan = f"Eselon III ({jabatan_str})"
        elif eselon_str in ("41", "42", "43", "44"):
            skala_jabatan = 2
            basis_jabatan = f"Eselon IV ({jabatan_str})"

    jabatan_lower = jabatan_str.lower()
    if "eselon ii" in jabatan_lower or "jpt" in jabatan_lower:
        skala_jabatan = 4
        basis_jabatan = f"Eselon II ({jabatan_str})"
    elif "eselon iii" in jabatan_lower or "administrator" in jabatan_lower:
        skala_jabatan = 3
        basis_jabatan = f"Eselon III ({jabatan_str})"
    elif "ahli madya" in jabatan_lower:
        skala_jabatan = 2
        basis_jabatan = f"JF Ahli Madya ({jabatan_str})"

    bobot_jabatan = _get_bobot(["Jabatan", "jabatan"])
    bobot_pct_jabatan = _parse_pct(bobot_jabatan) / 100.0
    skor_kontribusi_jabatan = round((skala_jabatan / 4) * bobot_pct_jabatan * 100, 2)

    # --- Pendidikan (19%) ---
    riwayat_pendidikan = candidate.get("riwayat_pendidikan", [])
    skala_pendidikan = 1
    basis_pendidikan = "Tidak teridentifikasi"

    pendidikan_text = " ".join(str(p) for p in riwayat_pendidikan).lower()
    if any(x in pendidikan_text for x in ["s3", "doktor", "profesi"]):
        skala_pendidikan = 4
        basis_pendidikan = "S3/Profesi"
    elif any(x in pendidikan_text for x in ["s2", "magister", "master", "pasca sarjana", "pascasarjana"]):
        skala_pendidikan = 3
        basis_pendidikan = "S2"
    elif any(x in pendidikan_text for x in ["s1", "sarjana", "d-iv", "div"]):
        skala_pendidikan = 2
        basis_pendidikan = "S1/D-IV"
    else:
        jenjang_id = str(candidate_data.get("jenjang_pendidikan_id", ""))
        if jenjang_id in ("14", "15"):
            skala_pendidikan = 4
            basis_pendidikan = "S3/Profesi"
        elif jenjang_id == "13":
            skala_pendidikan = 3
            basis_pendidikan = "S2"
        elif jenjang_id in ("10", "11"):
            skala_pendidikan = 2
            basis_pendidikan = "S1/D-IV"

    bobot_pendidikan = _get_bobot(["Pendidikan", "pendidikan"])
    bobot_pct_pendidikan = _parse_pct(bobot_pendidikan) / 100.0
    skor_kontribusi_pendidikan = round((skala_pendidikan / 4) * bobot_pct_pendidikan * 100, 2)

    # --- Pelatihan (19%) ---
    diklat_pim = candidate.get("diklat_pim_level")
    skala_pelatihan = 1
    basis_pelatihan = "Tidak ada diklat PIM"
    caveat_pelatihan = None

    if diklat_pim is not None and str(diklat_pim).strip():
        diklat_str = str(diklat_pim).strip().upper()
        if "II" in diklat_str:
            skala_pelatihan = 4
            basis_pelatihan = f"PIM Tingkat II ({diklat_pim})"
        elif "III" in diklat_str:
            skala_pelatihan = 3
            basis_pelatihan = f"PIM Tingkat III ({diklat_pim})"
        elif "IV" in diklat_str:
            skala_pelatihan = 2
            basis_pelatihan = f"PIM Tingkat IV ({diklat_pim})"
        else:
            basis_pelatihan = str(diklat_pim)
    else:
        caveat_pelatihan = "Tidak ada data diklat PIM yang tersedia"

    bobot_pelatihan = _get_bobot(["Pelatihan Struktural/Fungsional", "Pelatihan", "pelatihan"])
    bobot_pct_pelatihan = _parse_pct(bobot_pelatihan) / 100.0
    skor_kontribusi_pelatihan = round((skala_pelatihan / 4) * bobot_pct_pelatihan * 100, 2)

    # --- Disiplin (19%) ---
    skala_disiplin = 4
    basis_disiplin = "Asumsi: tidak ada catatan hukuman disiplin"
    caveat_disiplin = "Asumsi: data disiplin tidak tersedia"

    bobot_disiplin = _get_bobot(["Disiplin", "disiplin"])
    bobot_pct_disiplin = _parse_pct(bobot_disiplin) / 100.0
    skor_kontribusi_disiplin = round((skala_disiplin / 4) * bobot_pct_disiplin * 100, 2)

    # --- SKP (19%) ---
    nilai_kinerja_raw = candidate.get("nilai_kinerja", 0)
    skala_skp = int(nilai_kinerja_raw) if 1 <= int(nilai_kinerja_raw) <= 4 else 1
    basis_skp = f"SKP = {nilai_kinerja_raw} ({candidate.get('nilai_kinerja_label', '')})"

    bobot_skp = _get_bobot(["Penilaian Kinerja (SKP)", "SKP", "skp", "Penilaian Kinerja"])
    bobot_pct_skp = _parse_pct(bobot_skp) / 100.0
    skor_kontribusi_skp = round((skala_skp / 4) * bobot_pct_skp * 100, 2)

    # --- Integritas (5%) ---
    nilai_mansoskul = candidate.get("nilai_mansoskul", 0)
    if nilai_mansoskul <= 60:
        skala_integritas = 1
    elif nilai_mansoskul <= 75:
        skala_integritas = 2
    elif nilai_mansoskul <= 90:
        skala_integritas = 3
    else:
        skala_integritas = 4
    basis_integritas = f"Mansoskul = {nilai_mansoskul}"

    bobot_integritas = _get_bobot(
        ["Integritas dan Moralitas (Penilaian Kepemimpinan 360o)", "Integritas", "integritas"]
    )
    bobot_pct_integritas = _parse_pct(bobot_integritas) / 100.0
    skor_kontribusi_integritas = round((skala_integritas / 4) * bobot_pct_integritas * 100, 2)

    # --- Total ---
    total_rekam_jejak = round(
        skor_kontribusi_jabatan
        + skor_kontribusi_pendidikan
        + skor_kontribusi_pelatihan
        + skor_kontribusi_disiplin
        + skor_kontribusi_skp
        + skor_kontribusi_integritas,
        2,
    )

    catatan = (
        f"Skala menggunakan rubrik 1-4; disiplin diasumsikan 4 (tidak ada data hukuman). "
        f"Pelatihan: {'diklat PIM ' + str(diklat_pim) if diklat_pim else 'tidak tersedia'}."
    )

    return {
        "jabatan": {
            "skala": skala_jabatan,
            "basis": basis_jabatan,
            "bobot": bobot_jabatan,
            "skor_kontribusi": skor_kontribusi_jabatan,
            "caveat": None,
        },
        "pendidikan": {
            "skala": skala_pendidikan,
            "basis": basis_pendidikan,
            "bobot": bobot_pendidikan,
            "skor_kontribusi": skor_kontribusi_pendidikan,
            "caveat": None,
        },
        "pelatihan": {
            "skala": skala_pelatihan,
            "basis": basis_pelatihan,
            "bobot": bobot_pelatihan,
            "skor_kontribusi": skor_kontribusi_pelatihan,
            "caveat": caveat_pelatihan,
        },
        "disiplin": {
            "skala": skala_disiplin,
            "basis": basis_disiplin,
            "bobot": bobot_disiplin,
            "skor_kontribusi": skor_kontribusi_disiplin,
            "caveat": caveat_disiplin,
        },
        "skp": {
            "skala": skala_skp,
            "basis": basis_skp,
            "bobot": bobot_skp,
            "skor_kontribusi": skor_kontribusi_skp,
            "caveat": None,
        },
        "integritas": {
            "skala": skala_integritas,
            "basis": basis_integritas,
            "bobot": bobot_integritas,
            "skor_kontribusi": skor_kontribusi_integritas,
            "caveat": None,
        },
        "total_rekam_jejak": total_rekam_jejak,
        "catatan": catatan,
    }


# ---------------------------------------------------------------------------
# Sub-function 3: _compute_skor_komposit_parsial
# ---------------------------------------------------------------------------


def _compute_skor_komposit_parsial(
    skor_rekam_jejak_total: float, nilai_mansoskul: float
) -> dict:
    """Compute partial composite score (only 45% of total weight available).

    Components:
    - Rekam Jejak (20%): total_rekam_jejak on 0-100 scale → (total/100) × 20
    - Assessment Center proxy (25%): nilai_mansoskul on 0-100 scale → (mansoskul/100) × 25
    - Penulisan Makalah (20%): belum_dinilai
    - Wawancara & Presentasi (35%): belum_dinilai
    """
    kontribusi_rj = round((skor_rekam_jejak_total / 100) * 20, 2)
    kontribusi_ac = round((nilai_mansoskul / 100) * 25, 2)
    total_parsial = round(kontribusi_rj + kontribusi_ac, 2)

    catatan = (
        f"Skor komposit parsial hanya mencakup 45% bobot seleksi "
        f"(rekam jejak 20% + assessment center proxy 25%). "
        f"Komponen penulisan makalah (20%) dan wawancara (35%) belum dinilai."
    )

    return {
        "komponen_rekam_jejak": {
            "nilai": round(skor_rekam_jejak_total, 2),
            "bobot_dalam_seleksi": "20%",
            "kontribusi_ke_total": kontribusi_rj,
        },
        "proxy_assessment_center": {
            "nilai": round(nilai_mansoskul, 2),
            "basis": "Manajemen Sumber Daya Manusia & Sosial Budaya (mansoskul) sebagai proxy",
            "bobot_dalam_seleksi": "25%",
            "kontribusi_ke_total": kontribusi_ac,
        },
        "penulisan_makalah": {
            "nilai": "belum_dinilai",
            "bobot_dalam_seleksi": "20%",
            "kontribusi_ke_total": None,
        },
        "wawancara_presentasi": {
            "nilai": "belum_dinilai",
            "bobot_dalam_seleksi": "35%",
            "kontribusi_ke_total": None,
        },
        "total_parsial": total_parsial,
        "pct_bobot_terhitung": "45%",
        "catatan": catatan,
    }


# ---------------------------------------------------------------------------
# Sub-function 4: _sort_and_rank
# ---------------------------------------------------------------------------


def _sort_and_rank(matrix_rows: list[dict]) -> list[dict]:
    """Sort by skor_domain_fit desc, then skor_rekam_jejak desc, then nilai_potensi desc.

    Assigns 1-based rank.
    """
    for row in matrix_rows:
        srj = row.get("skor_rekam_jejak", {})
        row["_total_rj"] = srj.get("total_rekam_jejak", 0) if isinstance(srj, dict) else 0

    sorted_rows = sorted(
        matrix_rows,
        key=lambda r: (
            -r.get("skor_domain_fit", 0),
            -r.get("_total_rj", 0),
            -r.get("nilai_potensi", 0),
        ),
    )

    for i, row in enumerate(sorted_rows):
        row["rank"] = i + 1
        row.pop("_total_rj", None)

    return sorted_rows


# ---------------------------------------------------------------------------
# Sub-function 5: _detect_systemic_gaps
# ---------------------------------------------------------------------------


def _detect_systemic_gaps(matrix_rows: list[dict], analysis_data: dict) -> list[dict]:
    """Threshold-based detection of systemic gaps across all candidates."""
    gaps: list[dict] = []
    all_nips = [r.get("nip", "") for r in matrix_rows]

    if not matrix_rows:
        return gaps

    # SG-DOMAIN-MISMATCH: all skor_domain_fit < 50
    if all(r.get("skor_domain_fit", 0) < 50 for r in matrix_rows):
        gaps.append({
            "gap_id": "SG-DOMAIN-MISMATCH",
            "dimensi": "Domain Fit",
            "deskripsi": (
                "Semua kandidat memiliki skor domain fit di bawah 50%, "
                "mengindikasikan ketidaksesuaian sistemik antara profil kandidat "
                "dan kebutuhan jabatan target."
            ),
            "kandidat_terdampak": all_nips,
            "severity": "critical",
            "implikasi": "Proses seleksi mungkin perlu diperluas ke pool kandidat tambahan.",
        })

    # SG-FUNCTIONAL-GAP: all functional_overlap_pct < 33%
    if all(r.get("functional_overlap_pct", 0) < 33 for r in matrix_rows):
        gaps.append({
            "gap_id": "SG-FUNCTIONAL-GAP",
            "dimensi": "Functional Overlap",
            "deskripsi": (
                "Semua kandidat memiliki overlap fungsional di bawah 33% "
                "dengan jabatan target, menunjukkan kesenjangan kompetensi "
                "yang signifikan secara sistemik."
            ),
            "kandidat_terdampak": all_nips,
            "severity": "critical",
            "implikasi": "Perlu dipertimbangkan pelatihan pra-jabatan yang intensif.",
        })

    # SG-STRUCTURAL-DISTANCE: all hop_distance > 2
    if all(r.get("hop_distance", 0) > 2 for r in matrix_rows):
        gaps.append({
            "gap_id": "SG-STRUCTURAL-DISTANCE",
            "dimensi": "Structural Proximity",
            "deskripsi": (
                "Semua kandidat berada lebih dari 2 hop dari jabatan target "
                "dalam struktur organisasi, mengindikasikan jarak struktural "
                "yang signifikan."
            ),
            "kandidat_terdampak": all_nips,
            "severity": "major",
            "implikasi": "Kandidat memerlukan transisi karir yang lebih panjang menuju jabatan target.",
        })

    # SG-DIKLAT-GAP: all diklat_pim_level null/empty
    if all(
        r.get("diklat_pim_level") is None or str(r.get("diklat_pim_level", "")).strip() == ""
        for r in matrix_rows
    ):
        gaps.append({
            "gap_id": "SG-DIKLAT-GAP",
            "dimensi": "Diklat Kepemimpinan",
            "deskripsi": (
                "Tidak ada kandidat yang memiliki diklat kepemimpinan PIM, "
                "yang merupakan prasyarat preferensi untuk JPT."
            ),
            "kandidat_terdampak": all_nips,
            "severity": "warning",
            "implikasi": "Semua kandidat perlu mengikuti diklat kepemimpinan sebelum pengangkatan.",
        })

    # SG-VECTOR-RAG-ERROR: all candidates have vector_rag error in sumber_data
    report = analysis_data.get("xai_justification_report", analysis_data)
    candidates_raw = _safe_get(report, "candidates", default=[])
    if candidates_raw:
        all_have_error = True
        for cand in candidates_raw:
            fj = cand.get("final_justification", {})
            sumber = fj.get("sumber_data", [])
            has_vector_error = any(
                isinstance(s, str) and "vector_rag" in s.lower() and "error" in s.lower()
                for s in sumber
            )
            if not has_vector_error:
                all_have_error = False
                break
        if all_have_error:
            gaps.append({
                "gap_id": "SG-VECTOR-RAG-ERROR",
                "dimensi": "Infrastructure",
                "deskripsi": (
                    "Semua kandidat mengalami error pada vector RAG retrieval, "
                    "mengindikasikan masalah infrastruktur yang mempengaruhi "
                    "kualitas analisis."
                ),
                "kandidat_terdampak": all_nips,
                "severity": "infrastructure_warning",
                "implikasi": "Hasil analisis mungkin tidak lengkap; perlu verifikasi ulang setelah infrastruktur pulih.",
            })

    return gaps


# ---------------------------------------------------------------------------
# Sub-function 6: _validate_consistency
# ---------------------------------------------------------------------------


def _validate_consistency(analysis_data: dict) -> list[dict]:
    """Cross-candidate consistency checks."""
    flags: list[dict] = []

    blueprint = analysis_data.get("xai_blueprint", {})
    report = analysis_data.get("xai_justification_report", analysis_data)
    candidates_raw = _safe_get(report, "candidates", default=[])
    nips = [c.get("nip", "") for c in candidates_raw]

    if len(candidates_raw) < 2:
        return flags

    # CF-FUNGSI-UTAMA: fungsi_utama must be identical across candidates
    blueprint_candidates = _safe_get(blueprint, "candidates", default=[])
    if blueprint_candidates:
        fungsi_sets = []
        for bc in blueprint_candidates:
            rc = bc.get("retrieved_context", {})
            fu = tuple(sorted(rc.get("fungsi_utama", [])))
            fungsi_sets.append(fu)

        if len(set(fungsi_sets)) > 1:
            flags.append({
                "flag_id": "CF-FUNGSI-UTAMA",
                "dimensi": "Fungsi Utama",
                "deskripsi": (
                    "Fungsi utama jabatan target tidak konsisten antar kandidat. "
                    "Ini seharusnya identik karena semua kandidat menargetkan jabatan yang sama."
                ),
                "kandidat_terdampak": nips,
                "dampak": "Dapat menyebabkan perbandingan yang tidak adil antar kandidat.",
            })

    # CF-SYARAT-PENGALAMAN: syarat_pengalaman must be identical
    if blueprint_candidates:
        syarat_sets = []
        for bc in blueprint_candidates:
            rc = bc.get("retrieved_context", {})
            sp = str(rc.get("syarat_pengalaman", ""))
            syarat_sets.append(sp)

        if len(set(syarat_sets)) > 1:
            flags.append({
                "flag_id": "CF-SYARAT-PENGALAMAN",
                "dimensi": "Syarat Pengalaman",
                "deskripsi": (
                    "Syarat pengalaman tidak konsisten antar kandidat. "
                    "Semua kandidat seharusnya memiliki syarat pengalaman yang sama."
                ),
                "kandidat_terdampak": nips,
                "dampak": "Perbedaan syarat pengalaman dapat menyebabkan penilaian yang tidak konsisten.",
            })

    # CF-EXPERIENCE-GAP-METHOD: experience gap status consistency
    gap_methods = []
    for cand in candidates_raw:
        em = _safe_get(cand, "mining_results", "explainability_metrics", default={})
        ceg = em.get("cumulative_experience_gap", {})
        status = ceg.get("status", "Unknown")
        gap_methods.append(status)

    unique_statuses = set(gap_methods)
    if len(unique_statuses) == 0 or any(s == "Unknown" for s in unique_statuses):
        flags.append({
            "flag_id": "CF-EXPERIENCE-GAP-METHOD",
            "dimensi": "Metodologi Experience Gap",
            "deskripsi": (
                "Terdapat inkonsistensi dalam metode perhitungan experience gap "
                "antara kandidat. Beberapa kandidat tidak memiliki status yang jelas."
            ),
            "kandidat_terdampak": nips,
            "dampak": "Perbandingan experience gap antar kandidat mungkin tidak apple-to-apple.",
        })

    return flags


# ---------------------------------------------------------------------------
# Sub-function 7: _check_regulatory_compliance
# ---------------------------------------------------------------------------


def _check_regulatory_compliance(
    matrix_rows: list[dict], aturan_penilaian: dict
) -> dict[str, dict]:
    """Per-candidate regulatory compliance check against persyaratan_jpt."""

    persyaratan = _safe_get(
        aturan_penilaian, "seleksi_terbuka_jpt", "persyaratan_jpt", default={}
    )

    results: dict[str, dict] = {}

    for row in matrix_rows:
        nip = row.get("nip", "")
        nama = row.get("nama_lengkap", row.get("nama", ""))
        raw = row.get("_raw_candidate", {})
        cd = raw.get("candidate_data", {})

        # --- Hard requirements ---
        syarat_wajib: dict[str, str] = {}

        # Status kepegawaian: PNS
        is_eligible = cd.get("is_eligible")
        if is_eligible is True:
            syarat_wajib["status_kepegawaian_PNS"] = "Terpenuhi"
        elif is_eligible is False:
            syarat_wajib["status_kepegawaian_PNS"] = "Tidak Terpenuhi"
        else:
            syarat_wajib["status_kepegawaian_PNS"] = "BELUM_TERVERIFIKASI"

        # Kualifikasi pendidikan: minimal S1
        riwayat_pendidikan = cd.get("riwayat_pendidikan", row.get("riwayat_pendidikan", []))
        pendidikan_text = " ".join(str(p) for p in riwayat_pendidikan).lower() if riwayat_pendidikan else ""
        if any(x in pendidikan_text for x in ["s1", "sarjana", "s2", "magister", "s3", "doktor", "d-iv", "div", "profesi"]):
            syarat_wajib["kualifikasi_pendidikan_S1"] = "Terpenuhi"
        elif pendidikan_text:
            syarat_wajib["kualifikasi_pendidikan_S1"] = "Tidak Terpenuhi"
        else:
            syarat_wajib["kualifikasi_pendidikan_S1"] = "BELUM_TERVERIFIKASI"

        # Pengalaman jabatan: ≥2 tahun
        pengalaman_struktural = row.get("pengalaman_struktural_tahun", 0)
        if pengalaman_struktural >= 2:
            syarat_wajib["pengalaman_jabatan_2_tahun"] = "Terpenuhi"
        elif pengalaman_struktural > 0:
            syarat_wajib["pengalaman_jabatan_2_tahun"] = "Tidak Terpenuhi"
        else:
            syarat_wajib["pengalaman_jabatan_2_tahun"] = "BELUM_TERVERIFIKASI"

        # --- Preferences ---
        syarat_preferensi: dict[str, str] = {}

        # Diklat PIM III
        diklat = row.get("diklat_pim_level")
        if diklat is not None and str(diklat).strip():
            diklat_str = str(diklat).strip().upper()
            if "III" in diklat_str or "II" in diklat_str:
                syarat_preferensi["diklat_PIM_III"] = "Terpenuhi"
            elif "IV" in diklat_str:
                syarat_preferensi["diklat_PIM_III"] = "Terpenuhi Sebagian (PIM IV)"
            else:
                syarat_preferensi["diklat_PIM_III"] = "Tidak Terpenuhi"
        else:
            syarat_preferensi["diklat_PIM_III"] = "Tidak Terpenuhi"

        # Pangkat IV/a
        pangkat_minimal = str(persyaratan.get("pangkat_minimal", ""))
        if "iv" in pangkat_minimal.lower() or "pembina" in pangkat_minimal.lower():
            current_eselon = str(cd.get("current_eselon_id", ""))
            if current_eselon in ("21", "31", "32", "33", "34"):
                syarat_preferensi["pangkat_IV_a"] = "Terpenuhi"
            elif current_eselon:
                syarat_preferensi["pangkat_IV_a"] = "Tidak Terpenuhi"
            else:
                syarat_preferensi["pangkat_IV_a"] = "BELUM_TERVERIFIKASI"
        else:
            syarat_preferensi["pangkat_IV_a"] = "BELUM_TERVERIFIKASI"

        # Determine overall status
        has_unfulfilled_wajib = any(
            v == "Tidak Terpenuhi" for v in syarat_wajib.values()
        )
        has_unverified_wajib = any(
            v == "BELUM_TERVERIFIKASI" for v in syarat_wajib.values()
        )

        if has_unfulfilled_wajib:
            overall_status = "Tidak Memenuhi"
        elif has_unverified_wajib:
            overall_status = "Memenuhi Sebagian"
        else:
            overall_status = "Memenuhi"

        results[nip] = {
            "nama": nama,
            "syarat_wajib": syarat_wajib,
            "syarat_preferensi": syarat_preferensi,
            "hard_disqualification": None,
            "overall_formal_status": overall_status,
            "catatan": (
                f"Verifikasi berdasarkan data yang tersedia; "
                f"{len([v for v in syarat_wajib.values() if v == 'BELUM_TERVERIFIKASI'])} "
                f"syarat belum terverifikasi."
            ),
        }

    return results


# ---------------------------------------------------------------------------
# Main tool: comparative_matrix_aggregator (in-memory variant)
# ---------------------------------------------------------------------------


@tool
def comparative_matrix_aggregator(analysis_output: str) -> str:
    """Compute all deterministic fields for synthesis cross-candidate comparison.

    In the ai-bpom architecture, data flows in-memory between agents.
    This tool receives analysis output as a JSON string and automatically
    extracts blueprint context from it.

    Returns a JSON string with: comparison_matrix, systemic_gaps,
    consistency_flags, regulatory_compliance_summary, ranking_metadata.

    All numeric computations are deterministic — no LLM calls.

    Args:
        analysis_output: JSON string containing the Analysis Agent output,
            which should include xai_justification_report and optionally
            xai_blueprint or blueprint_context.
    """
    # Parse the analysis output JSON
    analysis_data = json.loads(analysis_output) if isinstance(analysis_output, str) else analysis_output

    # Extract aturan_penilaian from blueprint context
    # Priority: blueprint_context field > xai_blueprint field > analysis report
    blueprint_context = analysis_data.get("blueprint_context", analysis_data.get("xai_blueprint", {}))
    planner_aturan = _safe_get(blueprint_context, "aturan_penilaian", default={})

    report = analysis_data.get("xai_justification_report", analysis_data)
    analysis_aturan = _safe_get(report, "aturan_penilaian", default={})
    if not analysis_aturan:
        # Try nested structure
        analysis_aturan = _safe_get(analysis_data, "xai_justification_report", "aturan_penilaian", default={})

    # Use planner aturan_penilaian if available (more complete), fallback to analysis
    aturan_penilaian = planner_aturan if planner_aturan else analysis_aturan

    # Build input_candidates and planner_candidates from context
    input_candidates = {}
    planner_candidates = {}

    input_data = analysis_data.get("input", {})
    if isinstance(input_data, dict):
        input_candidates = {
            c.get("nip"): c
            for c in input_data.get("candidates", [])
            if isinstance(c, dict)
        }

    blueprint_candidates_list = _safe_get(blueprint_context, "candidates", default=[])
    planner_candidates = {
        c.get("nip"): c
        for c in blueprint_candidates_list
        if isinstance(c, dict)
    }

    # Step 1: Flatten candidates
    flat_candidates = _flatten_candidates(analysis_data, input_candidates, planner_candidates)

    # Step 2: Compute skor_rekam_jejak for each candidate
    matrix_rows: list[dict] = []
    for cand in flat_candidates:
        srj = _compute_skor_rekam_jejak(cand, aturan_penilaian)

        # Step 3: Compute skor_komposite_parsial
        skp = _compute_skor_komposit_parsial(
            srj["total_rekam_jejak"], cand["nilai_mansoskul"]
        )

        # Build experience gap display
        eg_status = cand.get("experience_gap_status", "Unknown")
        eg_years = cand.get("experience_gap_years", 0)
        if "defisit" in eg_status.lower() or "deficit" in eg_status.lower():
            eg_display = f"{eg_years} tahun"
        elif "surplus" in eg_status.lower() or "terpenuhi" in eg_status.lower():
            eg_display = "-"
        else:
            eg_display = "-"

        row = {
            "nip": cand["nip"],
            "nama": cand["nama"],
            "nama_lengkap": cand["nama_lengkap"],
            "jabatan_saat_ini": cand["jabatan_saat_ini"],
            "skor_domain_fit": cand["skor_domain_fit"],
            "functional_overlap_pct": cand["functional_overlap_pct"],
            "level_of_work_strategis": cand["level_of_work_strategis"],
            "level_of_work_operasional": cand["level_of_work_operasional"],
            "experience_gap_status": eg_status,
            "experience_gap_defisit_tahun": eg_display,
            "experience_gap_keterangan": cand.get("experience_gap_keterangan", ""),
            "hop_distance": cand["hop_distance"],
            "career_trajectory": cand["career_trajectory"],
            "career_trajectory_keterangan": cand.get("career_trajectory_keterangan", ""),
            "diklat_pim_level": cand["diklat_pim_level"],
            "nilai_kinerja": cand["nilai_kinerja"],
            "nilai_kinerja_label": cand["nilai_kinerja_label"],
            "nilai_potensi": cand["nilai_potensi"],
            "nilai_mansoskul": cand["nilai_mansoskul"],
            "pengalaman_struktural_tahun": cand["pengalaman_struktural_tahun"],
            "masa_kerja_tahun": cand["masa_kerja_tahun"],
            "riwayat_pendidikan": cand["riwayat_pendidikan"],
            "rekomendasi_sistem": cand["rekomendasi_sistem"],
            "skor_rekam_jejak": srj,
            "skor_komposit_parsial": skp,
            "alternatif_jabatan_terbaik": cand.get("alternatif_jabatan_terbaik"),
            "_raw_candidate": cand.get("_raw_candidate"),
        }
        matrix_rows.append(row)

    # Step 4: Sort and rank
    matrix_rows = _sort_and_rank(matrix_rows)

    # Step 5: Detect systemic gaps
    systemic_gaps = _detect_systemic_gaps(matrix_rows, analysis_data)

    # Step 6: Validate consistency
    consistency_flags = _validate_consistency(analysis_data)

    # Step 7: Check regulatory compliance
    regulatory_compliance = _check_regulatory_compliance(matrix_rows, aturan_penilaian)

    # Clean up internal fields before returning
    for row in matrix_rows:
        row.pop("_raw_candidate", None)

    result = {
        "comparison_matrix": matrix_rows,
        "systemic_gaps": systemic_gaps,
        "consistency_flags": consistency_flags,
        "regulatory_compliance_summary": regulatory_compliance,
        "ranking_metadata": {
            "sort_criteria": "skor_domain_fit DESC, skor_rekam_jejak DESC, nilai_potensi DESC",
            "total_candidates": len(matrix_rows),
            "rank_1_nip": matrix_rows[0]["nip"] if matrix_rows else None,
            "rank_1_nama": matrix_rows[0]["nama"] if matrix_rows else None,
        },
    }

    return json.dumps(result, ensure_ascii=False, default=str)