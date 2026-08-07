"""Stage 2 schema-filtering UQ — multiple-interpretation via semantic clustering.

Stage 1 (``semantic_disambiguation/uq.py``) mengukur ketidakpastian
*interpretasi pertanyaan* (M-sampling rewrite NL→NL → embed → cluster →
entropy). Stage 2 di sini mengukur ketidakpastian *interpretasi skema*: untuk
satu pertanyaan + satu himpunan kandidat skema yang diretrieval SEKALI, LLM
bisa memetakan konsep-konsep pertanyaan ke tabel/kolom yang berbeda-beda
(mis. "senior" → masa_kerja / jabatan / golongan). Bila pemetaan bercabang,
sistem minta klarifikasi sebelum men-generate SQL.

Yang di-cluster adalah **output reasoning LLM** (teks interpretasi skema),
BUKAN schema candidate maupun embedding tabel/kolom secara langsung. Ini
mengikuti desain dosen (terinspirasi AmbiSQL): entropy benar-benar mengukur
ketidakpastian interpretasi, bukan sekadar kedekatan embedding antar kolom.

Mekanisme:

1. Caller (pipeline) menjalankan retrieval SEKALI, lalu meng-generate M
   interpretasi skema via LLM @ temperature > 0 (paralel) → M string teks.
2. Tiap interpretasi di-embed (model NL, sama keluarga dengan kalibrasi Stage 1)
   lalu di-cluster single-link cosine.
3. Sinyal UQ dihitung lewat ``semantic_disambiguation/uq.py.compute_uq_signal``
   — **tidak ada duplikasi matematika**; kita menyuplai teks interpretasi
   sebagai "samples" dan embedding NL-nya sebagai "embeddings".

Threshold (τ_cluster, τ_U) berasal dari config dan masih STALE (lihat
``core/config.py``) — kalibrasi final di luar scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..semantic_disambiguation.uq import compute_uq_signal


@dataclass(frozen=True)
class SchemaInterpretationCluster:
    """Satu cluster interpretasi-skema (hasil semantic clustering)."""

    text: str  # representative (interpretasi pertama di cluster)
    support: int  # jumlah sample di cluster ini
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchemaUQResult:
    enabled: bool
    h_norm: float
    tau_u: float
    m_samples: int
    valid_samples: int
    unique_clusters: int
    majority_ratio: float
    is_uncertain: bool
    representative_text: str
    interpretations: list[SchemaInterpretationCluster] = field(default_factory=list)
    n_error: int = 0
    # Degraded = UQ tidak bisa diukur (semua sample gagal / embedding gagal).
    # Kita FAIL-OPEN (is_uncertain=False, mirror Stage 1) supaya pipeline lanjut
    # tanpa klarifikasi kosong/rusak, TAPI tandai eksplisit di sini agar jejak
    # bisa membedakan "konvergen" dari "tidak terukur". BUKAN downgrade diam-diam.
    degraded: bool = False
    degraded_reason: str = ""

    def to_metadata(self) -> dict[str, object]:
        """Bentuk ringkas untuk disurface ke ``pipeline_trace`` stage."""
        verdict = "confident"
        if self.degraded:
            verdict = "degraded"
        elif self.is_uncertain:
            verdict = "uncertain"
        return {
            "h_norm": round(float(self.h_norm), 4),
            "tau_u": round(float(self.tau_u), 4),
            "verdict": verdict,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "m_samples": self.m_samples,
            "valid_samples": self.valid_samples,
            "unique_clusters": self.unique_clusters,
            "majority_ratio": round(float(self.majority_ratio), 4),
            "n_error": self.n_error,
            "representative_interpretation": self.representative_text,
            "interpretations": [
                {"text": it.text, "support": it.support}
                for it in self.interpretations
            ],
        }


def _neutral_result(
    *,
    enabled: bool,
    tau_u: float,
    m_total: int,
    valid: int,
    n_error: int,
    rep: str = "",
) -> SchemaUQResult:
    interpretations: list[SchemaInterpretationCluster] = []
    if valid and rep:
        interpretations = [
            SchemaInterpretationCluster(text=rep, support=valid, members=(rep,))
        ]
    # Jalur 0/1 sampel valid bisa terjadi karena sebagian besar sampel GAGAL
    # (mis. temperature terlalu tinggi → JSON rusak). Itu BUKAN "confident"
    # sejati — tandai degraded agar trace jujur menjelaskan penyebabnya
    # (fail-open: tetap tidak memicu klarifikasi, tapi alasannya transparan).
    enough_valid = valid * 2 >= m_total
    degraded = not enough_valid
    degraded_reason = (
        f"hanya {valid}/{m_total} sampel valid (<50%)" if degraded else ""
    )
    return SchemaUQResult(
        enabled=enabled,
        h_norm=0.0,
        tau_u=tau_u,
        m_samples=m_total,
        valid_samples=valid,
        unique_clusters=1 if valid else 0,
        majority_ratio=1.0 if valid else 0.0,
        is_uncertain=False,
        representative_text=rep,
        interpretations=interpretations,
        n_error=n_error,
        degraded=degraded,
        degraded_reason=degraded_reason,
    )


def compute_schema_uq(
    samples: list[str],
    embeddings: list[list[float]],
    *,
    m_total: int,
    tau_cluster: float,
    tau_u: float,
    enabled: bool,
) -> SchemaUQResult:
    """Hitung sinyal UQ atas M interpretasi-skema (teks) + embedding-nya.

    ``samples`` dan ``embeddings`` harus sejajar. Interpretasi kosong / tanpa
    embedding diperlakukan sebagai ERROR — tidak ikut clustering tetapi tetap
    dihitung di ``m_total`` untuk fingerprint entropy (mirror Stage 1).
    """
    valid_pairs = [
        (s, e) for s, e in zip(samples, embeddings) if s and s.strip() and e
    ]
    valid_samples = [s for s, _ in valid_pairs]
    valid_embeddings = [e for _, e in valid_pairs]
    n_valid = len(valid_samples)
    n_error = max(0, m_total - n_valid)

    if n_valid <= 1:
        # 0 atau 1 interpretasi valid → tidak ada percabangan untuk diukur.
        return _neutral_result(
            enabled=enabled,
            tau_u=tau_u,
            m_total=m_total,
            valid=n_valid,
            n_error=n_error,
            rep=valid_samples[0] if n_valid == 1 else "",
        )

    signal = compute_uq_signal(
        valid_samples,
        valid_embeddings,
        m_total=m_total,
        tau_cluster=tau_cluster,
    )

    h_norm = float(signal["h_norm"])
    # ``fingerprints`` di signal = cluster label per sample valid + filler
    # "ERROR". Slice ke n_valid untuk memetakan cluster → sample.
    cluster_labels = list(signal["fingerprints"])[:n_valid]

    clusters: dict[str, list[int]] = {}
    for idx, label in enumerate(cluster_labels):
        clusters.setdefault(label, []).append(idx)

    interpretations: list[SchemaInterpretationCluster] = []
    for _label, indices in clusters.items():
        members = tuple(valid_samples[i] for i in indices)
        interpretations.append(
            SchemaInterpretationCluster(
                text=members[0],
                support=len(indices),
                members=members,
            )
        )
    interpretations.sort(key=lambda it: it.support, reverse=True)

    representative_text = (
        str(signal.get("majority_cluster_representative") or "")
        or (interpretations[0].text if interpretations else "")
    )

    # --- Verdict: gate by COUNT of distinct interpretations, not entropy. ---
    # Dengan normalisasi H/log(M) (M=10), entropi MAKSIMUM untuk 2 cluster
    # (split 5:5) hanya ln2/ln10 ≈ 0.30 — sehingga τ_U "stale" (0.40) secara
    # struktural MUSTAHIL tercapai untuk percabangan 2-interpretasi. Aturan yang
    # dipakai lebih sederhana & interpretable: bila LLM menghasilkan ≥2
    # interpretasi-skema yang BERBEDA → minta klarifikasi.
    #
    # ``len(clusters)`` sudah mengecualikan cluster ERROR (cluster_labels =
    # fingerprints[:n_valid]), jadi sampel rusak tidak memicu percabangan palsu.
    # h_norm / τ_U tetap dilaporkan sebagai diagnostik kontinu di trace.
    n_clusters = len(clusters)
    # Guard: butuh mayoritas sampel valid (≥50%). Saat sebagian besar sampel
    # gagal (mis. temperature terlalu tinggi → JSON rusak), keputusan tidak
    # boleh diambil dari segelintir sampel. Tandai degraded → fail-open.
    enough_valid = n_valid * 2 >= m_total
    degraded = not enough_valid
    degraded_reason = (
        f"hanya {n_valid}/{m_total} sampel valid (<50%)" if degraded else ""
    )
    is_uncertain = enabled and enough_valid and n_clusters >= 2

    return SchemaUQResult(
        enabled=enabled,
        h_norm=h_norm,
        tau_u=tau_u,
        m_samples=m_total,
        valid_samples=n_valid,
        unique_clusters=n_clusters,
        majority_ratio=float(signal["majority_ratio"]),
        is_uncertain=is_uncertain,
        representative_text=representative_text,
        interpretations=interpretations,
        n_error=n_error,
        degraded=degraded,
        degraded_reason=degraded_reason,
    )


__all__ = [
    "SchemaInterpretationCluster",
    "SchemaUQResult",
    "compute_schema_uq",
]
