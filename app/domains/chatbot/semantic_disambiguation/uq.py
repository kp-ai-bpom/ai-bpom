"""Uncertainty Quantification (UQ) helpers untuk Stage 1 ambiguity detection.

Helpers ini diport dari ``tests/stage1/scripts/lampiran_c_uq.py`` (kalibrasi
Stage 1 Skripsi S1 Ariq Hikari Hidayat). Konstanta produksi:

    M_SAMPLING   = 10   (dinaikkan dari 5 → 10 untuk menurunkan noise floor
                         sampling; gap signal-vs-noise lebih lebar sehingga
                         threshold lebih defensible)
    T_SAMPLING   = 1.0
    TAU_CLUSTER  = 0.80   (single-link cosine semantic-equivalent boundary)
    TAU_U        = 0.40   (STALE — perlu re-kalibrasi pada M=10. Nilai 0.40
                          adalah round-up dari celah bimodal 0.311–0.418 di
                          kalibrasi M=5 sebelumnya. Setelah re-kalibrasi M=10,
                          τ_U diharapkan turun ke ~0.25–0.30.)

Mekanisme ringkas:

1. Sample M=10 interpretasi dari pertanyaan user via LLM @ T=1.0 (paralel).
2. Embed setiap interpretasi.
3. Cluster single-link cosine pada threshold TAU_CLUSTER.
4. Hitung normalized entropy (H_norm) atas distribusi cluster id.
5. Bila ``H_norm > TAU_U`` → ambiguous (pertanyaan punya >1 interpretasi
   yang sama-sama masuk akal di skema). Bila ``H_norm <= TAU_U`` →
   non-ambiguous (rewriter konvergen ke satu interpretasi kanonik).

Catatan portabilitas threshold: kalibrasi dilakukan pada output question
rewriter dengan prompt UQ (Prinsip A-D). Production ini me-sample prompt
interpretasi yang serupa secara semantik (model diminta memilih satu
interpretasi acak bila multiple interpretation plausible). Threshold
diasumsikan transferable karena mekanisme entropy-of-clustered-rewrites
identik. Lihat skripsi §3.1.4 untuk justifikasi.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from typing import Sequence

import numpy as np


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity antara dua vektor embedding.

    Mengembalikan ``0.0`` bila salah satu vektor nol-norm (mis. embedding
    kosong). Tidak melempar exception supaya pipeline detection robust.
    """
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def cluster_single_link(
    embeddings: Sequence[Sequence[float]],
    threshold: float,
) -> list[int]:
    """Single-link agglomerative clustering atas cosine similarity.

    Dua sample masuk cluster yang sama bila ada path edge cosine ≥ threshold
    di antara keduanya. Mirror persis implementasi kalibrasi Stage 1.

    Mengembalikan list cluster id (0-indexed, urut sesuai input).
    """
    n = len(embeddings)
    if n == 0:
        return []
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if cosine_similarity(embeddings[i], embeddings[j]) >= threshold:
                adj[i].append(j)
                adj[j].append(i)
    cids = [-1] * n
    next_cid = 0
    for start in range(n):
        if cids[start] != -1:
            continue
        queue = deque([start])
        cids[start] = next_cid
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if cids[v] == -1:
                    cids[v] = next_cid
                    queue.append(v)
        next_cid += 1
    return cids


def normalized_entropy(fingerprints: Sequence[str]) -> float:
    """Shannon entropy dinormalisasi ke [0, 1].

    Entropy dihitung atas distribusi fingerprint (mis. cluster id sebagai
    string ``"C0"``, ``"C1"``, ``"ERROR"``). Dinormalisasi dengan ``log(M)``
    sehingga 0 = semua sama, 1 = semua berbeda (sebaran uniform maksimal).
    """
    M = len(fingerprints)
    if M <= 1:
        return 0.0
    H = 0.0
    for count in Counter(fingerprints).values():
        p = count / M
        H -= p * math.log(p)
    return H / math.log(M)


def mean_intra_cosine(embeddings: Sequence[Sequence[float]]) -> float:
    """Rata-rata cosine similarity antar pasangan sample.

    Berguna sebagai metric pendamping H_norm untuk logging/diagnostik;
    nilai mendekati 1.0 berarti sample konvergen, nilai rendah berarti
    sample divergen. Tidak dipakai dalam keputusan threshold.
    """
    n = len(embeddings)
    if n < 2:
        return 1.0
    sims: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_similarity(embeddings[i], embeddings[j]))
    return float(np.mean(sims)) if sims else 1.0


def compute_uq_signal(
    samples: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    *,
    m_total: int,
    tau_cluster: float,
) -> dict[str, float | int | list[str]]:
    """Hitung sinyal UQ lengkap dari sample text + embedding.

    Args:
        samples: list teks interpretasi (sudah difilter non-empty).
        embeddings: list vektor embedding sejajar dengan ``samples``.
        m_total: M_SAMPLING penuh sebelum filter error (untuk fingerprint
            ``"ERROR"`` filler).
        tau_cluster: threshold single-link cosine.

    Returns dict:
        - ``h_norm``: float di [0,1].
        - ``unique_outcomes``: jumlah cluster unik (termasuk ERROR bila ada).
        - ``majority_ratio``: rasio sample di cluster terbesar.
        - ``mean_intra_cosine``: rata-rata cosine antar pasangan valid.
        - ``n_error``: jumlah sample gagal/empty.
        - ``fingerprints``: list fingerprint per sample (``"C0"`` / ``"ERROR"``).
    """
    n_valid = len(samples)
    n_error = max(0, m_total - n_valid)

    if n_valid == 0:
        fingerprints = ["ERROR"] * m_total
        return {
            "h_norm": normalized_entropy(fingerprints),
            "unique_outcomes": 1 if m_total else 0,
            "majority_ratio": 1.0 if m_total else 0.0,
            "mean_intra_cosine": 0.0,
            "n_error": n_error,
            "fingerprints": fingerprints,
            "majority_cluster_representative": "",
        }

    cids = cluster_single_link(embeddings, tau_cluster)
    valid_fps = [f"C{c}" for c in cids]
    fingerprints = valid_fps + ["ERROR"] * n_error
    cnt = Counter(fingerprints)
    maj_label, maj_n = cnt.most_common(1)[0]
    # Representative = sample pertama (urutan input) di cluster terbesar
    # non-ERROR. Bila cluster terbesar adalah "ERROR" (semua-ish gagal),
    # ambil sample pertama dari cluster non-ERROR terbesar berikutnya;
    # bila semua ERROR, return string kosong (caller fallback ke original).
    representative = ""
    if maj_label != "ERROR":
        for idx, fp in enumerate(valid_fps):
            if fp == maj_label:
                representative = samples[idx]
                break
    else:
        for label, _count in cnt.most_common():
            if label == "ERROR":
                continue
            for idx, fp in enumerate(valid_fps):
                if fp == label:
                    representative = samples[idx]
                    break
            if representative:
                break
    return {
        "h_norm": normalized_entropy(fingerprints),
        "unique_outcomes": len(cnt),
        "majority_ratio": maj_n / m_total if m_total else 0.0,
        "mean_intra_cosine": mean_intra_cosine(embeddings),
        "n_error": n_error,
        "fingerprints": fingerprints,
        "majority_cluster_representative": representative,
    }
