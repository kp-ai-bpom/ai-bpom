"""Seed data untuk testing penilaian_makalah domain."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from app.db.database import AsyncSessionLocal
from app.domains.penilaian_makalah.models import EvaluationResult, IngestionLog


async def seed_evaluation_results():
    """Insert sample evaluation results untuk testing."""
    async with AsyncSessionLocal() as session:
        sample_results = [
            {
                "paper_filename": "analisis_pasar_2026.pdf",
                "jabatan": "Analis Data Senior",
                "query_mode": "hybrid",
                "scores": {
                    "n1_kesesuaian_judul": 82.0,
                    "n2_kesesuaian_isi": 78.0,
                    "n3_sistematika": 75.0,
                    "n4_ketajaman_analisis": 80.0,
                    "n5_penggunaan_bahasa": 77.0,
                },
                "justification": {
                    "n1_kesesuaian_judul": "Judul mencerminkan isi makalah dengan baik.",
                    "n2_kesesuaian_isi": "Isi konsisten dengan judul dan tema.",
                    "n3_sistematika": "Penulisan terstruktur dengan baik.",
                    "n4_ketajaman_analisis": "Analisis mendalam dan mendukung kesimpulan.",
                    "n5_penggunaan_bahasa": "Bahasa formal dan tepat.",
                },
                "evidence": {
                    "n1_kesesuaian_judul": "Lihat halaman judul.",
                    "n2_kesesuaian_isi": "Struktur BAB 1-4 relevan.",
                    "n3_sistematika": "Numbering dan format konsisten.",
                    "n4_ketajaman_analisis": "BAB 3 menunjukkan analisis kuantitatif.",
                    "n5_penggunaan_bahasa": "Tidak ada typo signifikan.",
                },
                "uncertainty_metrics": {
                    "per_criteria": {
                        "n1": {
                            "mean": 82.0,
                            "std": 2.1,
                            "nsv": 0.021,
                            "status": "✅ YAKIN",
                            "raw_samples": [80, 82, 84, 81, 82, 83, 82],
                        },
                        "n2": {
                            "mean": 78.0,
                            "std": 3.2,
                            "nsv": 0.032,
                            "status": "✅ YAKIN",
                            "raw_samples": [76, 78, 79, 77, 78, 79, 80],
                        },
                        "n3": {
                            "mean": 75.0,
                            "std": 2.8,
                            "nsv": 0.028,
                            "status": "✅ YAKIN",
                            "raw_samples": [73, 75, 76, 74, 75, 76, 75],
                        },
                        "n4": {
                            "mean": 80.0,
                            "std": 3.5,
                            "nsv": 0.035,
                            "status": "✅ YAKIN",
                            "raw_samples": [78, 80, 82, 79, 80, 81, 80],
                        },
                        "n5": {
                            "mean": 77.0,
                            "std": 2.9,
                            "nsv": 0.029,
                            "status": "✅ YAKIN",
                            "raw_samples": [75, 77, 78, 76, 77, 78, 77],
                        },
                    },
                    "weighted_aggregate": 0.029,
                    "overall_status": "✅ YAKIN (Konsisten)",
                    "most_uncertain_criteria": "n4",
                },
                "ringkasan": "Makalah berkualitas baik dengan analisis mendalam tentang pasar Indonesia.",
                "final_score": 78.4,
            },
            {
                "paper_filename": "proposal_inovasi_produk.docx",
                "jabatan": "Manajer Inovasi",
                "query_mode": "local",
                "scores": {
                    "n1_kesesuaian_judul": 75.0,
                    "n2_kesesuaian_isi": 72.0,
                    "n3_sistematika": 70.0,
                    "n4_ketajaman_analisis": 68.0,
                    "n5_penggunaan_bahasa": 73.0,
                },
                "justification": {
                    "n1_kesesuaian_judul": "Judul cukup mencerminkan isi.",
                    "n2_kesesuaian_isi": "Isi sebagian besar relevan.",
                    "n3_sistematika": "Struktur cukup jelas.",
                    "n4_ketajaman_analisis": "Analisis sederhana, perlu lebih mendalam.",
                    "n5_penggunaan_bahasa": "Bahasa mudah dipahami.",
                },
                "evidence": {
                    "n1_kesesuaian_judul": "Judul proposal sesuai dengan BAB 1.",
                    "n2_kesesuaian_isi": "Beberapa bagian tidak relevan dengan tema.",
                    "n3_sistematika": "Format penulisan konsisten.",
                    "n4_ketajaman_analisis": "Analisis SWOT sangat sederhana.",
                    "n5_penggunaan_bahasa": "Terdapat beberapa kesalahan penulisan.",
                },
                "uncertainty_metrics": {
                    "per_criteria": {
                        "n1": {"mean": 75.0, "std": 5.5, "nsv": 0.055, "status": "⚠️ PERLU REVIEW", "raw_samples": [70, 75, 78, 72, 75, 77, 74]},
                        "n2": {"mean": 72.0, "std": 6.2, "nsv": 0.062, "status": "⚠️ PERLU REVIEW", "raw_samples": [68, 72, 75, 70, 72, 74, 71]},
                        "n3": {"mean": 70.0, "std": 4.8, "nsv": 0.048, "status": "✅ YAKIN", "raw_samples": [68, 70, 72, 69, 70, 71, 70]},
                        "n4": {"mean": 68.0, "std": 7.1, "nsv": 0.071, "status": "⚠️ PERLU REVIEW", "raw_samples": [62, 68, 72, 65, 68, 70, 67]},
                        "n5": {"mean": 73.0, "std": 3.9, "nsv": 0.039, "status": "✅ YAKIN", "raw_samples": [70, 73, 75, 71, 73, 74, 72]},
                    },
                    "weighted_aggregate": 0.055,
                    "overall_status": "⚠️ BUTUH REVIEW HUMAN",
                    "most_uncertain_criteria": "n4",
                },
                "ringkasan": "Proposal inovasi yang baik namun memerlukan analisis lebih mendalam.",
                "final_score": 71.6,
            },
        ]

        for sample in sample_results:
            result = EvaluationResult(
                paper_filename=sample["paper_filename"],
                jabatan=sample["jabatan"],
                query_mode=sample["query_mode"],
                scores=sample["scores"],
                justification=sample["justification"],
                evidence=sample["evidence"],
                uncertainty_metrics=sample["uncertainty_metrics"],
                ringkasan=sample["ringkasan"],
                final_score=sample["final_score"],
                created_at=datetime.utcnow() - timedelta(days=len(sample_results) - sample_results.index(sample)),
            )
            session.add(result)

        await session.commit()
        print(f"✅ Seeded {len(sample_results)} evaluation results")


async def seed_ingestion_logs():
    """Insert sample ingestion logs untuk testing."""
    async with AsyncSessionLocal() as session:
        sample_logs = [
            {
                "filename": "knowledge_base_001.pdf",
                "status": "success",
                "error_message": None,
            },
            {
                "filename": "knowledge_base_002.txt",
                "status": "success",
                "error_message": None,
            },
            {
                "filename": "invalid_file.xyz",
                "status": "failed",
                "error_message": "Unsupported file format",
            },
        ]

        for log in sample_logs:
            ingestion = IngestionLog(
                filename=log["filename"],
                status=log["status"],
                error_message=log["error_message"],
            )
            session.add(ingestion)

        await session.commit()
        print(f"✅ Seeded {len(sample_logs)} ingestion logs")


async def main():
    """Run all seed operations."""
    print("🌱 Starting seed operations...")
    try:
        await seed_evaluation_results()
        await seed_ingestion_logs()
        print("✅ Seed completed successfully")
    except Exception as exc:
        print(f"❌ Seed failed: {exc}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
