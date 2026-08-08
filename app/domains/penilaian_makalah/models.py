"""
SQLAlchemy Models for Penilaian Makalah.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
)

from app.db.database import Base


class EvaluationResult(Base):
    """
    Menyimpan hasil evaluasi makalah.
    """

    __tablename__ = "evaluation_results"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
    )

    paper_filename = Column(
        String(255),
        nullable=False,
    )

    jabatan = Column(
        String(255),
        nullable=False,
    )

    query_mode = Column(
        String(50),
        nullable=False,
        default="hybrid",
    )

    scores = Column(
        JSON,
        nullable=False,
    )

    justification = Column(
        JSON,
        nullable=False,
    )

    evidence = Column(
        JSON,
        nullable=False,
    )

    uncertainty_metrics = Column(
        JSON,
        nullable=True,
    )

    ringkasan = Column(
        Text,
        nullable=True,
    )

    final_score = Column(
        Float,
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )



class IngestionLog(Base):
    """
    Log proses ingestion ke LightRAG.
    """

    __tablename__ = "ingestion_logs"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
    )

    filename = Column(
        String(255),
        nullable=False,
    )

    status = Column(
        String(30),
        nullable=False,
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    ingested_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )