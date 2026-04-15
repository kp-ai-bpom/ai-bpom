from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.db.database import Base


class Suksesor(Base):
    """Model untuk pemetaan calon penerus jabatan."""

    __tablename__ = "suksesor"

    id = Column(
        UUID(as_uuid=False), primary_key=True, server_default=func.gen_random_uuid()
    )
    nip = Column(String(20), unique=True, nullable=False, index=True)
    nama = Column(String(255), nullable=False)
    unit_kerja = Column(String(255), nullable=True)
    grade = Column(String(5), nullable=True)
    kompetensi = Column(Text, nullable=True)  # JSON-like field untuk kompetensi
    potensi = Column(String(20), nullable=True)  # High, Medium, Low
    readiness = Column(Integer, nullable=True)  # 0-100
    is_active = Column(Boolean, server_default="true", nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self):
        return f"<Suksesor(nip={self.nip}, nama={self.nama})>"


class MatchingHistory(Base):
    """Model untuk riwayat hasil matching pemetaan suksesor."""

    __tablename__ = "matching_history"

    id = Column(
        UUID(as_uuid=False), primary_key=True, server_default=func.gen_random_uuid()
    )
    target_jabatan = Column(String(255), nullable=False)
    total_kandidat = Column(Integer, nullable=False)
    top_kandidat = Column(JSONB, nullable=False)
    sub_tugas = Column(JSONB, nullable=True)
    catatan_reviewer = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self):
        return f"<MatchingHistory(target_jabatan={self.target_jabatan})>"
