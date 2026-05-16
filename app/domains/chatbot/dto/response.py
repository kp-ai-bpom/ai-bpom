from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatErrorResponse(BaseModel):
    """Schema response error untuk endpoint chatbot baru."""

    status: int
    error: str


class ClarificationOption(BaseModel):
    """Satu opsi interpretasi yang dikembalikan kepada user.

    - ``id``: identifier opsi (mis. ``opt_1``) yang dikirim balik oleh frontend
      saat user memilih.
    - ``label``: kalimat singkat dalam bahasa bisnis tanpa jargon teknis —
      cocok ditampilkan sebagai teks utama tombol.
    - ``description``: penjelasan 1-2 kalimat untuk membantu user memahami
      pilihan, termasuk konteks domain dan referensi sumber data. Boleh
      kosong ``""``. Frontend disarankan menampilkan ini sebagai sub-text
      kecil di bawah label, atau di tooltip.
    """

    id: str
    label: str
    description: str = ""


class PipelineStageEntry(BaseModel):
    """Satu entri trace eksekusi tahap pipeline ``send_message``.

    Dirancang untuk konsumsi UI skripsi: frontend menampilkan setiap entry
    sebagai dropdown yang dapat diekspand untuk inspeksi input/output tiap
    tahap (mirip representasi langkah-langkah di Replit Agent). Trace juga
    di-persist ke kolom ``chat_messages.pipeline_trace`` sehingga history
    percakapan dapat di-replay dengan jejak utuh.

    Field utama:

    - ``stage``: identifier tahap (``question_rewriting``, ``schema_retrieval``,
      ``ambiguity_detection``, ``sql_generation``, ``sql_validation``).
    - ``label``: label human-readable yang konsisten dengan penomoran tahap
      pada skripsi (``Stage 1 — ...`` s.d. ``Stage 5 — ...``).
    - ``status``: ``executed`` (jalan), ``skipped`` (sengaja dilewati pada
      jalur eksekusi tertentu), atau ``error``.
    - ``duration_ms``: durasi eksekusi tahap dalam milidetik (0 jika
      di-skip atau dihitung post-hoc).
    - ``summary``: ringkasan satu baris untuk header dropdown.
    - ``input`` / ``output``: payload terstruktur (boleh nested) yang
      menggambarkan masukan dan keluaran tahap.
    - ``metadata``: detail opsional (mis. iterasi, model, catatan timing).
    - ``error``: pesan error bila ``status == "error"``.
    """

    stage: str
    label: str
    status: Literal["executed", "skipped", "error"]
    duration_ms: int = 0
    summary: str = ""
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SendMessageDataResponse(BaseModel):
    """Data response untuk endpoint kirim pesan.

    Discriminated by `type`:
    - `type="answer"`: pertanyaan terjawab → `query` & `explanation` berisi SQL final
      dan penjelasannya. `options` kosong.
    - `type="clarification"`: pertanyaan ambigu → backend butuh user memilih opsi.
      `query` & `explanation` dikembalikan string kosong (`""`). `options` berisi
      pilihan interpretasi yang harus dipilih user.

    Field internal pipeline (`standalone_question`, `unambiguous_question`,
    `ambiguity_metadata`) TIDAK diekspos ke frontend — hanya disimpan ke tabel
    `question_rewriting_episodes` untuk kebutuhan logging/evaluasi.
    """

    type: Literal["answer", "clarification"]
    user_id: str
    session_id: str
    question: str
    query: str = ""
    explanation: str = ""
    options: List[ClarificationOption] = Field(default_factory=list)

    # Metadata SQL Validation Pipeline (Tahap 5). Semua optional agar
    # backward-compatible dengan konsumen lama dan tetap kosong pada
    # respons clarification.
    validation_status: Optional[Literal["PASS", "PARTIAL", "FAIL"]] = None
    validation_iterations: Optional[Dict[str, int]] = None
    validation_rubric: Optional[Dict[str, Dict[str, str]]] = None
    validation_revisions: Optional[List[Dict[str, Any]]] = None

    # Jejak per-tahap pipeline untuk konsumsi UI (dropdown stages) dan
    # juga di-persist ke ``chat_messages.pipeline_trace``. Optional agar
    # tetap backward-compatible.
    pipeline_trace: Optional[List[PipelineStageEntry]] = None


class SendMessageResponse(BaseModel):
    """Schema response untuk endpoint kirim pesan."""

    status: int
    data: SendMessageDataResponse


class ConversationItemResponse(BaseModel):
    """Representasi satu item percakapan dalam session.

    Field ``clarification_*`` dan ``ambiguity_type`` membawa jejak resolusi
    ambiguitas (jika turn ini hasil klarifikasi atau auto-resolve via
    procedural memory). UI memakainya untuk menggambar ulang badge
    "Klarifikasi: dipilih X dari [A, B, C]" pada history setelah refresh.
    Untuk turn biasa (tidak ambigu), keempat field bernilai default kosong
    (``None`` / ``[]``).
    """

    question: str
    query: str
    explanation: str | None = None
    pipeline_trace: Optional[List[PipelineStageEntry]] = None
    clarification_asked: Optional[str] = None
    clarification_options: List[ClarificationOption] = Field(default_factory=list)
    interpretation_chosen: Optional[str] = None
    ambiguity_type: Optional[str] = None


class PendingClarificationResponse(BaseModel):
    """Pending clarification yang belum di-resolve user.

    Disertakan di history response supaya UI dapat me-render kembali opsi
    jawaban setelah user refresh halaman (turn ``clarification`` tidak
    dipersist ke ``chat_messages`` karena belum punya SQL final).
    """

    pending_id: str
    question: str
    clarification_question: str
    options: List[dict]
    ambiguity_type: str | None = None
    expires_at: Optional[datetime] = None


class SessionMessagesDataResponse(BaseModel):
    """Data response untuk detail percakapan per session."""

    user_id: str
    session_id: str
    created_at: datetime
    updated_at: datetime
    conversations: List[ConversationItemResponse]
    pending_clarification: Optional[PendingClarificationResponse] = None


class SessionMessagesResponse(BaseModel):
    """Schema response untuk endpoint detail session."""

    status: int
    data: SessionMessagesDataResponse


class SessionListItemResponse(BaseModel):
    """Representasi ringkas metadata session."""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SessionListDataResponse(BaseModel):
    """Data response untuk daftar session user."""

    user_id: str
    sessions: List[SessionListItemResponse]


class SessionListResponse(BaseModel):
    """Schema response untuk endpoint list session."""

    status: int
    data: SessionListDataResponse


class DeleteSessionDataResponse(BaseModel):
    """Data response untuk penghapusan session."""

    message: str


class DeleteSessionResponse(BaseModel):
    """Schema response untuk endpoint hapus session."""

    status: int
    data: DeleteSessionDataResponse


class ChatDataResponse(BaseModel):
    """Schema untuk data response"""

    response: str


class ChatResponse(BaseModel):
    """Schema untuk response chat"""

    message: str
    data: ChatDataResponse


class EmbeddingDataResponse(BaseModel):
    """Schema untuk data response embedding"""

    embedding: List[float] | None = None


class EmbeddingResponse(BaseModel):
    """Schema untuk response embedding"""

    message: str
    data: EmbeddingDataResponse


class ImportBaseKnowledgeCsvDataResponse(BaseModel):
    """Data response untuk import + embedding base knowledge CSV."""

    table_name: str
    csv_path: str
    embedding_model: str
    total_rows: int
    targeted_rows: int
    processed_rows: int
    inserted_rows: int
    failed_rows: int
    truncated: bool
    expected_dimensions: int | None = None
    sample_errors: List[str] = Field(default_factory=list)


class ImportBaseKnowledgeCsvResponse(BaseModel):
    """Schema response untuk endpoint import CSV base knowledge."""

    status: int
    data: ImportBaseKnowledgeCsvDataResponse
