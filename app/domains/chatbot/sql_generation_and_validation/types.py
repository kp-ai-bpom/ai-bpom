from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ValidationLevel(str, Enum):
    """Mode operasi keseluruhan modul SQL validation.

    - ``FULL``: jalankan execution loop + semantic loop (default).
    - ``EXECUTION_ONLY``: jalankan execution loop saja, lewati semantic.
    - ``NONE``: bypass total — kembali ke perilaku lama tanpa validasi tambahan.
    """

    FULL = "full"
    EXECUTION_ONLY = "execution_only"
    NONE = "none"

    @classmethod
    def from_string(cls, value: str | None) -> "ValidationLevel":
        if not value:
            return cls.FULL
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        return cls.FULL


class ValidationStatus(str, Enum):
    """Label diskrit ala Scale partial-reward.

    - ``PASS``: SQL aman dieksekusi, hasil dapat dipakai.
    - ``PARTIAL``: SQL berada pada arah benar tetapi belum sepenuhnya tepat;
      diberi kesempatan untuk satu kali revisi.
    - ``FAIL``: SQL tidak layak dieksekusi; sistem masuk safety-net dan tidak
      menjalankan SQL.
    - ``SKIPPED``: validasi tidak dijalankan (mode ``NONE`` atau procedural hit).
    """

    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


# Lima dimensi rubric (Bab 2.9): hybrid Huyen LLM-as-a-Judge + Spider Component Matching.
RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "faithfulness",
    "relevance",
    "select",
    "where",
    "group_by",
)


@dataclass(frozen=True)
class ExecutionVerdict:
    """Hasil satu kali eksekusi dry-run terhadap SQL kandidat."""

    ok: bool
    error_message: str | None = None
    rows_returned: int = 0


@dataclass(frozen=True)
class RubricDimensionVerdict:
    """Verdict satu dimensi rubric: label + alasan singkat."""

    label: ValidationStatus
    reason: str = ""


@dataclass(frozen=True)
class RubricVerdict:
    """Hasil penilaian semantic judge (rubric 5 dimensi + label agregat)."""

    overall: ValidationStatus
    dimensions: dict[str, RubricDimensionVerdict] = field(default_factory=dict)
    summary: str = ""

    def as_payload(self) -> dict[str, dict[str, str]]:
        """Bentuk dict yang siap diserialkan ke respons API."""
        return {
            name: {"label": verdict.label.value, "reason": verdict.reason}
            for name, verdict in self.dimensions.items()
        }


@dataclass(frozen=True)
class ValidationRevision:
    """Catatan satu langkah revisi (oleh refiner) untuk audit trail."""

    iteration: int
    trigger: str  # "execution_error" | "semantic_partial"
    feedback: str
    sql_before: str
    sql_after: str


@dataclass(frozen=True)
class ValidationResult:
    """Hasil akhir SQL validation pipeline yang diteruskan ke pipeline utama."""

    status: ValidationStatus
    final_sql: str
    explanation: str
    rubric: RubricVerdict | None = None
    execution_iterations: int = 0
    semantic_iterations: int = 0
    revisions: list[ValidationRevision] = field(default_factory=list)
    last_execution_error: str | None = None

    @property
    def is_safe_to_execute(self) -> bool:
        """SQL boleh dieksekusi oleh pipeline jika status PASS atau SKIPPED."""
        return self.status in (ValidationStatus.PASS, ValidationStatus.SKIPPED)
