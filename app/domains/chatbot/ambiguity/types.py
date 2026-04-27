from dataclasses import dataclass, field


AMBIGUITY_TYPES = {
    "vagueness",
    "scope",
    "column",
    "table",
    "join",
    "precomputed_aggregate",
    "attachment",
}


@dataclass(frozen=True)
class InterpretationOption:
    """Satu interpretasi yang ditawarkan ke pengguna saat ambigu.

    - ``label``: deskripsi singkat dalam bahasa bisnis sehari-hari (BUKAN
      jargon teknis seperti nama tabel/kolom/SQL). Maksimum ~120 karakter.
    - ``description``: penjelasan lebih lengkap (1-2 kalimat) yang membantu
      pengguna non-teknis memahami pilihan ini, termasuk konteks domain
      ("pool 1 = top talent / Star") dan referensi teknis sumber data
      (mis. "diambil dari kolom pool tabel period_employees").
    """

    label: str
    description: str = ""


@dataclass(frozen=True)
class AmbiguityDetectionResult:
    is_ambiguous: bool
    ambiguity_type: str | None = None
    clarification_question: str | None = None
    interpretation_options: list[InterpretationOption] = field(default_factory=list)


@dataclass(frozen=True)
class AmbiguityResolutionResult:
    unambiguous_question: str
    user_response: str
    auto_resolved: bool = False
    defaulted: bool = False
