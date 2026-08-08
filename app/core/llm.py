from dataclasses import dataclass
from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from app.core.config import settings
from app.core.logger import log


@dataclass
class LLMAdapter:
    """
    Adapter yang menyimpan semua instance LLM.
    Dapat di-inject ke service/dependency manapun yang membutuhkan AI.
    """

    instruct: BaseChatModel
    think: BaseChatModel
    deep_think: BaseChatModel
    embeddings: OpenAIEmbeddings


class LLMManager:
    """
    Singleton class untuk mengelola koneksi LangChain LLM.
    Memastikan hanya ada satu instance LLM di memori selama aplikasi berjalan.

    Provider routing:
        - settings.LLM_PROVIDER = "openai" | "anthropic" → eksplisit
        - settings.LLM_PROVIDER = ""                     → auto-detect
                                                           (openai > anthropic)
    """

    _instance = None
    _instruct: Optional[BaseChatModel] = None
    _think: Optional[BaseChatModel] = None
    _deep_think: Optional[BaseChatModel] = None
    _embeddings: Optional[OpenAIEmbeddings] = None

    def __new__(cls):
        """Override __new__ untuk memastikan hanya satu instance LLMManager yang dibuat."""
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    # ── Provider selection ──────────────────────────────────────────────

    @staticmethod
    def _resolve_provider() -> str:
        """Pilih provider aktif.

        - Kalau LLM_PROVIDER di-set ke nilai valid → pakai itu.
        - Kalau LLM_PROVIDER kosong → auto-detect (openai > anthropic).
        - Kalau LLM_PROVIDER di-set ke nilai TIDAK dikenal → raise ValueError
          (jangan diam-diam fallback; misconfig harus terlihat).
        """
        explicit = (settings.LLM_PROVIDER or "").strip().lower()
        if explicit:
            if explicit not in {"openai", "anthropic"}:
                raise ValueError(
                    f"LLM_PROVIDER={explicit!r} tidak dikenal. "
                    "Nilai valid: 'openai' | 'anthropic' | '' (auto)."
                )
            return explicit
        # Auto-detect (backward compatible dengan perilaku lama)
        if settings.OPENAI_API_KEY:
            return "openai"
        if settings.ANTHROPIC_API_KEY:
            return "anthropic"
        # Default terakhir
        return "openai"

    def _build_chat_model(
        self,
        model_name: str,
        *,
        temperature: Optional[float] = None,
    ) -> BaseChatModel:
        """Bangun ChatModel sesuai provider aktif. Dipakai oleh _initialize_*.

        ``temperature`` adalah parameter EKSPLISIT supaya tiap domain
        (chatbot, pemetaan_suksesor, dll) bisa menyuplai nilainya sendiri.
        Bila ``None``, dipakai default shared ``settings.LLM_DEFAULT_TEMPERATURE``
        (0.7) — sengaja TIDAK lagi mengacu ke setting domain manapun
        agar ``app/core/`` tetap netral terhadap domain.
        """
        provider = self._resolve_provider()
        if temperature is None:
            temperature = settings.LLM_DEFAULT_TEMPERATURE

        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise RuntimeError(
                    "LLM_PROVIDER=openai tapi OPENAI_API_KEY kosong."
                )
            return ChatOpenAI(
                model=model_name,
                api_key=SecretStr(settings.OPENAI_API_KEY),
                base_url=settings.AI_BASE_URL,
                temperature=temperature,
            )

        if provider == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise RuntimeError(
                    "LLM_PROVIDER=anthropic tapi ANTHROPIC_API_KEY kosong."
                )
            return ChatAnthropic(
                model_name=model_name,
                api_key=SecretStr(settings.ANTHROPIC_API_KEY),
                base_url=settings.AI_BASE_URL,
                temperature=temperature,
                timeout=60,
                stop=["\n\nHuman:"],
            )

        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r} (expected: openai | anthropic)"
        )

    # ── Lazy initializers ───────────────────────────────────────────────

    def _initialize_instruct(self):
        """Inisialisasi Instruct Model jika belum ada"""
        if self._instruct is None:
            self._instruct = self._build_chat_model(settings.AI_INSTRUCT_MODEL_NAME)
            log.info(
                f"✅ {self._resolve_provider().capitalize()} Instruct Initialized "
                f"({settings.AI_INSTRUCT_MODEL_NAME}, T={settings.LLM_DEFAULT_TEMPERATURE})"
            )

    def _initialize_think(self):
        """Inisialisasi Think Model jika belum ada"""
        if self._think is None:
            self._think = self._build_chat_model(settings.AI_THINK_MODEL_NAME)
            log.info(
                f"✅ {self._resolve_provider().capitalize()} Think Initialized "
                f"({settings.AI_THINK_MODEL_NAME}, T={settings.LLM_DEFAULT_TEMPERATURE})"
            )

    def _initialize_deep_think(self):
        """Inisialisasi Deep Think Model jika belum ada"""
        if self._deep_think is None:
            self._deep_think = self._build_chat_model(settings.AI_DEEP_THINK_MODEL_NAME)
            log.info(
                f"✅ {self._resolve_provider().capitalize()} Deep Think Initialized "
                f"({settings.AI_DEEP_THINK_MODEL_NAME}, T={settings.LLM_DEFAULT_TEMPERATURE})"
            )

    def get_embeddings(self) -> OpenAIEmbeddings:
        """Inisialisasi embeddings jika belum ada, lalu kembalikan instance-nya.

        NOTE: Embeddings sengaja dikunci ke OpenAI karena vector store sudah
        di-index pakai OpenAI embeddings. Mengganti embeddings model di tengah
        eksperimen cross-LLM akan invalidate seluruh index. Kalau mau ganti,
        re-index dulu seluruh knowledge_entities.
        """
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=settings.AI_EMBEDDINGS_MODEL_NAME,
                api_key=SecretStr(settings.OPENAI_API_KEY),
                base_url=settings.AI_BASE_URL,
            )
            log.info(
                f"✅ OpenAI Embeddings Initialized ({settings.AI_EMBEDDINGS_MODEL_NAME})"
            )
        return self._embeddings

    def get_llm(self, model_type: str = "instruct") -> BaseChatModel:
        """Inisialisasi LLM jika belum ada, lalu kembalikan instance-nya"""
        if model_type == "instruct":
            self._initialize_instruct()
            assert self._instruct is not None
            return self._instruct
        elif model_type == "think":
            self._initialize_think()
            assert self._think is not None
            return self._think
        elif model_type == "deep_think":
            self._initialize_deep_think()
            assert self._deep_think is not None
            return self._deep_think
        else:
            self._initialize_instruct()
            assert self._instruct is not None
            return self._instruct

    def invoke(
        self,
        messages: List,
        max_tokens: Optional[int] = None,
        model_type: str = "instruct",
    ) -> Optional[str]:
        """Synchronous Invoke LangChain client"""
        try:
            response = (
                self.get_llm(model_type).bind(max_tokens=max_tokens).invoke(messages)
            )
            return str(response.content)
        except Exception as e:
            log.exception(f"Error invoking LLM: {e}")
            return None

    async def ainvoke(
        self,
        messages: List,
        max_tokens: Optional[int] = None,
        model_type: str = "instruct",
    ) -> Optional[str]:
        """Asynchronous Invoke LangChain client"""
        try:
            response = (
                await self.get_llm(model_type)
                .bind(max_tokens=max_tokens)
                .ainvoke(messages)
            )
            return str(response.content)
        except Exception as e:
            log.exception(f"Error invoking LLM: {e}")
            return None

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed text using the embeddings model"""
        try:
            return self.get_embeddings().embed_query(text)
        except Exception as e:
            log.exception(f"Error embedding text: {e}")
            return None


# Dependency Factory
def init_llm() -> LLMAdapter:
    """
    Dependency Injection untuk mendapatkan semua instance LLM.
    Bisa disuntikkan ke service mana pun yang membutuhkan AI.
    """
    manager = LLMManager()
    return LLMAdapter(
        instruct=manager.get_llm("instruct"),
        think=manager.get_llm("think"),
        deep_think=manager.get_llm("deep_think"),
        embeddings=manager.get_embeddings(),
    )
