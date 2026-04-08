from dataclasses import dataclass
from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
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


class LLMManager:
    """
    Singleton class untuk mengelola koneksi LangChain LLM.
    Memastikan hanya ada satu instance LLM di memori selama aplikasi berjalan.
    """

    _instance = None
    _instruct: Optional[BaseChatModel] = None
    _think: Optional[BaseChatModel] = None
    _deep_think: Optional[BaseChatModel] = None

    def __new__(cls):
        """Override __new__ untuk memastikan hanya satu instance LLMManager yang dibuat."""
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    def _initialize_instruct(self):
        """Inisialisasi Instruct Model jika belum ada"""
        if self._instruct is None:
            if settings.OPENAI_API_KEY:
                self._instruct = ChatOpenAI(
                    model=settings.AI_INSTRUCT_MODEL_NAME,
                    api_key=SecretStr(settings.OPENAI_API_KEY),
                    base_url=settings.AI_BASE_URL,
                    temperature=0.7,
                )
                log.info("✅ OpenAI Instruct Model Initialized")
            else:
                self._instruct = ChatAnthropic(
                    model_name=settings.AI_INSTRUCT_MODEL_NAME,
                    api_key=SecretStr(settings.ANTHROPIC_API_KEY),
                    base_url=settings.AI_BASE_URL,
                    temperature=0.7,
                    timeout=60,
                    stop=["\n\nHuman:"],
                )
                log.info("✅ Anthropic Instruct Model Initialized")

    def _initialize_think(self):
        """Inisialisasi Think Model jika belum ada"""
        if self._think is None:
            if settings.OPENAI_API_KEY:
                self._think = ChatOpenAI(
                    model=settings.AI_THINK_MODEL_NAME,
                    api_key=SecretStr(settings.OPENAI_API_KEY),
                    base_url=settings.AI_BASE_URL,
                    temperature=0.7,
                )
                log.info("✅ OpenAI Think Model Initialized")
            else:
                self._think = ChatAnthropic(
                    model_name=settings.AI_THINK_MODEL_NAME,
                    api_key=SecretStr(settings.ANTHROPIC_API_KEY),
                    base_url=settings.AI_BASE_URL,
                    temperature=0.7,
                    timeout=60,
                    stop=["\n\nHuman:"],
                )
                log.info("✅ Anthropic Think Model Initialized")

    def _initialize_deep_think(self):
        """Inisialisasi Deep Think Model jika belum ada"""
        if self._deep_think is None:
            if settings.OPENAI_API_KEY:
                self._deep_think = ChatOpenAI(
                    model=settings.AI_DEEP_THINK_MODEL_NAME,
                    api_key=SecretStr(settings.OPENAI_API_KEY),
                    base_url=settings.AI_BASE_URL,
                    temperature=0.7,
                )
                log.info("✅ OpenAI Deep Think Model Initialized")
            else:
                self._deep_think = ChatAnthropic(
                    model_name=settings.AI_DEEP_THINK_MODEL_NAME,
                    api_key=SecretStr(settings.ANTHROPIC_API_KEY),
                    base_url=settings.AI_BASE_URL,
                    temperature=0.7,
                    timeout=60,
                    stop=["\n\nHuman:"],
                )
                log.info("✅ Anthropic Deep Think Model Initialized")

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
    )
