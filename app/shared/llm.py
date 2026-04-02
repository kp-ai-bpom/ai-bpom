from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import settings
from app.core.logger import log


class LLMManager:
    """
    Singleton class untuk mengelola koneksi LangChain LLM.
    Memastikan hanya ada satu instance LLM di memori selama aplikasi berjalan.
    """

    _instance = None
    _llm: Optional[BaseChatModel] = None

    def __new__(cls):
        """Override __new__ untuk memastikan hanya satu instance LLMManager yang dibuat."""
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    def get_llm(self) -> BaseChatModel:
        """Inisialisasi LLM jika belum ada, lalu kembalikan instance-nya"""

        if self._llm is None:
            log.info("🤖 Initialize LLM Connection")

            self._llm = ChatOpenAI(
                model=settings.OPENAI_MODEL_NAME,
                api_key=SecretStr(settings.OPENAI_API_KEY),
                base_url=settings.OPENAI_BASE_URL,
                temperature=0.7,
            )
            log.info("✅ LLM Initialized")

        return self._llm

    async def ainvoke(
        self, messages: List, max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """Asynchronous Invoke LangChain client"""
        try:
            response = (
                await self.get_llm().bind(max_tokens=max_tokens).ainvoke(messages)
            )
            return str(response.content)
        except Exception as e:
            log.exception(f"Error invoking LLM: {e}")
            return None


# Dependency Factory
def init_llm() -> BaseChatModel:
    """
    Dependency Injection untuk mendapatkan instance LLM.
    Bisa disuntikkan ke service mana pun yang membutuhkan AI.
    """
    manager = LLMManager()
    return manager.get_llm()
