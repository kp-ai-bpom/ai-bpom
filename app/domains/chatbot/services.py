from typing import List, Optional

from fastapi import Depends

from app.core.llm import LLMAdapter, init_llm


class ChatbotService:
    """
    Service for handling chatbot LLM interactions.
    Uses LLMAdapter to access instruct, think, and deep_think models.
    """

    def __init__(self, llm_adapter: LLMAdapter):
        self._llm_adapter = llm_adapter

    def instruct(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Invoke the instruct model for general purpose conversation"""
        response = self._llm_adapter.instruct.bind(max_tokens=max_tokens).invoke(
            messages
        )
        return str(response.content)

    def think(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Invoke the think model for reasoning-focused responses"""
        response = self._llm_adapter.think.bind(max_tokens=max_tokens).invoke(messages)
        return str(response.content)

    def deep_think(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Invoke the deep_think model for in-depth analysis"""
        response = self._llm_adapter.deep_think.bind(max_tokens=max_tokens).invoke(
            messages
        )
        return str(response.content)

    async def a_instruct(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Async invoke the instruct model"""
        response = await self._llm_adapter.instruct.bind(max_tokens=max_tokens).ainvoke(
            messages
        )
        return str(response.content)

    async def a_think(self, messages: List, max_tokens: Optional[int] = None) -> str:
        """Async invoke the think model"""
        response = await self._llm_adapter.think.bind(max_tokens=max_tokens).ainvoke(
            messages
        )
        return str(response.content)

    async def a_deep_think(
        self, messages: List, max_tokens: Optional[int] = None
    ) -> str:
        """Async invoke the deep_think model"""
        response = await self._llm_adapter.deep_think.bind(
            max_tokens=max_tokens
        ).ainvoke(messages)
        return str(response.content)


def get_llm_adapter() -> LLMAdapter:
    """Dependency untuk mendapatkan LLMAdapter"""
    return init_llm()


# dependencies injection
def get_chatbot_service(
    llm_adapter: LLMAdapter = Depends(get_llm_adapter),
) -> ChatbotService:
    """Dependency untuk mendapatkan ChatbotService"""
    return ChatbotService(llm_adapter)
