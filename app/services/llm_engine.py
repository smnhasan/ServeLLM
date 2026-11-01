from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from app.schemas.openai import ChatMessage


class LLMEngine(ABC):
    """Abstract interface for LLM inference backends."""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Generate a chat completion and return the full response text."""
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens one by one."""
        ...

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 16,
        stop: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Legacy text completion."""
        ...

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str,
        **kwargs,
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    def count_tokens(self, text: str, model: str) -> int:
        """Estimate token count for billing/usage."""
        ...
