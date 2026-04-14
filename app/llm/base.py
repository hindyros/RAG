"""
Structural protocol for LLM provider clients.

Any class that implements embed(), chat(), and close() satisfies this protocol
and can be used anywhere an LLMClient is expected — no inheritance needed.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one float vector per text."""
        ...

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Call the chat completions endpoint. Returns the assistant reply text."""
        ...

    async def close(self) -> None:
        """Release any held resources (connection pool, etc.)."""
        ...
