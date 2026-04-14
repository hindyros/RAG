"""
Async HTTP client for the OpenAI API.

Mirrors the interface of MistralClient exactly so the two are interchangeable
everywhere an LLMClient is expected.  We use httpx directly (same as the
Mistral client) rather than the openai SDK to keep the dependency footprint
small and retry logic consistent.

Default models:
  embed : text-embedding-3-small  (1536-dim, cheap, strong)
  chat  : gpt-4o-mini             (fast, low cost, GPT-4 quality)
"""

import asyncio
import logging
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

_EMBED_URL = "https://api.openai.com/v1/embeddings"
_CHAT_URL  = "https://api.openai.com/v1/chat/completions"

_MAX_RETRIES        = 3
_RETRY_BACKOFF_BASE = 1.5


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        self._api_key    = settings.openai_api_key
        self._embed_model = settings.openai_embed_model
        self._chat_model  = settings.openai_chat_model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ── Embeddings ────────────────────────────────────────────────────────────

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts via the OpenAI embeddings endpoint.

        OpenAI returns results in the order sent, but we sort by index anyway
        to be safe (consistent with the Mistral client behaviour).
        """
        payload: dict[str, Any] = {
            "model": self._embed_model,
            "input": texts,
        }
        response = await self._post_with_retry(_EMBED_URL, payload)
        sorted_data = sorted(response["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in sorted_data]

    # ── Chat completions ──────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        response = await self._post_with_retry(_CHAT_URL, payload)
        return response["choices"][0]["message"]["content"]

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _post_with_retry(self, url: str, payload: dict) -> dict:
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(url, json=payload)

                if response.status_code == 200:
                    return response.json()

                if response.status_code in (429, 500, 502, 503, 504):
                    wait = _RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "OpenAI API returned %d, retrying in %.1fs (attempt %d/%d)",
                        response.status_code, wait, attempt + 1, _MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    continue

                response.raise_for_status()

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "OpenAI API network error: %s — retrying in %.1fs (attempt %d/%d)",
                    exc, wait, attempt + 1, _MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                last_error = exc

        raise RuntimeError(
            f"OpenAI API call failed after {_MAX_RETRIES} attempts"
        ) from last_error
