"""
Async HTTP client for the Mistral AI API.

All API calls (embeddings, chat completions) funnel through this module.
Centralising them here means:
- Retry logic and error handling live in one place.
- The API key is never passed around as a parameter — it's read once at
  startup from config.
- Every other module depends on this abstraction, not on the Mistral SDK,
  so swapping the provider only requires changes here.

We use httpx rather than the official Mistral SDK to avoid SDK version
drift and to keep full control over retry behaviour and connection pooling.
"""

import asyncio
import logging
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

_EMBED_URL = "https://api.mistral.ai/v1/embeddings"
_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.5   # seconds; doubles on each attempt


class MistralClient:
    """
    Thin async wrapper around the Mistral REST API.

    Instantiate once at application startup (via dependency injection) and
    reuse across requests — the underlying httpx.AsyncClient maintains a
    connection pool that amortises TLS handshake cost.
    """

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.mistral_api_key
        self._embed_model = settings.mistral_embed_model
        self._chat_model = settings.mistral_chat_model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        """Call during application shutdown to release the connection pool."""
        await self._client.aclose()

    # ── Embeddings ────────────────────────────────────────────────────────────

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts and return their float32 vectors.

        Mistral's embedding endpoint accepts up to 2048 tokens per input and
        processes inputs in a single batched call.  Callers that need to embed
        more than max_embed_batch_size texts should split and call this method
        multiple times (see ingestion/pipeline.py).
        """
        payload: dict[str, Any] = {
            "model": self._embed_model,
            "input": texts,
        }
        response = await self._post_with_retry(_EMBED_URL, payload)
        # Response shape: {"data": [{"embedding": [...], "index": N}, ...]}
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
        """
        Call the chat completions endpoint and return the assistant message text.

        temperature=0.1 is the default to favour determinism in retrieval-based
        answers.  HyDE generation uses a higher temperature (see query/hyde.py).

        If `response_format={"type": "json_object"}` is passed, Mistral will
        constrain its output to valid JSON — used by intent detection and the
        reranker.
        """
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
        """
        POST to `url` with exponential back-off on transient failures.

        We retry on:
        - Network errors (connection reset, timeout)
        - HTTP 429 (rate limit) and 5xx (server error)

        We do NOT retry on 4xx client errors (bad request, auth failure) as
        retrying won't help.
        """
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(url, json=payload)

                if response.status_code == 200:
                    return response.json()

                if response.status_code in (429, 500, 502, 503, 504):
                    wait = _RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "Mistral API returned %d, retrying in %.1fs (attempt %d/%d)",
                        response.status_code, wait, attempt + 1, _MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    continue

                # Non-retryable HTTP error
                response.raise_for_status()

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Mistral API network error: %s — retrying in %.1fs (attempt %d/%d)",
                    exc, wait, attempt + 1, _MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                last_error = exc

        raise RuntimeError(
            f"Mistral API call failed after {_MAX_RETRIES} attempts"
        ) from last_error
