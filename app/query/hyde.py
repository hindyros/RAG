"""
HyDE (Hypothetical Document Embeddings) query transformer.

See app/llm/prompts/hyde.py for a detailed explanation of the HyDE technique.

This module:
1. Generates a hypothetical passage using the Mistral LLM.
2. Embeds that passage using mistral-embed.
3. Returns the embedding vector (and the hypothetical text for logging).

The caller (query/pipeline.py) uses the returned vector for cosine retrieval
while continuing to use the original query text for BM25 retrieval.
"""

import logging

import numpy as np

from app.llm.base import LLMClient
from app.llm.prompts.hyde import build_hyde_messages

logger = logging.getLogger(__name__)


class HyDETransformer:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def transform(self, query: str) -> tuple[np.ndarray, str]:
        """
        Generate a hypothetical document for `query` and embed it.

        Returns
        -------
        embedding      : 1D float32 array — use as the dense search vector.
        hypothetical   : The generated text — logged for observability.
        """
        messages = build_hyde_messages(query)
        hypothetical = await self._client.chat(
            messages=messages,
            temperature=0.7,   # some diversity improves recall
            max_tokens=300,
        )
        logger.debug("HyDE passage: %s", hypothetical[:120])

        embeddings = await self._client.embed([hypothetical])
        vector = np.array(embeddings[0], dtype=np.float32)

        return vector, hypothetical
