"""
LLM-based reranker.

After hybrid RRF retrieval we have top_k_retrieval candidates.  The reranker
scores each one independently (pointwise) and re-sorts them by relevance score.

Why rerank at all?
    RRF fusion is rank-based and does not measure answer quality.  The reranker
    adds a cross-attention signal: "given this *specific* question, how useful
    is *this specific* passage?"  In practice this moves the most directly
    helpful chunk to position 1, which matters because the LLM tends to give
    more weight to early context (known as "lost in the middle" bias).

Trade-off:
    Pointwise reranking = N sequential LLM calls (expensive).  For top_k=10
    this is ~10 × 0.3s ≈ 3s extra latency.  Acceptable for a demo; a
    production system would use a specialised cross-encoder model (e.g.
    Cohere Rerank, BGE-reranker) which runs in < 200ms.
"""

import json
import logging

from app.llm.base import LLMClient
from app.llm.prompts.rerank import build_rerank_messages

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def rerank(
        self,
        query: str,
        candidates: list[dict],   # from hybrid_retrieve(): {index, rrf_score, cosine_score, ...}
        chunk_texts: list[str],   # parallel list — chunk_texts[i] is the text for candidates[i]
        top_k: int = 5,
    ) -> list[dict]:
        """
        Score each candidate chunk for relevance to `query` and return the top_k.

        Each returned dict is the original candidate dict augmented with:
            "rerank_score"   : int (0-10 from the LLM)
            "rerank_reason"  : str
        Results are sorted by rerank_score descending.

        On any scoring failure the candidate retains its RRF score as a proxy
        (converted to [0-10] range by multiplying by 100 and clamping).
        """
        scored: list[dict] = []

        for candidate, text in zip(candidates, chunk_texts):
            score, reason = await self._score_one(query, text)
            scored.append({**candidate, "rerank_score": score, "rerank_reason": reason})

        # Sort by rerank score, break ties by cosine_score
        scored.sort(key=lambda x: (x["rerank_score"], x["cosine_score"]), reverse=True)

        return scored[:top_k]

    async def _score_one(self, query: str, chunk_text: str) -> tuple[int, str]:
        """
        Ask Mistral to score one chunk's relevance to the query (0-10).
        Returns (score, reasoning).
        """
        messages = build_rerank_messages(query, chunk_text)
        try:
            raw = await self._client.chat(
                messages=messages,
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw)
            score = max(0, min(10, int(data.get("score", 5))))
            reason = data.get("reasoning", "")
            return score, reason
        except Exception as exc:
            logger.warning("Rerank scoring failed: %s — using fallback score.", exc)
            return 5, "Scoring unavailable"
