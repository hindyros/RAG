"""
Query pipeline orchestrator.

This is the core of the RAG system.  Every user question flows through here.

Full pipeline (knowledge_seeking path):

  query
    ↓
  IntentDetector.detect()          → primary intent + sub-intent
    ↓ (if conversational)
  direct LLM reply (no retrieval)
    ↓ (if knowledge_seeking)
  HyDETransformer.transform()      → hypothetical document embedding
    ↓
  hybrid_retrieve()                → RRF-fused candidate list
    ↓
  Reranker.rerank()                → top_k_rerank chunks by relevance
    ↓
  threshold check (cosine score)   → refusal if below MIN_SIMILARITY_THRESHOLD
    ↓
  build_prompt(sub_intent)         → answer template filled with chunks
    ↓
  LLM.chat()                       → final answer text
    ↓
  QueryResponse (answer + citations)

Each step is a separate function/class so it can be tested, swapped, or
instrumented independently.
"""

import logging
import textwrap

import numpy as np

from app.api.schemas import (
    Citation,
    IntentResult,
    PrimaryIntent,
    QueryResponse,
    SubIntent,
)
from app.config import Settings
from app.intent.detector import IntentDetector
from app.llm.client import MistralClient
from app.llm.prompts.answer import comparison, conversational, factual, list_format, table
from app.query.hyde import HyDETransformer
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.reranker import Reranker
from app.store.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Maps sub-intent enum values to their prompt builder functions
_TEMPLATE_MAP = {
    SubIntent.FACTUAL:    factual.build_prompt,
    SubIntent.LIST:       list_format.build_prompt,
    SubIntent.COMPARISON: comparison.build_prompt,
    SubIntent.TABLE:      table.build_prompt,
}


class QueryPipeline:
    def __init__(
        self,
        store: VectorStore,
        client: MistralClient,
        settings: Settings,
    ) -> None:
        self._store = store
        self._client = client
        self._settings = settings
        self._intent_detector = IntentDetector(client, settings)
        self._hyde = HyDETransformer(client)
        self._reranker = Reranker(client)

    async def run(self, question: str) -> QueryResponse:
        """
        Execute the full RAG pipeline for a user question.

        Returns a QueryResponse with answer text, citations, and intent metadata.
        """
        # ── Step 1: Intent detection ─────────────────────────────────────────
        intent = await self._intent_detector.detect(question)

        # ── Step 2: Short-circuit for conversational queries ─────────────────
        if intent.primary == PrimaryIntent.CONVERSATIONAL:
            return await self._handle_conversational(question, intent)

        # ── Step 3: Guard — nothing in the store yet ─────────────────────────
        if self._store.total_chunks == 0:
            return QueryResponse(
                answer="No documents have been ingested yet.  Please upload PDF files first.",
                refused=True,
                refusal_reason="empty_store",
                intent=intent,
            )

        # ── Step 4: HyDE query transformation ───────────────────────────────
        hyde_embedding, hypothetical_doc = await self._hyde.transform(question)

        # ── Step 5: Hybrid retrieval ─────────────────────────────────────────
        candidates = hybrid_retrieve(
            query_embedding=hyde_embedding,
            query_text=question,
            cosine_index=self._store.get_cosine_index(),
            bm25_index=self._store.get_bm25_index(),
            top_k=self._settings.top_k_retrieval,
            rrf_k=self._settings.rrf_k_constant,
        )

        if not candidates:
            return self._refusal("no_relevant_documents", intent)

        # ── Step 6: Reranking ────────────────────────────────────────────────
        candidate_texts = [
            self._store.get_chunk(c["index"]).text for c in candidates
        ]
        reranked = await self._reranker.rerank(
            query=question,
            candidates=candidates,
            chunk_texts=candidate_texts,
            top_k=self._settings.top_k_rerank,
        )

        # ── Step 7: Similarity threshold (citation quality gate) ─────────────
        # We check the best cosine score among reranked results, not just position 0,
        # because reranking may have promoted a chunk that scored well on BM25
        # but has a mediocre cosine score.
        best_cosine = max(r["cosine_score"] for r in reranked)
        if best_cosine < self._settings.min_similarity_threshold:
            logger.info(
                "Refusal: best_cosine=%.3f < threshold=%.3f",
                best_cosine, self._settings.min_similarity_threshold,
            )
            return self._refusal("insufficient_evidence", intent)

        # ── Step 8: Build citations ───────────────────────────────────────────
        citations, prompt_chunks = self._build_citations(reranked)

        # ── Step 9: Answer generation with intent-appropriate template ────────
        build_prompt = _TEMPLATE_MAP.get(intent.sub, factual.build_prompt)
        messages = build_prompt(question, prompt_chunks)

        answer = await self._client.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
        )

        return QueryResponse(
            answer=answer,
            citations=citations,
            intent=intent,
            refused=False,
            grounded=True,  # answer was built from retrieved document chunks
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _handle_conversational(
        self, question: str, intent: IntentResult
    ) -> QueryResponse:
        """Direct LLM response for chitchat — no retrieval."""
        messages = conversational.build_prompt(question)
        answer = await self._client.chat(messages=messages, temperature=0.5, max_tokens=300)
        return QueryResponse(answer=answer, intent=intent, refused=False)

    def _refusal(self, reason: str, intent: IntentResult) -> QueryResponse:
        """Construct a standardised refusal response."""
        messages = {
            "no_relevant_documents": "I could not find relevant information in the knowledge base to answer your question.",
            "insufficient_evidence": "The evidence I found does not meet the required confidence threshold.  I cannot answer this question reliably based on the available documents.",
            "empty_store": "No documents have been ingested yet.",
        }
        return QueryResponse(
            answer=messages.get(reason, "I cannot answer this question."),
            refused=True,
            refusal_reason=reason,
            intent=intent,
        )

    def _build_citations(
        self, reranked: list[dict]
    ) -> tuple[list[Citation], list[dict]]:
        """
        Convert reranked retrieval results into Citation objects and the chunk
        dicts that prompt templates expect.

        citation_index starts at 1 (human-readable [1], [2], …).
        """
        citations: list[Citation] = []
        prompt_chunks: list[dict] = []

        for i, result in enumerate(reranked):
            citation_index = i + 1
            meta = self._store.get_chunk(result["index"])

            # Excerpt: first 200 characters of the chunk text
            excerpt = textwrap.shorten(meta.text, width=200, placeholder="…")

            citations.append(
                Citation(
                    index=citation_index,
                    source_file=meta.source_file,
                    page_number=meta.page_number,
                    section_header=meta.section_header,
                    chunk_index=meta.chunk_index,
                    similarity_score=round(result["cosine_score"], 4),
                    excerpt=excerpt,
                )
            )
            prompt_chunks.append(
                {
                    "text": meta.text,
                    "citation_index": citation_index,
                    "source_file": meta.source_file,
                }
            )

        return citations, prompt_chunks
