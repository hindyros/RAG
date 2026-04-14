"""
Ingestion pipeline orchestrator.

Wires together: PDF extraction → chunking → embedding → vector store.

The pipeline is intentionally sequential within a single file upload (each
chunk must be embedded before the next batch) but processes multiple files
in the order they are received.  Parallel embedding of multiple files would
require more complex concurrency management and is left as a future optimisation.

Embedding in batches:
    The Mistral embed API accepts up to max_embed_batch_size inputs per call.
    We batch chunk texts to minimise round-trips while staying within the limit.
    Each batch is a separate API call; results are concatenated in order.
"""

import logging
import uuid
from dataclasses import dataclass

from app.config import Settings
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.pdf_extractor import extract_pdf
from app.llm.base import LLMClient
from app.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    filename: str
    document_id: str          # UUID assigned to this file; "" when skipped
    chunks_added: int
    skipped: bool = False     # True when the file was already indexed
    skip_reason: str = ""


async def ingest_files(
    files: list[tuple[str, bytes]],  # (filename, raw_bytes) pairs
    store: VectorStore,
    client: LLMClient,
    settings: Settings,
) -> list[IngestionResult]:
    """
    Ingest one or more PDF files into the vector store.

    Parameters
    ----------
    files    : List of (filename, raw_bytes) tuples from the upload endpoint.
    store    : The live VectorStore instance.
    client   : Mistral API client for embedding.
    settings : App configuration (chunk size, batch size, etc.).

    Returns
    -------
    One IngestionResult per file.
    """
    results: list[IngestionResult] = []

    for filename, raw_bytes in files:
        result = await _ingest_one(filename, raw_bytes, store, client, settings)
        results.append(result)

    # Persist after processing all files in the request (one disk write per request)
    store.save()

    return results


async def _ingest_one(
    filename: str,
    raw_bytes: bytes,
    store: VectorStore,
    client: LLMClient,
    settings: Settings,
) -> IngestionResult:
    """Process a single PDF file end-to-end."""

    # ── Duplicate detection ───────────────────────────────────────────────────
    if store.already_ingested(filename):
        logger.info("Skipping %s — already in store.", filename)
        return IngestionResult(
            filename=filename,
            document_id="",
            chunks_added=0,
            skipped=True,
            skip_reason="already_indexed",
        )

    # Assign a stable UUID to this document before any processing so it can
    # be threaded through to every chunk and used for future document-level
    # operations (deletion, re-indexing) without relying on the filename.
    document_id = str(uuid.uuid4())
    logger.info("Ingesting %s (%d bytes) as doc_id=%s.", filename, len(raw_bytes), document_id)

    # ── Extract ───────────────────────────────────────────────────────────────
    extracted = extract_pdf(raw_bytes, filename)
    if not extracted.blocks:
        logger.warning("%s: no text blocks extracted — skipping.", filename)
        return IngestionResult(
            filename=filename,
            document_id="",
            chunks_added=0,
            skipped=True,
            skip_reason="no_text_extracted",
        )

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunks: list[Chunk] = chunk_document(
        extracted,
        chunk_size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    logger.info("%s: %d chunks produced.", filename, len(chunks))

    if not chunks:
        return IngestionResult(
            filename=filename,
            document_id="",
            chunks_added=0,
            skipped=True,
            skip_reason="chunking_produced_no_output",
        )

    # ── Embed in batches ──────────────────────────────────────────────────────
    all_embeddings: list[list[float]] = []
    batch_size = settings.max_embed_batch_size
    texts = [c.text for c in chunks]

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        batch_embeddings = await client.embed(batch)
        all_embeddings.extend(batch_embeddings)
        logger.debug(
            "%s: embedded batch %d-%d.",
            filename, batch_start, batch_start + len(batch),
        )

    # ── Add to store ──────────────────────────────────────────────────────────
    added = store.add_chunks(chunks, all_embeddings, document_id=document_id)

    return IngestionResult(filename=filename, document_id=document_id, chunks_added=added)
