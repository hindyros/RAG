"""
In-memory vector store — the single source of truth for all indexed chunks.

This class owns three parallel data structures that must stay in sync:
    1. CosineIndex  — float32 embedding matrix for dense retrieval
    2. BM25 index   — tokenised corpus for sparse retrieval
    3. metadata     — list of dicts with source_file, page_number, etc.

All three are indexed by the same integer position (0, 1, 2, ...) so that
retrieval results (which carry integer indices) can be looked up in O(1).

Thread safety: FastAPI runs request handlers in an async event loop.  Python's
GIL protects dict/list operations, and NumPy operations are performed in
one-shot array calls (no partial state visible).  For a single-process
deployment this is sufficient.  A production multi-process deployment would
use a shared-memory backend (Redis, PostgreSQL pgvector, etc.) instead.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from app.ingestion.chunker import Chunk
from app.retrieval.bm25 import BM25
from app.retrieval.cosine import CosineIndex
from app.store.persistence import load_store, save_store

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """
    All non-embedding information about a chunk.

    Stored as a plain dataclass rather than a dict so that access is typed
    and IDE-navigable, but converted to dict for JSON serialisation.
    """
    source_file: str
    page_number: int
    chunk_index: int
    section_header: str | None
    text: str                   # kept for citation excerpt generation

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "section_header": self.section_header,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkMetadata":
        return cls(
            source_file=d["source_file"],
            page_number=d["page_number"],
            chunk_index=d["chunk_index"],
            section_header=d.get("section_header"),
            text=d["text"],
        )


class VectorStore:
    """
    Central registry for all ingested chunks.

    Usage pattern:
        store = VectorStore(store_path)
        store.load()                         # at app startup
        store.add_chunks(chunks, embeddings) # during ingest
        results = store.retrieve(...)        # during query
        store.save()                         # after each ingest
    """

    def __init__(self, store_path: str) -> None:
        self._store_path = store_path
        self._cosine = CosineIndex()
        self._bm25 = BM25()
        self._metadata: list[ChunkMetadata] = []
        # Track which source files are in the store (for stats)
        self._source_files: set[str] = set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Restore state from disk.  Call once at application startup.
        """
        result = load_store(self._store_path)
        if result is None:
            return  # Fresh start — empty store is the correct state

        embeddings, metadata_dicts, bm25_state = result

        self._cosine.add(embeddings)
        self._bm25 = BM25.from_dict(bm25_state)
        self._metadata = [ChunkMetadata.from_dict(d) for d in metadata_dicts]
        self._source_files = {m.source_file for m in self._metadata}

        logger.info("VectorStore ready: %d chunks across %d files.",
                    len(self._metadata), len(self._source_files))

    def save(self) -> None:
        """
        Persist current state to disk.  Call after each ingest batch.
        """
        if not self._metadata:
            logger.debug("Store is empty — nothing to save.")
            return

        save_store(
            store_path=self._store_path,
            embeddings=self._cosine.matrix,
            metadata=[m.to_dict() for m in self._metadata],
            bm25_state=self._bm25.to_dict(),
        )

    # ── Ingest ────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add a batch of chunks with their precomputed embeddings.

        Parameters
        ----------
        chunks     : Chunk objects from the chunker — provides metadata.
        embeddings : Float vectors from mistral-embed, aligned with chunks.

        Returns
        -------
        Number of chunks added.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must be the same length"
            )

        # Build metadata list and collect texts for BM25
        new_metadata = [
            ChunkMetadata(
                source_file=c.source_file,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                section_header=c.section_header,
                text=c.text,
            )
            for c in chunks
        ]
        texts = [c.text for c in chunks]

        # Update all three parallel structures
        emb_array = np.array(embeddings, dtype=np.float32)
        self._cosine.add(emb_array)
        self._bm25.add_documents(texts)
        self._metadata.extend(new_metadata)
        self._source_files.update(c.source_file for c in chunks)

        logger.info("Added %d chunks. Store total: %d.", len(chunks), len(self._metadata))
        return len(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_cosine_index(self) -> CosineIndex:
        return self._cosine

    def get_bm25_index(self) -> BM25:
        return self._bm25

    def get_chunk(self, index: int) -> ChunkMetadata:
        """Look up a chunk's metadata by its integer index."""
        return self._metadata[index]

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return len(self._metadata)

    @property
    def total_documents(self) -> int:
        return len(self._source_files)

    def already_ingested(self, filename: str) -> bool:
        """Check if a file has already been indexed (to avoid duplicates)."""
        return filename in self._source_files
