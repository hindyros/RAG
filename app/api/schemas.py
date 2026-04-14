"""
Pydantic models for all API request/response shapes.

Keeping schemas in one place makes it easy to review the contract the API
exposes to clients (e.g. the Lovable frontend) without hunting across files.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Intent taxonomy ──────────────────────────────────────────────────────────

class PrimaryIntent(str, Enum):
    CONVERSATIONAL = "conversational"
    KNOWLEDGE_SEEKING = "knowledge_seeking"


class SubIntent(str, Enum):
    FACTUAL = "factual"
    LIST = "list"
    COMPARISON = "comparison"
    TABLE = "table"
    CHITCHAT = "chitchat"


class IntentResult(BaseModel):
    primary: PrimaryIntent
    sub: SubIntent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ── Citation ─────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """One cited source chunk attached to an answer."""
    index: int              # [1], [2], … reference number used inline
    source_file: str
    page_number: int
    section_header: Optional[str] = None
    chunk_index: int        # position in the source document
    similarity_score: float = Field(ge=0.0, le=1.0)
    excerpt: str            # short snippet for UI display


# ── Hallucination detection ──────────────────────────────────────────────────

class SentenceScore(BaseModel):
    sentence: str
    score: float = Field(ge=0.0, le=1.0)
    flagged: bool          # True when score < 0.5


class HallucinationResult(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)   # minimum sentence score (conservative)
    consistent: bool                                 # overall_score >= 0.5
    sentences: list[SentenceScore]


# ── Ingestion ────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    message: str
    files_processed: int
    chunks_added: int
    total_chunks_in_store: int
    document_ids: list[str]   # UUID per successfully ingested file


# ── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    intent: Optional[IntentResult] = None
    refused: bool = False
    refusal_reason: Optional[str] = None   # "no_relevant_documents" | "policy"
    grounded: bool = False
    # grounded=True  → answer was built from retrieved document chunks
    # grounded=False → conversational reply, refusal, or empty-store fallback
    # Intended for the UI to show a "sourced from documents" indicator
    hallucination: Optional[HallucinationResult] = None


# ── Documents ────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    document_id: str
    source_file: str
    chunk_count: int
    page_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class DeleteDocumentResponse(BaseModel):
    message: str
    chunks_removed: int
    total_chunks_in_store: int


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    total_chunks: int
    total_documents: int
