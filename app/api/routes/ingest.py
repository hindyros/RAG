"""
Ingestion endpoint: POST /ingest

Accepts one or more PDF files as multipart/form-data and runs them through
the ingestion pipeline.  The endpoint is intentionally simple — it delegates
all logic to app/ingestion/pipeline.py and returns structured status to the
caller (the Lovable frontend).

Security considerations:
- File type is validated by checking the MIME type AND the PDF magic bytes
  (%PDF header).  A renamed .exe with Content-Type: application/pdf would
  fail the magic byte check.
- Maximum file size is enforced before reading into memory (early rejection
  avoids memory exhaustion).
- Filenames are sanitised — they are stored only as metadata strings, never
  used as file system paths.
"""

import logging
import re

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.schemas import IngestResponse
from app.config import Settings, get_settings
from app.dependencies import get_mistral_client, get_vector_store
from app.ingestion.pipeline import ingest_files
from app.llm.base import LLMClient
from app.store.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter()

_PDF_MAGIC = b"%PDF"
_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB per file

# Allowed characters in filenames stored as metadata (letters, digits, spaces, hyphens, dots)
_SAFE_FILENAME_RE = re.compile(r"[^\w\s\-.]", re.UNICODE)


def _sanitise_filename(name: str) -> str:
    """Strip unsafe characters from a filename used as metadata."""
    # Replace unsafe chars with underscore, normalise whitespace
    safe = _SAFE_FILENAME_RE.sub("_", name).strip()
    return safe or "unnamed.pdf"


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest one or more PDF files",
    description=(
        "Upload PDF files for text extraction, chunking, and embedding.  "
        "Duplicate files (identified by filename) are silently skipped.  "
        "Returns a summary of how many chunks were added."
    ),
)
async def ingest_endpoint(
    files: list[UploadFile] = File(..., description="One or more PDF files"),
    store: VectorStore = Depends(get_vector_store),
    client: LLMClient = Depends(get_mistral_client),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file is required.",
        )

    validated: list[tuple[str, bytes]] = []

    for upload in files:
        # ── MIME type check ───────────────────────────────────────────────────
        if upload.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File '{upload.filename}' is not a PDF (got {upload.content_type}).",
            )

        raw = await upload.read()

        # ── Size check ────────────────────────────────────────────────────────
        if len(raw) > _MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{upload.filename}' exceeds the 50 MB limit.",
            )

        # ── Magic bytes check (defence against spoofed MIME types) ───────────
        if not raw.startswith(_PDF_MAGIC):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File '{upload.filename}' does not appear to be a valid PDF.",
            )

        filename = _sanitise_filename(upload.filename or "unnamed.pdf")
        validated.append((filename, raw))

    # Run the ingestion pipeline
    results = await ingest_files(validated, store, client, settings)

    total_added = sum(r.chunks_added for r in results)
    files_processed = sum(1 for r in results if not r.skipped)

    logger.info(
        "Ingest complete: %d/%d files processed, %d chunks added.",
        files_processed, len(results), total_added,
    )

    document_ids = [r.document_id for r in results if not r.skipped]

    return IngestResponse(
        message=f"Processed {files_processed} file(s), added {total_added} chunks.",
        files_processed=files_processed,
        chunks_added=total_added,
        total_chunks_in_store=store.total_chunks,
        document_ids=document_ids,
    )
