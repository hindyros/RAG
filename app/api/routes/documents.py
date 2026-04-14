"""
Document management endpoints.

GET  /documents                  — list all ingested documents
DELETE /documents/{document_id}  — remove a document and all its chunks
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import DeleteDocumentResponse, DocumentInfo, DocumentListResponse
from app.dependencies import get_vector_store
from app.store.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents")


@router.get("", response_model=DocumentListResponse, summary="List all ingested documents")
async def list_documents(store: VectorStore = Depends(get_vector_store)) -> DocumentListResponse:
    docs = store.list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total_documents=store.total_documents,
        total_chunks=store.total_chunks,
    )


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document and all its chunks",
)
async def delete_document(
    document_id: str,
    store: VectorStore = Depends(get_vector_store),
) -> DeleteDocumentResponse:
    try:
        chunks_removed = store.delete_document(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    store.save()

    return DeleteDocumentResponse(
        message=f"Document '{document_id}' deleted ({chunks_removed} chunks removed).",
        chunks_removed=chunks_removed,
        total_chunks_in_store=store.total_chunks,
    )
