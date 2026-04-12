"""
Health / readiness endpoint: GET /health

Returns the current state of the vector store so the Lovable frontend can
show meaningful status information (e.g. "0 documents indexed" on first load).
"""

from fastapi import APIRouter, Depends

from app.api.schemas import HealthResponse
from app.dependencies import get_vector_store
from app.store.vector_store import VectorStore

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health and store statistics",
)
async def health_endpoint(
    store: VectorStore = Depends(get_vector_store),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        total_chunks=store.total_chunks,
        total_documents=store.total_documents,
    )
