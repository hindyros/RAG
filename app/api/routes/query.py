"""
Query endpoint: POST /query

Accepts a JSON body with a `question` field and returns a structured answer
with citations.  All pipeline logic lives in app/query/pipeline.py — this
file is purely the HTTP adapter.
"""

import logging

from fastapi import APIRouter, Depends

from app.api.schemas import QueryRequest, QueryResponse
from app.config import Settings, get_settings
from app.dependencies import get_mistral_client, get_query_pipeline, get_vector_store
from app.llm.base import LLMClient
from app.query.pipeline import QueryPipeline
from app.store.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    description=(
        "Send a natural-language question.  The system detects intent, retrieves "
        "relevant chunks from ingested PDFs, reranks them, and generates a cited answer.  "
        "Returns 'refused=true' with a reason when the retrieved evidence does not meet "
        "the confidence threshold."
    ),
)
async def query_endpoint(
    body: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
) -> QueryResponse:
    logger.info("Query received: %.80s", body.question)
    response = await pipeline.run(body.question)
    logger.info(
        "Query answered: refused=%s intent=%s citations=%d",
        response.refused,
        response.intent.sub if response.intent else "n/a",
        len(response.citations),
    )
    return response
