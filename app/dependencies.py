"""
FastAPI dependency providers.

All application-wide singletons are initialised once at startup (via the
lifespan context in main.py) and stored on app.state.  The Depends()
functions here read from app.state, making them testable by simply setting
app.state.{name} in test setup.
"""

from fastapi import Request

from app.config import Settings, get_settings
from app.hallucination.checker import HallucinationChecker
from app.llm.base import LLMClient
from app.query.pipeline import QueryPipeline
from app.store.vector_store import VectorStore


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


# Backwards-compatible alias used by existing route files
get_mistral_client = get_llm_client


def get_query_pipeline(request: Request) -> QueryPipeline:
    return request.app.state.query_pipeline


def get_hallucination_checker(request: Request) -> HallucinationChecker:
    return request.app.state.hallucination_checker
