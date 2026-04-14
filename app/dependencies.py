"""
FastAPI dependency providers.

FastAPI's Depends() system is used for dependency injection of the three
application-wide singletons:
- VectorStore   : the in-memory index
- MistralClient : the HTTP client (connection pool)
- QueryPipeline : orchestrator (wraps the two above)

All three are initialised once at startup (via the lifespan context in main.py)
and stored on app.state.  The Depends() functions here read from app.state,
making them testable by simply setting app.state.{name} in test setup.
"""

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.hallucination.checker import HallucinationChecker
from app.llm.client import MistralClient
from app.query.pipeline import QueryPipeline
from app.store.vector_store import VectorStore


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_mistral_client(request: Request) -> MistralClient:
    return request.app.state.mistral_client


def get_query_pipeline(request: Request) -> QueryPipeline:
    return request.app.state.query_pipeline


def get_hallucination_checker(request: Request) -> HallucinationChecker:
    return request.app.state.hallucination_checker
