"""
FastAPI application factory.

The lifespan context manager handles startup and shutdown:
- Startup: load the vector store from disk, create the LLM client
  (OpenAI or Mistral depending on LLM_PROVIDER), wire up the QueryPipeline.
- Shutdown: flush the connection pool gracefully.
"""

import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import documents, health, ingest, query, visualize
from app.config import get_settings
from app.hallucination.checker import HallucinationChecker
from app.llm.base import LLMClient
from app.llm.client import MistralClient
from app.llm.openai_client import OpenAIClient
from app.query.pipeline import QueryPipeline
from app.store.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_llm_client(settings) -> LLMClient:
    """Instantiate the right LLM client based on LLM_PROVIDER."""
    provider = settings.llm_provider.lower()
    if provider == "openai":
        logger.info("LLM provider: OpenAI (chat=%s, embed=%s)",
                    settings.openai_chat_model, settings.openai_embed_model)
        return OpenAIClient(settings=settings)
    else:
        logger.info("LLM provider: Mistral (chat=%s, embed=%s)",
                    settings.mistral_chat_model, settings.mistral_embed_model)
        return MistralClient(settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    store = VectorStore(store_path=settings.store_path)
    store.load()

    client: LLMClient = _build_llm_client(settings)

    checker = HallucinationChecker()
    pipeline = QueryPipeline(store=store, client=client, settings=settings, checker=checker)

    app.state.vector_store = store
    app.state.llm_client = client
    app.state.hallucination_checker = checker
    app.state.query_pipeline = pipeline
    app.state.viz_cache = None

    logger.info(
        "Startup complete — %d chunks across %d documents in store.",
        store.total_chunks, store.total_documents,
    )

    yield

    await client.close()
    logger.info("LLM client closed. Shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Pipeline API",
        description=(
            "Retrieval-Augmented Generation over PDF documents.  "
            "Supports OpenAI and Mistral AI — set LLM_PROVIDER in .env.  "
            "Implements hybrid BM25 + dense retrieval, HyDE query transformation, "
            "LLM reranking, and intent-aware answer shaping."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(ingest.router, tags=["Ingestion"])
    app.include_router(query.router, tags=["Query"])
    app.include_router(documents.router, tags=["Documents"])
    app.include_router(visualize.router, tags=["Visualize"])

    # Must be mounted LAST — StaticFiles on "/" would shadow API routes if first
    app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

    return app


app = create_app()
