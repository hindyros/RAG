"""
FastAPI application factory.

The lifespan context manager handles startup and shutdown:
- Startup: load the vector store from disk (if a previous run persisted data),
  create the Mistral HTTP client, wire up the QueryPipeline.
- Shutdown: flush the connection pool gracefully.

CORS is configured to accept all origins by default so that the Lovable
frontend (hosted on a different domain/port) can call the API without
browser-level blocks.  In production, restrict allowed_origins to the
specific frontend domain.
"""

import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import health, ingest, query
from app.config import get_settings
from app.hallucination.checker import HallucinationChecker
from app.llm.client import MistralClient
from app.query.pipeline import QueryPipeline
from app.store.vector_store import VectorStore

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise singletons on startup; clean up on shutdown.

    Using the lifespan pattern (rather than @app.on_event) is the modern
    FastAPI approach — it works correctly with pytest-asyncio test fixtures
    and avoids deprecation warnings.
    """
    settings = get_settings()

    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Starting RAG API (model: %s).", settings.mistral_chat_model)

    store = VectorStore(store_path=settings.store_path)
    store.load()  # no-op if no data persisted yet

    client = MistralClient(settings=settings)

    checker = HallucinationChecker()
    pipeline = QueryPipeline(store=store, client=client, settings=settings, checker=checker)

    # Attach to app.state so route handlers can access via Depends()
    app.state.vector_store = store
    app.state.mistral_client = client
    app.state.hallucination_checker = checker
    app.state.query_pipeline = pipeline

    logger.info(
        "Startup complete — %d chunks across %d documents in store.",
        store.total_chunks, store.total_documents,
    )

    yield  # Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await client.close()
    logger.info("Mistral client closed. Shutdown complete.")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Pipeline API",
        description=(
            "Retrieval-Augmented Generation over PDF documents using Mistral AI.  "
            "Implements hybrid BM25 + dense retrieval, HyDE query transformation, "
            "LLM reranking, and intent-aware answer shaping."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router, tags=["Health"])
    app.include_router(ingest.router, tags=["Ingestion"])
    app.include_router(query.router, tags=["Query"])

    # ── Static files (UI) — must be mounted LAST so API routes take priority ──
    app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

    return app


app = create_app()
