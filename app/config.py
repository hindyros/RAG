"""
Central configuration module.

All tuneable parameters live here and are read from environment variables
(or .env file).  Importing `settings` anywhere in the app gives a single
typed view of the current configuration, so there are no magic strings
scattered across the codebase.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Mistral AI ───────────────────────────────────────────────────────────
    mistral_api_key: str
    mistral_embed_model: str = "mistral-embed"
    mistral_chat_model: str = "mistral-large-latest"

    # ── Storage ──────────────────────────────────────────────────────────────
    store_path: str = "./data/store"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_retrieval: int = 10   # candidates per retriever before fusion
    top_k_rerank: int = 5       # survivors after reranking fed to the LLM
    rrf_k_constant: int = 60    # RRF denominator constant (Cormack et al.)

    # ── Quality gates ────────────────────────────────────────────────────────
    min_similarity_threshold: float = 0.35  # below this → refusal
    intent_confidence_min: float = 0.60     # below this → default to factual

    # ── Batching ─────────────────────────────────────────────────────────────
    max_embed_batch_size: int = 32


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
