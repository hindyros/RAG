"""
Central configuration module.

All tuneable parameters live here and are read from environment variables
(or .env file).  Importing `get_settings()` anywhere in the app gives a
single typed view of the current configuration.

LLM_PROVIDER controls which API is used for embeddings and chat:
  "openai"   — requires OPENAI_API_KEY
  "mistral"  — requires MISTRAL_API_KEY

Switching providers with an existing store will fail at query time because
the stored embedding dimensions will not match the new provider's output.
Delete ./data/store and re-ingest all documents after changing providers.
"""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Provider selection ───────────────────────────────────────────────────
    llm_provider: str = "openai"   # "openai" | "mistral"

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"  # 1536-dim
    openai_chat_model: str = "gpt-4o-mini"

    # ── Mistral AI ───────────────────────────────────────────────────────────
    mistral_api_key: str = ""
    mistral_embed_model: str = "mistral-embed"          # 1024-dim
    mistral_chat_model: str = "mistral-large-latest"

    # ── Storage ──────────────────────────────────────────────────────────────
    store_path: str = "./data/store"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    rrf_k_constant: int = 60

    # ── Quality gates ────────────────────────────────────────────────────────
    min_similarity_threshold: float = 0.35
    intent_confidence_min: float = 0.60

    # ── Batching ─────────────────────────────────────────────────────────────
    max_embed_batch_size: int = 32

    @model_validator(mode="after")
    def _check_api_key(self) -> "Settings":
        provider = self.llm_provider.lower()
        if provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                "Add OPENAI_API_KEY to your .env file."
            )
        if provider == "mistral" and not self.mistral_api_key:
            raise ValueError(
                "LLM_PROVIDER is 'mistral' but MISTRAL_API_KEY is not set. "
                "Add MISTRAL_API_KEY to your .env file."
            )
        if provider not in ("openai", "mistral"):
            raise ValueError(
                f"LLM_PROVIDER must be 'openai' or 'mistral', got '{provider}'."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
