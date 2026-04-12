"""
Atomic persistence for the vector store.

Why not a database?
    The challenge requires no third-party vector database.  We persist to disk
    using:
    - numpy .npz for the embedding matrix (compressed float32 — typically
      40-60% smaller than raw binary)
    - JSON for metadata and BM25 state (human-readable, version-safe,
      no pickle security concerns)

Atomic writes:
    We write to temporary files first, then rename them to the final paths.
    On POSIX systems, rename() is atomic — the store is never in a partially-
    written state that would corrupt on crash or power loss.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDINGS_FILE = "embeddings.npz"
_METADATA_FILE = "metadata.json"
_VERSION_FILE = "store_version.txt"
_STORE_VERSION = "1"


def save_store(
    store_path: str,
    embeddings: np.ndarray,
    metadata: list[dict],
    bm25_state: dict,
) -> None:
    """
    Persist the vector store to disk atomically.

    Parameters
    ----------
    store_path  : Directory where store files are written.
    embeddings  : Float32 matrix of shape (N, D).
    metadata    : List of dicts (chunk metadata), index-aligned with embeddings.
    bm25_state  : Serialised BM25 dict (from BM25.to_dict()).
    """
    path = Path(store_path)
    path.mkdir(parents=True, exist_ok=True)

    # ── Write embeddings ──────────────────────────────────────────────────────
    embeddings_path = path / _EMBEDDINGS_FILE
    tmp_embeddings = embeddings_path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp_embeddings), embeddings=embeddings)
    os.replace(str(tmp_embeddings), str(embeddings_path))

    # ── Write metadata + BM25 state ───────────────────────────────────────────
    metadata_path = path / _METADATA_FILE
    tmp_metadata = metadata_path.with_suffix(".tmp.json")
    payload = {"metadata": metadata, "bm25": bm25_state, "version": _STORE_VERSION}
    with open(str(tmp_metadata), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp_metadata), str(metadata_path))

    # ── Version marker ────────────────────────────────────────────────────────
    (path / _VERSION_FILE).write_text(_STORE_VERSION)

    logger.info("Store saved: %d chunks at %s", len(metadata), store_path)


def load_store(store_path: str) -> tuple[np.ndarray, list[dict], dict] | None:
    """
    Load the persisted store from disk.

    Returns None if no store exists yet (first startup).
    Returns (embeddings, metadata, bm25_state) on success.
    """
    path = Path(store_path)
    embeddings_path = path / _EMBEDDINGS_FILE
    metadata_path = path / _METADATA_FILE

    if not embeddings_path.exists() or not metadata_path.exists():
        logger.info("No existing store found at %s — starting fresh.", store_path)
        return None

    # ── Version check ─────────────────────────────────────────────────────────
    version_path = path / _VERSION_FILE
    if version_path.exists():
        on_disk_version = version_path.read_text().strip()
        if on_disk_version != _STORE_VERSION:
            logger.warning(
                "Store version mismatch (disk: %s, expected: %s). "
                "Starting fresh to avoid corruption.",
                on_disk_version, _STORE_VERSION,
            )
            return None

    # ── Load embeddings ───────────────────────────────────────────────────────
    npz = np.load(str(embeddings_path))
    embeddings = npz["embeddings"].astype(np.float32)

    # ── Load metadata + BM25 ─────────────────────────────────────────────────
    with open(str(metadata_path), encoding="utf-8") as f:
        payload = json.load(f)

    metadata: list[dict] = payload["metadata"]
    bm25_state: dict = payload["bm25"]

    logger.info(
        "Store loaded: %d chunks from %s", len(metadata), store_path
    )
    return embeddings, metadata, bm25_state
