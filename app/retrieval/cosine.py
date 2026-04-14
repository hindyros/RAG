"""
Vectorised cosine similarity for dense retrieval.

All computation is done with NumPy — no FAISS, Annoy, or other vector
search library.  For a corpus of up to ~500k chunks this is fast enough:
a single matrix-vector multiply over 100k × 1024 float32 values takes
< 100ms on a single CPU core when NumPy is linked against BLAS (default
on macOS via Accelerate and on Linux via OpenBLAS).

The pre-normalised matrix approach (store unit vectors, do a dot product)
avoids recomputing norms on every query — norms are recomputed only when
chunks are added to the store.
"""

import numpy as np


class CosineIndex:
    """
    Maintains a pre-normalised embedding matrix for O(N×D) query time.

    The matrix is updated incrementally as chunks are ingested.  The unit
    vectors are cached; any add invalidates and regenerates the cache.
    """

    def __init__(self) -> None:
        # Raw embeddings — shape (N, D)
        self._matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        # Row-normalised version — same shape, recomputed on each add
        self._unit_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def add(self, embeddings: np.ndarray) -> None:
        """
        Append new embedding rows and refresh the unit-vector cache.

        Parameters
        ----------
        embeddings : shape (M, D) — the M new chunk embeddings to add.
        """
        if self._matrix.size == 0:
            self._matrix = embeddings.astype(np.float32)
        else:
            self._matrix = np.vstack(
                [self._matrix, embeddings.astype(np.float32)]
            )
        self._refresh_unit_matrix()

    def _refresh_unit_matrix(self) -> None:
        """Recompute unit-vector cache after any corpus change."""
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors (shouldn't happen with real text)
        norms = np.where(norms == 0, 1.0, norms)
        self._unit_matrix = self._matrix / norms

    def get_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between `query_vec` and every indexed chunk.

        Returns a float32 array of length N (corpus size).  Values are in
        [-1, 1] but in practice all embeddings from Mistral-embed are positive,
        so scores cluster in [0, 1].

        Parameters
        ----------
        query_vec : shape (D,) — the query embedding vector.
        """
        if self._unit_matrix.size == 0:
            return np.array([], dtype=np.float32)

        # Normalise the query vector
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            return np.zeros(len(self._unit_matrix), dtype=np.float32)
        query_unit = query_vec.astype(np.float32) / norm

        # Single matrix-vector multiply: O(N × D)
        return self._unit_matrix @ query_unit

    def get_top_k(
        self, query_vec: np.ndarray, k: int
    ) -> list[tuple[int, float]]:
        """
        Return (index, cosine_score) pairs for the top-k most similar chunks.

        Uses argpartition (O(N)) rather than full argsort (O(N log N)) for
        efficiency on large corpora.
        """
        scores = self.get_scores(query_vec)
        if len(scores) == 0:
            return []

        k = min(k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def remove_rows(self, indices: list[int]) -> None:
        """
        Delete the rows at the given indices and refresh the unit-vector cache.

        Parameters
        ----------
        indices : positions to remove (order does not matter, duplicates ignored).
        """
        if not indices or self._matrix.size == 0:
            return
        keep = np.ones(len(self._matrix), dtype=bool)
        for i in indices:
            keep[i] = False
        self._matrix = self._matrix[keep]
        if self._matrix.size > 0:
            self._refresh_unit_matrix()
        else:
            self._matrix = np.empty((0, 0), dtype=np.float32)
            self._unit_matrix = np.empty((0, 0), dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self._matrix)

    @property
    def matrix(self) -> np.ndarray:
        """Raw embedding matrix — used by persistence layer to save to disk."""
        return self._matrix
