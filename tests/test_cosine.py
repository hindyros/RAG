"""
Unit tests for the cosine similarity index.
"""

import numpy as np
import pytest

from app.retrieval.cosine import CosineIndex


def random_unit(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestCosineIndex:
    def test_empty_index_returns_empty(self):
        idx = CosineIndex()
        scores = idx.get_scores(random_unit(8))
        assert len(scores) == 0

    def test_identical_vectors_score_one(self):
        idx = CosineIndex()
        v = random_unit(16)
        idx.add(np.array([v]))
        scores = idx.get_scores(v)
        assert pytest.approx(scores[0], abs=1e-5) == 1.0

    def test_orthogonal_vectors_score_near_zero(self):
        idx = CosineIndex()
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        idx.add(np.array([v2]))
        scores = idx.get_scores(v1)
        assert pytest.approx(scores[0], abs=1e-5) == 0.0

    def test_scores_length_matches_corpus(self):
        idx = CosineIndex()
        matrix = np.random.randn(5, 8).astype(np.float32)
        idx.add(matrix)
        scores = idx.get_scores(random_unit(8))
        assert len(scores) == 5

    def test_top_k_sorted_descending(self):
        idx = CosineIndex()
        query = random_unit(16)
        # Insert query itself as one of the docs (will score 1.0)
        matrix = np.vstack([
            query,
            np.random.randn(9, 16).astype(np.float32),
        ])
        idx.add(matrix)
        results = idx.get_top_k(query, k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)
        # The query itself should be rank 0
        assert results[0][0] == 0

    def test_incremental_add(self):
        idx = CosineIndex()
        idx.add(np.random.randn(3, 8).astype(np.float32))
        idx.add(np.random.randn(2, 8).astype(np.float32))
        assert idx.size == 5
