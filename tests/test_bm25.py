"""
Unit tests for the BM25 implementation.

These tests verify the mathematical properties of BM25 scoring — they do not
call any external API and have no dependencies beyond numpy.
"""

import math

import numpy as np
import pytest

from app.retrieval.bm25 import BM25


def make_index(*docs: str) -> BM25:
    idx = BM25()
    idx.add_documents(list(docs))
    return idx


class TestBM25Basics:
    def test_empty_index_returns_empty_scores(self):
        idx = BM25()
        scores = idx.get_scores("anything")
        assert len(scores) == 0

    def test_scores_length_matches_corpus(self):
        idx = make_index("the quick brown fox", "jumps over the lazy dog")
        scores = idx.get_scores("fox")
        assert len(scores) == 2

    def test_exact_match_scores_higher(self):
        idx = make_index(
            "machine learning is transforming the world",
            "the weather is sunny today",
        )
        scores = idx.get_scores("machine learning")
        assert scores[0] > scores[1], "Doc with query terms should score higher"

    def test_non_matching_document_scores_zero(self):
        idx = make_index("alpha beta gamma", "delta epsilon zeta")
        scores = idx.get_scores("omega")
        assert np.all(scores == 0.0), "OOV query should produce all-zero scores"

    def test_get_top_k_returns_correct_count(self):
        idx = make_index("a b c", "d e f", "g h i", "a b", "c d")
        results = idx.get_top_k("a b c", k=3)
        assert len(results) == 3

    def test_get_top_k_sorted_descending(self):
        idx = make_index("python is great", "java is okay", "python python python")
        results = idx.get_top_k("python", k=3)
        scores_ordered = [score for _, score in results]
        assert scores_ordered == sorted(scores_ordered, reverse=True)

    def test_k_larger_than_corpus_returns_all(self):
        idx = make_index("doc one", "doc two")
        results = idx.get_top_k("doc", k=100)
        assert len(results) == 2


class TestBM25IDFProperties:
    def test_idf_decreases_with_more_occurrences(self):
        """A term in more documents should have lower IDF."""
        idx = make_index(
            "cat sat on the mat",
            "cat in the hat",
            "cat chased the mouse",
            "dog barked at cat",   # 4 docs contain 'cat'
        )
        # 'cat' appears in all 4 docs; its IDF should be low
        cat_idf = idx._idf.get("cat", 0)
        # 'sat' appears in 1 doc; its IDF should be higher
        sat_idf = idx._idf.get("sat", 0)
        assert sat_idf > cat_idf


class TestBM25Serialisation:
    def test_roundtrip_preserves_scores(self):
        idx = make_index("retrieval augmented generation", "language model inference")
        original_scores = idx.get_scores("retrieval generation")

        state = idx.to_dict()
        restored = BM25.from_dict(state)
        restored_scores = restored.get_scores("retrieval generation")

        np.testing.assert_array_almost_equal(original_scores, restored_scores)

    def test_roundtrip_preserves_size(self):
        idx = make_index("doc a", "doc b", "doc c")
        restored = BM25.from_dict(idx.to_dict())
        assert restored.size == 3
