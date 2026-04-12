"""
Unit tests for hybrid RRF retrieval fusion.
"""

import pytest

from app.retrieval.hybrid import reciprocal_rank_fusion


class TestRRF:
    def test_document_in_both_lists_scores_higher(self):
        """A doc appearing in both lists should have a higher score than one in just one."""
        dense = [(0, 0.9), (1, 0.8), (2, 0.7)]
        sparse = [(1, 5.0), (3, 4.0), (0, 3.0)]
        fused = reciprocal_rank_fusion([dense, sparse])
        scores = dict(fused)

        # Doc 0 appears at rank 0 in dense and rank 2 in sparse → should score well
        # Doc 3 appears only in sparse at rank 1 → lower combined score than doc 0 or 1
        assert scores[0] > scores[3]

    def test_output_sorted_descending(self):
        dense = [(0, 1.0), (1, 0.9), (2, 0.8)]
        sparse = [(2, 2.0), (0, 1.5), (1, 1.0)]
        fused = reciprocal_rank_fusion([dense, sparse])
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_single_list_fusion(self):
        """Fusion of one list is just that list's ranking."""
        single = [(0, 1.0), (1, 0.5), (2, 0.2)]
        fused = reciprocal_rank_fusion([single])
        # Order should be preserved
        assert [idx for idx, _ in fused] == [0, 1, 2]

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_rrf_score_formula(self):
        """Verify RRF score matches the formula: 1/(k+rank+1)."""
        k = 60
        dense = [(42, 0.99)]  # rank 0 in dense
        fused = reciprocal_rank_fusion([dense], k=k)
        expected_score = 1.0 / (k + 0 + 1)
        assert pytest.approx(fused[0][1], rel=1e-6) == expected_score
