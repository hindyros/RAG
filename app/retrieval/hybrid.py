"""
Hybrid retrieval combining dense (cosine) and sparse (BM25) search via
Reciprocal Rank Fusion (RRF).

Why hybrid?
    Dense retrieval (embeddings) excels at semantic similarity — it finds
    documents that "mean the same thing" even when they use different words.
    But it can miss exact keyword matches, especially for rare technical terms
    or proper nouns that are underrepresented in the embedding space.

    Sparse retrieval (BM25) excels at exact keyword matches but misses
    paraphrases and synonyms.

    Hybrid search with RRF consistently outperforms either alone across
    BEIR and other information retrieval benchmarks — typically +3 to +8
    points nDCG@10.

Why RRF over score interpolation?
    Score interpolation (α × semantic_score + (1-α) × bm25_score) requires
    tuning α and assumes the scores are on comparable scales — they are not.
    Cosine similarity is bounded in [-1, 1]; BM25 scores are unbounded and
    corpus-size-dependent.

    RRF is rank-based and therefore scale-free.  The formula
        rrf(d) = Σ 1 / (k + rank_i(d))
    with k=60 (from Cormack et al., SIGIR 2009) is robust across domains
    without any tuning.

The cosine scores are preserved alongside RRF scores because the citation
threshold check (POST /query) needs raw cosine similarity — RRF scores are
not comparable to a fixed threshold.
"""

import numpy as np

from app.retrieval.bm25 import BM25
from app.retrieval.cosine import CosineIndex


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : Each inner list is [(doc_index, score), ...] sorted by
                   score descending.  Scores are only used for ordering; the
                   RRF formula uses rank positions.
    k            : The RRF constant (default 60 from the original paper).
                   Higher k → less aggressive rank weighting; top results
                   from one system count for less compared to lower-ranked
                   matches that appear in multiple lists.

    Returns
    -------
    A fused list of (doc_index, rrf_score) pairs sorted by rrf_score descending.
    """
    rrf_scores: dict[int, float] = {}

    for ranked_list in ranked_lists:
        for rank, (doc_idx, _score) in enumerate(ranked_list):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(
    query_embedding: np.ndarray,
    query_text: str,
    cosine_index: CosineIndex,
    bm25_index: BM25,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Run both retrievers and fuse their results with RRF.

    Parameters
    ----------
    query_embedding : Dense vector for the query (or HyDE document vector).
    query_text      : Raw query string for BM25 tokenisation.
    cosine_index    : Pre-built dense index.
    bm25_index      : Pre-built sparse index.
    top_k           : Number of candidates to fetch from each retriever before
                      fusion.  More candidates → better fusion quality but
                      higher latency.
    rrf_k           : RRF constant forwarded to reciprocal_rank_fusion().

    Returns
    -------
    List of result dicts (sorted by rrf_score descending):
        {
            "index"         : int,   # position in the vector store
            "rrf_score"     : float, # fused rank score (not comparable to cosine)
            "cosine_score"  : float, # preserved for the similarity threshold check
            "bm25_score"    : float, # preserved for transparency / debugging
        }
    """
    # Retrieve top-k candidates from each retriever independently
    dense_results = cosine_index.get_top_k(query_embedding, k=top_k)
    sparse_results = bm25_index.get_top_k(query_text, k=top_k)

    # Build lookup dicts so we can attach component scores to fused results
    cosine_scores = {idx: score for idx, score in dense_results}
    bm25_scores = {idx: score for idx, score in sparse_results}

    # Fuse the two ranked lists
    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=rrf_k,
    )

    return [
        {
            "index": idx,
            "rrf_score": rrf_score,
            "cosine_score": cosine_scores.get(idx, 0.0),
            "bm25_score": bm25_scores.get(idx, 0.0),
        }
        for idx, rrf_score in fused
    ]
