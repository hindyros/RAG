"""
BM25Okapi implementation from scratch using only NumPy.

BM25 (Best Match 25) is the standard sparse retrieval algorithm.  We implement
it without any external search library because:
1. The challenge requirements forbid external search libraries.
2. For a document store of reasonable size (< 1M chunks), a pure-Python + NumPy
   BM25 is fast enough and gives us full transparency over the scoring.

Algorithm (BM25+):
    score(q, d) = Σ IDF(t) × TF(t, d)

    IDF(t) = log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        Where N is the corpus size and df_t is the number of documents
        containing term t.  The +1 outside the log keeps IDF ≥ 0 even when
        a term appears in every document.

    TF(t, d) = (f_t,d × (k1 + 1)) / (f_t,d + k1 × (1 - b + b × |d| / avgdl))
        Where f_t,d is raw term frequency in document d, |d| is document length
        in tokens, and avgdl is the average document length in the corpus.

Parameter choices:
    k1 = 1.5  — controls TF saturation.  Values in [1.2, 2.0] are standard;
                 1.5 is the BM25+ default, slightly favouring longer exact matches
                 over short but frequent ones.
    b  = 0.75 — length normalisation strength.  0 means no normalisation; 1 means
                 full normalisation.  0.75 is the universal default from the
                 original Robertson et al. paper.

State that must be persisted between restarts (see store/persistence.py):
    - term_doc_freq : {term: number_of_documents_containing_term}
    - tokenized_corpus : list of token lists, one per document
    - avgdl : average document length
"""

import math
from collections import Counter

import numpy as np

from app.utils.text import tokenize_for_bm25


class BM25:
    """
    BM25Okapi index that supports incremental updates.

    Documents can be added one at a time (as chunks are ingested).  Each add
    triggers an IDF recomputation because IDF is a corpus-global statistic.
    For a write-rarely, read-heavily workload this is acceptable: ingest is
    infrequent; query is the hot path.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        # Tokenized corpus — parallel to the embedding matrix in VectorStore
        self._corpus: list[list[str]] = []

        # document frequency: term → number of docs containing it
        self._doc_freq: dict[str, int] = {}

        # Cached per-document term-frequency Counter objects (avoids re-counting)
        self._tf_cache: list[Counter] = []

        # Precomputed IDF scores, recomputed after every ingest
        self._idf: dict[str, float] = {}

        # Total and average document length (in tokens)
        self._total_length: int = 0
        self.avgdl: float = 0.0

    # ── Corpus management ─────────────────────────────────────────────────────

    def add_document(self, text: str) -> None:
        """
        Tokenize `text` and add it to the index.

        Calling this N times is O(N × vocabulary) because IDF is recomputed
        after each add.  In practice, ingest happens in batches — see
        add_documents() for the batch variant.
        """
        tokens = tokenize_for_bm25(text)
        self._add_tokens(tokens)
        self._recompute_idf()

    def add_documents(self, texts: list[str]) -> None:
        """
        Batch-add documents and recompute IDF once at the end.

        Prefer this over calling add_document() in a loop when ingesting a
        full file — it recomputes IDF once instead of once per chunk.
        """
        for text in texts:
            tokens = tokenize_for_bm25(text)
            self._add_tokens(tokens)
        self._recompute_idf()

    def _add_tokens(self, tokens: list[str]) -> None:
        tf = Counter(tokens)
        self._corpus.append(tokens)
        self._tf_cache.append(tf)

        # Update document frequency for each unique term in this document
        for term in tf:
            self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

        self._total_length += len(tokens)
        n = len(self._corpus)
        self.avgdl = self._total_length / n if n > 0 else 0.0

    def _recompute_idf(self) -> None:
        """
        Recompute IDF for every term in the vocabulary.

        Called after any corpus modification.  Uses Robertson-Ogilvie IDF with
        +1 smoothing outside the log to guarantee non-negative IDF values.
        """
        n = len(self._corpus)
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1)
            for term, df in self._doc_freq.items()
        }

    # ── Query scoring ─────────────────────────────────────────────────────────

    def get_scores(self, query: str) -> np.ndarray:
        """
        Score all documents in the corpus against `query`.

        Returns a float64 NumPy array of length len(corpus), where index i
        corresponds to the i-th document added via add_document(s).

        Returns an empty array if the corpus is empty.
        """
        if not self._corpus:
            return np.array([], dtype=np.float64)

        query_terms = tokenize_for_bm25(query)
        scores = np.zeros(len(self._corpus), dtype=np.float64)

        for term in query_terms:
            if term not in self._idf:
                continue  # OOV term — contributes nothing

            idf = self._idf[term]

            for doc_idx, (tf_counter, doc_tokens) in enumerate(
                zip(self._tf_cache, self._corpus)
            ):
                f = tf_counter.get(term, 0)
                if f == 0:
                    continue

                dl = len(doc_tokens)
                tf_norm = (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                scores[doc_idx] += idf * tf_norm

        return scores

    def get_top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """
        Return the top-k (doc_index, bm25_score) pairs for `query`.

        Sorted by score descending.  Indices are absolute positions in the
        corpus (same as the embedding matrix row index in VectorStore).
        """
        scores = self.get_scores(query)
        if len(scores) == 0:
            return []

        # argpartition is O(N) vs argsort's O(N log N) — faster for large corpora
        k = min(k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    # ── Serialisation helpers (used by store/persistence.py) ─────────────────

    def to_dict(self) -> dict:
        """Serialise index state to a JSON-compatible dict."""
        return {
            "k1": self.k1,
            "b": self.b,
            "corpus": self._corpus,
            "doc_freq": self._doc_freq,
            "tf_cache": [dict(c) for c in self._tf_cache],
            "idf": self._idf,
            "total_length": self._total_length,
            "avgdl": self.avgdl,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25":
        """Reconstruct a BM25 index from a previously serialised dict."""
        instance = cls(k1=data["k1"], b=data["b"])
        instance._corpus = data["corpus"]
        instance._doc_freq = data["doc_freq"]
        instance._tf_cache = [Counter(d) for d in data["tf_cache"]]
        instance._idf = data["idf"]
        instance._total_length = data["total_length"]
        instance.avgdl = data["avgdl"]
        return instance

    @property
    def size(self) -> int:
        return len(self._corpus)
