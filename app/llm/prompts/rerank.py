"""
Prompt for LLM-based reranking.

Why rerank after hybrid retrieval?
    RRF fusion already combines two ranking signals, but neither signal
    directly answers "is this chunk useful for answering the specific question?"
    The embedding measures topical proximity; BM25 measures keyword overlap.
    Neither measures *answer quality*.

    A reranking step asks the LLM to score each candidate chunk on its actual
    utility for answering the query.  This is equivalent to a cross-encoder
    in a bi-encoder + cross-encoder retrieval pipeline, but we use the same
    Mistral model instead of a specialised cross-encoder to avoid an extra
    dependency.

Approach — pointwise scoring:
    We score each chunk independently rather than doing listwise comparison.
    Pointwise is simpler, more predictable, and allows early termination.
    The score is 0-10 (integers, which the model produces more reliably than
    floats when asked for JSON output).

    For top_k_rerank=5 and top_k_retrieval=10, this is 10 sequential LLM calls
    — expensive.  A practical optimisation (not implemented here for clarity)
    is to batch all chunks into one prompt with a listwise ranking request,
    which reduces latency at the cost of a more complex prompt.
"""


def build_rerank_messages(query: str, chunk_text: str) -> list[dict[str, str]]:
    """
    Build the messages to score one chunk's relevance to a query.

    Returns a JSON string: {"score": <int 0-10>, "reasoning": "<one sentence>"}
    """
    system = (
        "You are a relevance judge for a retrieval-augmented QA system.  "
        "Given a user question and a document passage, rate how useful the "
        "passage is for answering the question.\n\n"
        "Return ONLY a JSON object:\n"
        '{"score": <integer 0-10>, "reasoning": "<one sentence>"}\n\n'
        "Score guide:\n"
        "  10 — passage directly and completely answers the question\n"
        "  7-9 — passage is highly relevant and provides key information\n"
        "  4-6 — passage is partially relevant or tangentially related\n"
        "  1-3 — passage is barely related\n"
        "  0   — passage is completely irrelevant\n"
        "Output ONLY the JSON object. No markdown, no extra text."
    )
    user = f"Question: {query}\n\nPassage:\n{chunk_text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
