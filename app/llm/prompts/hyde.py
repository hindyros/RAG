"""
Prompt for HyDE (Hypothetical Document Embeddings) query transformation.

The Problem HyDE Solves
-----------------------
There is a distributional mismatch between short user queries and longer
document chunks.  A query like "What are transformer attention mechanisms?"
has a very different embedding trajectory from a paragraph in a textbook that
answers that question, even though they convey related information.

The HyDE Approach (Gao et al., 2022 — arxiv.org/abs/2212.10496):
1. Ask the LLM to generate a *hypothetical* document that would answer the query.
2. Embed that hypothetical document (not the original query).
3. Use the hypothetical document's embedding as the search vector.

Because the hypothetical document is written in the same style and length as
real documents, its embedding lands closer in the embedding space to actual
relevant chunks — improving recall especially for complex or abstract queries.

Trade-offs:
- One extra LLM call per query (latency cost, ~0.5-1s with Mistral).
- BM25 still uses the original query tokens (HyDE is embedding-specific).
- Works best for knowledge-dense queries; adds noise for simple lookups.

We apply HyDE unconditionally for knowledge_seeking queries.  For a future
optimisation, one could skip it for short exact-match queries.
"""

HYDE_SYSTEM_PROMPT = """\
You are a document synthesis assistant.  Given a question, write a single
paragraph (100-200 words) that would appear in a high-quality technical
document as the answer to that question.

Write as if you are the author of the document — authoritative, specific, and
detailed.  Do not mention the question itself.  Do not say "I" or "you".
Do not add disclaimers.  Just write the hypothetical document passage.
"""


def build_hyde_messages(query: str) -> list[dict[str, str]]:
    """Return the messages list for HyDE document generation."""
    return [
        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
