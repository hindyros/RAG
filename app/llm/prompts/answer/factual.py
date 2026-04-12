"""
Answer template for factual / definitional queries.

Examples of queries routed here:
  "What is RAG?"
  "Who invented the transformer architecture?"
  "When was this regulation enacted?"

Shape: 2-4 concise sentences, inline citations [N], grounded in context only.
"""


def build_prompt(
    query: str,
    chunks: list[dict],  # {"text": str, "citation_index": int, "source_file": str}
) -> list[dict[str, str]]:
    system = (
        "You are a precise research assistant.  Answer the question using ONLY "
        "the provided source passages.  Be concise (2-4 sentences).  "
        "Cite every factual claim with an inline reference like [1] or [2] "
        "that corresponds to the passage index.  "
        "If the passages do not contain enough information to answer, say so explicitly."
    )

    context_block = _format_context(chunks)
    user = f"Question: {query}\n\nSource passages:\n{context_block}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for chunk in chunks:
        lines.append(f"[{chunk['citation_index']}] ({chunk['source_file']}):\n{chunk['text']}")
    return "\n\n".join(lines)
