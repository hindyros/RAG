"""
Answer template for comparison queries.

Examples of queries routed here:
  "Compare GPT-4 and Claude in terms of context window."
  "What are the differences between BM25 and dense retrieval?"
  "Pros and cons of microservices vs monoliths."

Shape: structured prose with three sections — Overview, Key Differences
(bullet points), Summary — plus [N] citations throughout.
"""


def build_prompt(
    query: str,
    chunks: list[dict],
) -> list[dict[str, str]]:
    system = (
        "You are a precise research assistant.  The user wants a comparison.  "
        "Structure your answer in three sections:\n\n"
        "**Overview**: One sentence introducing what is being compared.\n"
        "**Key Differences**: A bullet-point list where each bullet describes "
        "one difference.  Each bullet must cite its source with [N].\n"
        "**Summary**: One sentence summarising the most important distinction.\n\n"
        "Use ONLY information from the provided source passages.  "
        "Every factual claim must have an inline [N] citation.  "
        "If the passages lack enough information for a full comparison, say so."
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
