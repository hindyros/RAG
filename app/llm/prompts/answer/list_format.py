"""
Answer template for list / enumeration queries.

Examples of queries routed here:
  "List all safety requirements mentioned in the document."
  "What are the steps in the onboarding process?"
  "Give me all the models compared in the paper."

Shape: numbered list, one item per line, each item ends with [N] citation.
No prose paragraphs.
"""


def build_prompt(
    query: str,
    chunks: list[dict],
) -> list[dict[str, str]]:
    system = (
        "You are a precise research assistant.  The user wants a list of items.  "
        "Extract and enumerate every relevant item from the provided source passages.  "
        "Format your answer as a numbered list.  "
        "Each list item must end with a citation [N] that corresponds to the passage index.  "
        "Do not write introductory sentences or concluding paragraphs — only the list.  "
        "If the passages contain no relevant items, state that explicitly."
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
