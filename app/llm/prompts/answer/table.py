"""
Answer template for table queries.

Examples of queries routed here:
  "Give me a table of all models and their parameter counts."
  "Summarise the comparison between providers in a table."
  "Show me the feature matrix for each product."

Shape: a GitHub-flavoured Markdown table.  The model is instructed to
determine appropriate columns from the query and source content.  A "Source"
column carries the [N] citation for each row.
"""


def build_prompt(
    query: str,
    chunks: list[dict],
) -> list[dict[str, str]]:
    system = (
        "You are a precise research assistant.  The user wants a Markdown table.  "
        "Determine the appropriate columns from the user's question and the source passages.  "
        "Include a 'Source' column as the last column, listing the [N] passage indices "
        "that support each row's data.  "
        "Use ONLY information found in the source passages — do not invent data.  "
        "If a cell has no data in the passages, write 'N/A'.  "
        "Output ONLY the Markdown table, no prose before or after."
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
