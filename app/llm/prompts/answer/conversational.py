"""
Prompt for conversational / chitchat responses.

When the intent detector classifies a query as conversational, we skip
retrieval entirely and use this lightweight prompt.  The model responds
as a helpful assistant without fabricating knowledge-base content.
"""


def build_prompt(query: str) -> list[dict[str, str]]:
    system = (
        "You are a helpful assistant for a document Q&A system.  "
        "The user is having a conversation rather than asking a research question.  "
        "Respond naturally and helpfully.  "
        "If the user seems to be asking about the system's capabilities, explain "
        "that you can answer questions about uploaded PDF documents."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
