"""
Prompt for intent detection.

We ask the model to classify a query on two axes:
1. primary: is this knowledge-seeking (needs RAG) or conversational (direct reply)?
2. sub: if knowledge-seeking, what answer shape does the user expect?

Why structured JSON output?
    The intent detector result drives downstream branching (retrieval vs. skip,
    which answer template to load).  JSON with a fixed schema is safer than
    free-text classification because we can validate the output and fall back
    gracefully on schema violations.

Few-shot examples are embedded in the system prompt rather than provided as
message history to keep the context window efficient — they don't grow with
conversation length.
"""

INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for a document question-answering system.
Your task: classify the user's query and return a JSON object with this schema:

{
  "primary": "conversational" | "knowledge_seeking",
  "sub": "factual" | "list" | "comparison" | "table" | "chitchat",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining your classification>"
}

Definitions:
- conversational: greetings, thanks, small talk, clarifications about the system
  itself — these do NOT need a knowledge base search.
- knowledge_seeking: any question that may be answered by information in documents.

Sub-intent guide (only relevant when primary = knowledge_seeking):
- factual   : seeks a specific fact, definition, or explanation ("What is X?")
- list      : seeks an enumeration of items ("List all X", "What are the types of Y?")
- comparison: seeks a contrast between two or more things ("Compare X and Y")
- table     : explicitly asks for tabular output ("Give me a table of X by Y")
- chitchat  : use this when primary = conversational

Examples (do not repeat these in your answer — they are for calibration only):

Q: "hello"
A: {"primary":"conversational","sub":"chitchat","confidence":0.99,"reasoning":"Greeting with no information need."}

Q: "What is retrieval-augmented generation?"
A: {"primary":"knowledge_seeking","sub":"factual","confidence":0.97,"reasoning":"Asks for a definition of a specific concept."}

Q: "List all the safety guidelines mentioned in the document."
A: {"primary":"knowledge_seeking","sub":"list","confidence":0.95,"reasoning":"Explicit enumeration request."}

Q: "How does GPT-4 compare to Claude in terms of context window?"
A: {"primary":"knowledge_seeking","sub":"comparison","confidence":0.93,"reasoning":"Asks for a direct comparison of two models on a specific dimension."}

Q: "Can you give me a table of all models and their parameter counts?"
A: {"primary":"knowledge_seeking","sub":"table","confidence":0.96,"reasoning":"Explicit request for tabular output."}

Q: "Thank you, that was helpful!"
A: {"primary":"conversational","sub":"chitchat","confidence":0.98,"reasoning":"Expression of gratitude, no question asked."}

Output ONLY the JSON object. No markdown fences, no extra text.
"""


def build_intent_messages(query: str) -> list[dict[str, str]]:
    """Return the messages list to send to the chat API for intent detection."""
    return [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
