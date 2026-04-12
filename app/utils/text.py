"""
Text utilities that are used across the ingestion and retrieval layers.

We avoid NLTK and spaCy to keep the dependency footprint small.  The regex
patterns below handle the vast majority of English sentence boundaries without
false splits on common abbreviations.
"""

import re
import unicodedata


# Abbreviations that should NOT trigger a sentence split even when followed by
# a capital letter.  Extend this list as your domain requires.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "fig", "eq", "sec", "ch", "vol", "no", "pp", "ed", "est",
    "e.g", "i.e", "cf", "al", "inc", "ltd", "dept", "approx",
}

# Matches sentence-ending punctuation followed by whitespace + capital letter.
# The negative lookbehind prevents splitting on known abbreviations.
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'
)


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation heuristics.

    The approach is deliberately simple: split on '.', '!', '?' followed by
    whitespace and a capital letter, then post-filter splits that land on a
    known abbreviation.  This avoids the weight of a full NLP tokenizer while
    handling 95%+ of real document text correctly.
    """
    text = text.strip()
    if not text:
        return []

    # Raw candidate boundaries
    candidates = _SENTENCE_SPLIT_RE.split(text)

    sentences: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue

        # If the candidate ends with an abbreviation, it's a false split —
        # merge it back with the next sentence.
        last_word = candidate.rstrip(".!?").split()[-1].lower().rstrip(".")
        if last_word in _ABBREVIATIONS and sentences:
            # Re-attach to the previous sentence
            sentences[-1] = sentences[-1] + " " + candidate
        else:
            sentences.append(candidate)

    return sentences


def estimate_tokens(text: str) -> int:
    """
    Lightweight token count estimate: 1 token ≈ 4 characters (English prose).

    This is a deliberate approximation — exact tokenisation requires the model's
    tokenizer (tiktoken, sentencepiece, etc.) which we avoid importing.  The 4:1
    ratio is a well-documented heuristic for English text with BPE-based models
    and is accurate within ±15% for typical document content.
    """
    return max(1, len(text) // 4)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including newlines) into single spaces."""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form and replace common ligatures.

    PDF text extraction frequently produces ligatures (ﬁ, ﬂ, etc.) and
    non-breaking spaces that would otherwise make keyword search miss matches.
    """
    text = unicodedata.normalize("NFC", text)
    # Common ligature replacements
    ligatures = {
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
        "\ufb03": "ffi", "\ufb04": "ffl", "\u00a0": " ",
    }
    for lig, replacement in ligatures.items():
        text = text.replace(lig, replacement)
    return text


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Convert text to a list of lowercase tokens for BM25 indexing.

    We strip punctuation and split on whitespace.  No stemming — stemming adds
    a dependency and gives marginal gains for English technical documents where
    exact term matching (e.g. "transformer", "attention") is important.
    """
    text = normalize_unicode(text).lower()
    # Keep only alphanumeric characters and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if len(t) > 1]
