"""
Semantic text chunker.

Chunking strategy — three passes:

Pass 1 — Structural split at section headers.
    We use PyMuPDF's font-size metadata (set by the extractor) to find
    section headers.  Each header starts a new chunk boundary.  This prevents
    a header from being buried mid-chunk where its semantic signal would be
    averaged away during embedding.

Pass 2 — Sentence-boundary split within sections.
    We split on sentence boundaries (see utils/text.py) and accumulate
    sentences until we reach the target token count.  We never cut mid-sentence
    — we overshoot to the next boundary instead.  This keeps each chunk
    semantically complete.

Pass 3 — Sliding window overlap.
    Adjacent chunks share `overlap_tokens` worth of text by prepending the
    tail of the previous chunk.  This preserves context that straddles a
    boundary, which is important for questions whose answer spans two chunks.

Parameter choices (512 tokens, 64-token overlap):
    - Mistral-embed accepts up to 2048 tokens but retrieval quality peaks at
      shorter, topically-focused chunks.  512 tokens is the empirical sweet
      spot across information retrieval benchmarks (BEIR, MTEB).
    - 64 tokens of overlap is ~12.5% of chunk size — enough for boundary
      continuity without excessive storage duplication.
"""

from dataclasses import dataclass, field

from app.ingestion.pdf_extractor import ExtractedDocument, TextBlock
from app.utils.text import estimate_tokens, normalize_whitespace, split_sentences


@dataclass
class Chunk:
    """One retrieval unit — the atom that gets embedded and searched."""
    text: str
    source_file: str
    page_number: int
    chunk_index: int           # sequential position in the document
    section_header: str | None # nearest preceding header, used in citations


def chunk_document(
    doc: ExtractedDocument,
    chunk_size_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """
    Convert an ExtractedDocument into retrieval-ready Chunks.

    Each returned Chunk has enough metadata to construct a citation without
    re-reading the source file.
    """
    chunks: list[Chunk] = []
    current_header: str | None = None

    # ── Pass 1: group blocks into sections ───────────────────────────────────
    # A section is a list of consecutive blocks under the same header.
    sections: list[tuple[str | None, int, list[TextBlock]]] = []
    current_section_blocks: list[TextBlock] = []
    current_section_page: int = 1

    for block in doc.blocks:
        if block.is_header:
            if current_section_blocks:
                sections.append(
                    (current_header, current_section_page, current_section_blocks)
                )
            current_header = normalize_whitespace(block.text)
            current_section_page = block.page_number
            current_section_blocks = []
        else:
            if not current_section_blocks:
                current_section_page = block.page_number
            current_section_blocks.append(block)

    if current_section_blocks:
        sections.append((current_header, current_section_page, current_section_blocks))

    # ── Passes 2 & 3: sentence split + overlap within each section ───────────
    chunk_index = 0
    overlap_sentences: list[str] = []  # tail sentences from the previous chunk

    for header, section_page, blocks in sections:
        # Flatten blocks into sentences, tracking which page each sentence is on
        all_sentences: list[tuple[str, int]] = []
        for block in blocks:
            block_text = normalize_whitespace(block.text)
            for sentence in split_sentences(block_text):
                all_sentences.append((sentence, block.page_number))

        if not all_sentences:
            continue

        # Accumulate sentences into chunks respecting the token budget
        current_sentences: list[str] = list(overlap_sentences)  # seed with overlap
        current_page = section_page
        current_tokens = sum(estimate_tokens(s) for s in current_sentences)

        for sentence, page in all_sentences:
            sentence_tokens = estimate_tokens(sentence)

            if current_tokens + sentence_tokens > chunk_size_tokens and current_sentences:
                # Flush the current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source_file=doc.source_file,
                        page_number=current_page,
                        chunk_index=chunk_index,
                        section_header=header,
                    )
                )
                chunk_index += 1

                # Compute overlap: take enough tail sentences to fill overlap_tokens
                overlap_sentences = _compute_overlap(current_sentences, overlap_tokens)
                current_sentences = list(overlap_sentences) + [sentence]
                current_tokens = sum(estimate_tokens(s) for s in current_sentences)
                current_page = page
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Flush any remaining sentences as the final chunk of this section
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_file=doc.source_file,
                    page_number=current_page,
                    chunk_index=chunk_index,
                    section_header=header,
                )
            )
            chunk_index += 1
            overlap_sentences = _compute_overlap(current_sentences, overlap_tokens)

    return chunks


def _compute_overlap(sentences: list[str], overlap_tokens: int) -> list[str]:
    """
    Return the tail of `sentences` whose total token count ≤ overlap_tokens.

    We walk from the end backwards, accumulating sentences until we would
    exceed the overlap budget, then return what fits.
    """
    result: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        t = estimate_tokens(sentence)
        if total + t > overlap_tokens:
            break
        result.insert(0, sentence)
        total += t
    return result
