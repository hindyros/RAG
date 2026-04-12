"""
PDF text extraction using PyMuPDF (fitz).

Why PyMuPDF over pdfplumber or pdfminer:
- 5-10x faster on large documents
- Exposes span-level font metadata (size, bold, italic flags) which lets the
  chunker detect section headers by font size rather than regex heuristics
- Handles multi-column layouts and rotated text more reliably
- Lower memory footprint — important when ingesting many files in one request

The extractor outputs a structured list of blocks where each block knows its
page number, approximate font size, and whether it looks like a header.  The
chunker then uses this structure instead of treating the document as a flat
string.
"""

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class TextBlock:
    """One coherent block of text from a PDF page."""
    text: str
    page_number: int          # 1-indexed for human-readable citations
    is_header: bool = False   # True when font size suggests a section heading
    font_size: float = 0.0
    bbox: tuple[float, float, float, float] = field(default_factory=tuple)  # type: ignore[assignment]


@dataclass
class ExtractedDocument:
    """All text blocks extracted from a single PDF file."""
    source_file: str          # original filename, used in citations
    blocks: list[TextBlock]

    @property
    def full_text(self) -> str:
        """Flat concatenation of all blocks — useful for debugging."""
        return "\n".join(b.text for b in self.blocks)


def extract_pdf(file_bytes: bytes, filename: str) -> ExtractedDocument:
    """
    Extract structured text from a PDF file given its raw bytes.

    Strategy:
    1. Use page.get_text("dict") to get span-level font metadata.
    2. Collect text spans grouped by their block bounding box.
    3. Detect headers by comparing each block's dominant font size to the
       body font size (median across the whole document).
    4. Filter out noise: blocks that are too short, appear to be page numbers,
       or contain only whitespace.

    Parameters
    ----------
    file_bytes : raw PDF content (already read from the upload)
    filename   : original filename, stored in chunk metadata for citations
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    raw_blocks: list[TextBlock] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in page_dict.get("blocks", []):
            # Blocks of type 1 are images — skip them
            if block.get("type") != 0:
                continue

            block_text_parts: list[str] = []
            font_sizes: list[float] = []

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        block_text_parts.append(span_text)
                        font_sizes.append(span.get("size", 0.0))

            block_text = " ".join(block_text_parts).strip()
            if not block_text or len(block_text) < 10:
                # Skip near-empty blocks (page numbers, single characters, etc.)
                continue

            dominant_font_size = (
                sum(font_sizes) / len(font_sizes) if font_sizes else 0.0
            )
            bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))

            raw_blocks.append(
                TextBlock(
                    text=block_text,
                    page_number=page_number,
                    font_size=dominant_font_size,
                    bbox=bbox,  # type: ignore[arg-type]
                )
            )

    doc.close()

    # ── Header detection ─────────────────────────────────────────────────────
    # Compute the body font size as the median of all block font sizes.
    # Any block whose font size exceeds body_size * 1.2 is treated as a header.
    if raw_blocks:
        sorted_sizes = sorted(b.font_size for b in raw_blocks if b.font_size > 0)
        body_font_size = sorted_sizes[len(sorted_sizes) // 2] if sorted_sizes else 0.0
        header_threshold = body_font_size * 1.2

        for block in raw_blocks:
            if block.font_size >= header_threshold and len(block.text) < 200:
                block.is_header = True

    return ExtractedDocument(source_file=filename, blocks=raw_blocks)
