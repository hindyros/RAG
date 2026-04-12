"""
Unit tests for the text chunker.

Uses synthetic ExtractedDocument objects — no real PDFs needed.
"""

import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_extractor import ExtractedDocument, TextBlock


def make_doc(blocks: list[tuple[str, int, bool]]) -> ExtractedDocument:
    """
    Helper: (text, page, is_header) → ExtractedDocument
    """
    return ExtractedDocument(
        source_file="test.pdf",
        blocks=[
            TextBlock(text=t, page_number=p, is_header=h, font_size=12.0 if not h else 18.0)
            for t, p, h in blocks
        ],
    )


class TestChunkerBasics:
    def test_empty_document_returns_no_chunks(self):
        doc = make_doc([])
        assert chunk_document(doc) == []

    def test_single_short_block_produces_one_chunk(self):
        doc = make_doc([("This is a short sentence.", 1, False)])
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert "short sentence" in chunks[0].text

    def test_chunk_carries_correct_metadata(self):
        doc = make_doc([("Sentence one. Sentence two.", 3, False)])
        chunks = chunk_document(doc)
        assert chunks[0].source_file == "test.pdf"
        assert chunks[0].page_number == 3

    def test_header_is_recorded_in_subsequent_chunks(self):
        doc = make_doc([
            ("Introduction", 1, True),
            ("This is the introduction text.", 1, False),
        ])
        chunks = chunk_document(doc)
        assert any(c.section_header == "Introduction" for c in chunks)

    def test_large_text_produces_multiple_chunks(self):
        # 600 tokens worth of text should produce at least 2 chunks with default size 512
        long_text = ("The quick brown fox jumps over the lazy dog. " * 50)
        doc = make_doc([(long_text, 1, False)])
        chunks = chunk_document(doc, chunk_size_tokens=50, overlap_tokens=10)
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self):
        long_text = "Sentence number " + ". Sentence number ".join(str(i) for i in range(100)) + "."
        doc = make_doc([(long_text, 1, False)])
        chunks = chunk_document(doc, chunk_size_tokens=30, overlap_tokens=5)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap_means_adjacent_chunks_share_content(self):
        # With overlap, the end of chunk N should appear at the start of chunk N+1
        sentences = " ".join([f"This is sentence {i}." for i in range(40)])
        doc = make_doc([(sentences, 1, False)])
        chunks = chunk_document(doc, chunk_size_tokens=20, overlap_tokens=8)
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")
        # The last sentence of chunk 0 should appear in chunk 1
        last_sentence_chunk0 = chunks[0].text.split(".")[-2].strip()
        assert last_sentence_chunk0 in chunks[1].text
