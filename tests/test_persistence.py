"""
Unit tests for vector store persistence.

These tests write and read actual files in a temporary directory — no mocking.
This is intentional: persistence bugs (e.g. partial writes, wrong dtype) only
manifest when real I/O is involved.
"""

import tempfile

import numpy as np
import pytest

from app.store.persistence import load_store, save_store


class TestPersistence:
    def test_roundtrip_embeddings(self):
        matrix = np.random.randn(10, 64).astype(np.float32)
        metadata = [{"source_file": f"doc{i}.pdf", "page_number": 1,
                     "chunk_index": i, "section_header": None, "text": f"chunk {i}"}
                    for i in range(10)]
        bm25_state = {"k1": 1.5, "b": 0.75, "corpus": [], "doc_freq": {},
                      "tf_cache": [], "idf": {}, "total_length": 0, "avgdl": 0.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_store(tmpdir, matrix, metadata, bm25_state)
            result = load_store(tmpdir)

        assert result is not None
        loaded_matrix, loaded_meta, loaded_bm25 = result
        np.testing.assert_array_equal(matrix, loaded_matrix)
        assert len(loaded_meta) == 10
        assert loaded_meta[3]["source_file"] == "doc3.pdf"

    def test_missing_store_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_store(tmpdir)
        assert result is None

    def test_save_creates_directory_if_missing(self):
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "nested", "store")
            save_store(
                subdir,
                np.zeros((2, 4), dtype=np.float32),
                [{"source_file": "f.pdf", "page_number": 1, "chunk_index": 0,
                  "section_header": None, "text": "x"}],
                {"k1": 1.5, "b": 0.75, "corpus": [], "doc_freq": {},
                 "tf_cache": [], "idf": {}, "total_length": 0, "avgdl": 0.0},
            )
            assert os.path.isdir(subdir)

    def test_metadata_unicode_preserved(self):
        """Unicode in chunk text and section headers must survive JSON serialisation."""
        matrix = np.zeros((1, 4), dtype=np.float32)
        metadata = [{
            "source_file": "日本語.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "section_header": "Résumé & Überblick",
            "text": "Héllo wörld — 你好世界",
        }]
        bm25_state = {"k1": 1.5, "b": 0.75, "corpus": [], "doc_freq": {},
                      "tf_cache": [], "idf": {}, "total_length": 0, "avgdl": 0.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_store(tmpdir, matrix, metadata, bm25_state)
            result = load_store(tmpdir)

        assert result is not None
        assert result[1][0]["section_header"] == "Résumé & Überblick"
        assert result[1][0]["text"] == "Héllo wörld — 你好世界"
