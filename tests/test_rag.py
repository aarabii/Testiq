"""
Tests for the RAG indexer and retriever.

All tests use a fake embedding function — no Ollama server required.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest

from src.config import TestIQConfig, RAGConfig
from src.parser.base_parser import FunctionChunk


# ── Fake embedding functions ─────────────────────────────────────────────────

EMBED_DIM = 64


def _deterministic_vector(text: str) -> list[float]:
    """Generate a deterministic embedding from a string's MD5 hash."""
    digest = hashlib.md5(text.encode()).hexdigest()
    # Convert hex chars to floats in [0, 1]
    vec = [int(c, 16) / 15.0 for c in digest]
    # Pad or trim to EMBED_DIM
    vec = (vec * ((EMBED_DIM // len(vec)) + 1))[:EMBED_DIM]
    return vec


def fake_embed_documents(texts: list[str]) -> list[list[float]]:
    """Batch embedding function for the Indexer."""
    return [_deterministic_vector(t) for t in texts]


def fake_embed_query(text: str) -> list[float]:
    """Single-query embedding function for the Retriever."""
    return _deterministic_vector(text)


# ── Helpers ──────────────────────────────────────────────────────────────────

SAMPLE_PY_A = textwrap.dedent('''\
    import os

    def hello(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"

    def goodbye(name: str) -> str:
        """Say goodbye."""
        return f"Goodbye, {name}!"
''')

SAMPLE_PY_B = textwrap.dedent('''\
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
''')


def _make_config(tmp_path: Path) -> TestIQConfig:
    """Create a config pointing ChromaDB at a temp directory."""
    db_path = str(tmp_path / "chroma_test_db")
    return TestIQConfig(rag=RAGConfig(db_path=db_path, top_k=3))


# ── Indexer tests ────────────────────────────────────────────────────────────

class TestIndexer:
    def test_index_directory(self, tmp_path: Path):
        from src.rag.indexer import Indexer

        # Set up sample source files
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "greetings.py").write_text(SAMPLE_PY_A, encoding="utf-8")
        (src_dir / "math_utils.py").write_text(SAMPLE_PY_B, encoding="utf-8")

        config = _make_config(tmp_path)
        indexer = Indexer(config, embed_fn=fake_embed_documents)

        result = indexer.index_directory(str(src_dir))

        assert result.files_processed == 2
        assert result.chunks_indexed >= 3  # hello, goodbye, add
        assert result.errors == []
        assert indexer.collection_count >= 3

    def test_index_file(self, tmp_path: Path):
        from src.rag.indexer import Indexer

        py_file = tmp_path / "sample.py"
        py_file.write_text(SAMPLE_PY_A, encoding="utf-8")

        config = _make_config(tmp_path)
        indexer = Indexer(config, embed_fn=fake_embed_documents)

        count = indexer.index_file(str(py_file))
        assert count == 2  # hello + goodbye
        assert indexer.collection_count == 2

    def test_metadata_stored(self, tmp_path: Path):
        """Verify that metadata is correctly stored in ChromaDB."""
        from src.rag.indexer import Indexer

        py_file = tmp_path / "sample.py"
        py_file.write_text(SAMPLE_PY_A, encoding="utf-8")

        config = _make_config(tmp_path)
        indexer = Indexer(config, embed_fn=fake_embed_documents)
        indexer.index_file(str(py_file))

        # Peek inside the collection
        all_docs = indexer._collection.get(include=["metadatas"])
        meta_list = all_docs["metadatas"]

        func_names = [m["function_name"] for m in meta_list]
        assert "hello" in func_names
        assert "goodbye" in func_names

        hello_meta = next(m for m in meta_list if m["function_name"] == "hello")
        assert hello_meta["language"] == "python"
        assert hello_meta["filename"] == "sample.py"
        assert hello_meta["docstring"] == "Say hello."
        assert hello_meta["line_start"] > 0

    def test_clear_index(self, tmp_path: Path):
        from src.rag.indexer import Indexer

        py_file = tmp_path / "sample.py"
        py_file.write_text(SAMPLE_PY_A, encoding="utf-8")

        config = _make_config(tmp_path)
        indexer = Indexer(config, embed_fn=fake_embed_documents)
        indexer.index_file(str(py_file))
        assert indexer.collection_count > 0

        indexer.clear_index()
        assert indexer.collection_count == 0

    def test_skips_unsupported_files(self, tmp_path: Path):
        from src.rag.indexer import Indexer

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "readme.txt").write_text("Not code", encoding="utf-8")
        (src_dir / "data.csv").write_text("a,b,c", encoding="utf-8")
        (src_dir / "greetings.py").write_text(SAMPLE_PY_A, encoding="utf-8")

        config = _make_config(tmp_path)
        indexer = Indexer(config, embed_fn=fake_embed_documents)
        result = indexer.index_directory(str(src_dir))

        # Only the .py file should be processed
        assert result.files_processed == 1
        assert result.errors == []

    def test_reindex_upserts(self, tmp_path: Path):
        """Re-indexing the same file should not duplicate documents."""
        from src.rag.indexer import Indexer

        py_file = tmp_path / "sample.py"
        py_file.write_text(SAMPLE_PY_A, encoding="utf-8")

        config = _make_config(tmp_path)
        indexer = Indexer(config, embed_fn=fake_embed_documents)
        indexer.index_file(str(py_file))
        first_count = indexer.collection_count

        # Index the same file again
        indexer.index_file(str(py_file))
        assert indexer.collection_count == first_count


# ── Retriever tests ──────────────────────────────────────────────────────────

class TestRetriever:
    def _index_sample(self, tmp_path: Path):
        """Helper: set up an indexed collection and return the config."""
        from src.rag.indexer import Indexer

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "greetings.py").write_text(SAMPLE_PY_A, encoding="utf-8")
        (src_dir / "math_utils.py").write_text(SAMPLE_PY_B, encoding="utf-8")

        config = _make_config(tmp_path)
        indexer = Indexer(config, embed_fn=fake_embed_documents)
        indexer.index_directory(str(src_dir))
        return config

    def test_query(self, tmp_path: Path):
        from src.rag.retriever import Retriever

        config = self._index_sample(tmp_path)
        retriever = Retriever(config, embed_fn=fake_embed_query)

        results = retriever.query("say hello to someone", top_k=2)

        assert len(results) == 2
        assert all(r.content for r in results)
        assert all(r.metadata for r in results)
        assert all(isinstance(r.distance, float) for r in results)

    def test_query_returns_metadata(self, tmp_path: Path):
        from src.rag.retriever import Retriever

        config = self._index_sample(tmp_path)
        retriever = Retriever(config, embed_fn=fake_embed_query)

        results = retriever.query("add numbers together", top_k=3)

        # All results should have the expected metadata keys
        for r in results:
            assert "function_name" in r.metadata
            assert "language" in r.metadata
            assert "filename" in r.metadata
            assert "line_start" in r.metadata

    def test_query_chunk(self, tmp_path: Path):
        from src.rag.retriever import Retriever

        config = self._index_sample(tmp_path)
        retriever = Retriever(config, embed_fn=fake_embed_query)

        chunk = FunctionChunk(
            name="test_fn",
            body="def test_fn():\n    return 42",
        )
        results = retriever.query_chunk(chunk, top_k=2)
        assert len(results) == 2

    def test_empty_collection(self, tmp_path: Path):
        from src.rag.retriever import Retriever

        config = _make_config(tmp_path)
        retriever = Retriever(config, embed_fn=fake_embed_query)

        results = retriever.query("anything", top_k=5)
        assert results == []

    def test_top_k_from_config(self, tmp_path: Path):
        """When top_k is not passed, it defaults to config.rag.top_k (3)."""
        from src.rag.retriever import Retriever

        config = self._index_sample(tmp_path)
        retriever = Retriever(config, embed_fn=fake_embed_query)

        results = retriever.query("hello")
        assert len(results) == config.rag.top_k  # 3
