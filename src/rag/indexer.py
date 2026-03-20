"""
RAG Indexer — parses source files, embeds function chunks, and persists to ChromaDB.

Usage::

    from src.rag.indexer import Indexer

    indexer = Indexer()
    result = indexer.index_directory("./my_project/src")
    print(f"Indexed {result.chunks_indexed} chunks from {result.files_processed} files")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import chromadb
from langchain_ollama import OllamaEmbeddings

from src.config import TestIQConfig, load_config
from src.parser.base_parser import FunctionChunk
from src.parser.language_registry import EXTENSION_MAP, registry

logger = logging.getLogger(__name__)

COLLECTION_NAME = "testiq_functions"


@dataclass
class IndexResult:
    """Summary of an indexing run."""

    files_processed: int = 0
    chunks_indexed: int = 0
    errors: list[str] = field(default_factory=list)


class Indexer:
    """Parse source files, embed function chunks, and store in ChromaDB."""

    def __init__(
        self,
        config: TestIQConfig | None = None,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self._config = config or load_config()

        # Allow injecting a custom embedding function (used by tests)
        if embed_fn is not None:
            self._embed = embed_fn
        else:
            self._ollama = OllamaEmbeddings(
                model=self._config.embeddings.model,
                base_url=self._config.embeddings.base_url,
            )
            self._embed = self._ollama.embed_documents

        db_path = str(Path(self._config.rag.db_path).resolve())
        self._chroma = chromadb.PersistentClient(path=db_path)
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ───────────────────────────────────────────────────────

    def index_directory(self, directory: str) -> IndexResult:
        """
        Walk *directory*, parse all supported source files, embed the
        extracted function chunks, and store them in ChromaDB.

        Returns
        -------
        IndexResult
            Summary with counts and any errors encountered.
        """
        result = IndexResult()
        dir_path = Path(directory).resolve()

        if not dir_path.is_dir():
            result.errors.append(f"Not a directory: {directory}")
            return result

        supported_extensions = set(EXTENSION_MAP.keys())

        for file_path in sorted(dir_path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in supported_extensions:
                continue
            # Skip files inside hidden directories (e.g. .git, .testiq)
            if any(part.startswith(".") for part in file_path.relative_to(dir_path).parts):
                continue

            try:
                count = self.index_file(str(file_path))
                result.files_processed += 1
                result.chunks_indexed += count
            except Exception as exc:
                msg = f"Error indexing {file_path}: {exc}"
                logger.warning(msg)
                result.errors.append(msg)

        return result

    def index_file(self, filepath: str) -> int:
        """
        Parse and index a single source file.

        Returns
        -------
        int
            Number of chunks indexed from this file.
        """
        try:
            parser = registry.get_parser_for_file(filepath)
        except ValueError:
            logger.debug("Skipping unsupported file: %s", filepath)
            return 0

        chunks = parser.parse_file(filepath)
        if not chunks:
            return 0

        # Also extract file-level imports to enrich chunks
        try:
            file_imports = parser.extract_imports(filepath)
        except Exception:
            file_imports = []

        # Prepare batch for ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            chunk.imports = file_imports
            doc_id = f"{filepath}::{chunk.name}::{chunk.line_start}"
            ids.append(doc_id)
            documents.append(chunk.body)
            metadatas.append({
                "filename": str(Path(filepath).name),
                "function_name": chunk.name,
                "language": chunk.language,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "docstring": chunk.docstring or "",
            })

        # Generate embeddings
        embeddings = self._embed(documents)

        # Upsert into ChromaDB (handles re-indexing gracefully)
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("Indexed %d chunks from %s", len(chunks), filepath)
        return len(chunks)

    def clear_index(self) -> None:
        """Delete and re-create the ChromaDB collection."""
        self._chroma.delete_collection(COLLECTION_NAME)
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Index cleared.")

    @property
    def collection_count(self) -> int:
        """Return the total number of documents in the collection."""
        return self._collection.count()
