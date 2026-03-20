"""
RAG Retriever — queries ChromaDB for semantically similar function chunks.

Usage::

    from src.rag.retriever import Retriever

    retriever = Retriever()
    results = retriever.query("validate user token")
    for r in results:
        print(r.metadata["function_name"], r.distance)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chromadb
from langchain_ollama import OllamaEmbeddings

from src.config import TestIQConfig, load_config
from src.parser.base_parser import FunctionChunk
from src.rag.indexer import COLLECTION_NAME

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result from ChromaDB."""

    content: str
    metadata: dict
    distance: float


class Retriever:
    """Query ChromaDB for the most relevant function chunks."""

    def __init__(
        self,
        config: TestIQConfig | None = None,
        *,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._config = config or load_config()

        # Allow injecting a custom embedding function (used by tests)
        if embed_fn is not None:
            self._embed_query = embed_fn
        else:
            self._ollama = OllamaEmbeddings(
                model=self._config.embeddings.model,
                base_url=self._config.embeddings.base_url,
            )
            self._embed_query = self._ollama.embed_query

        db_path = str(Path(self._config.rag.db_path).resolve())
        self._chroma = chromadb.PersistentClient(path=db_path)

        try:
            self._collection = self._chroma.get_collection(name=COLLECTION_NAME)
        except Exception:
            # Collection doesn't exist yet — create it so queries return empty
            self._collection = self._chroma.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

    # ── Public API ───────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Embed *text* and return the *top_k* most similar chunks.

        Parameters
        ----------
        text : str
            Free-form query text (e.g. a function body, a description).
        top_k : int | None
            Number of results to return.  Falls back to ``config.rag.top_k``.

        Returns
        -------
        list[RetrievalResult]
            Ordered by ascending distance (most similar first).
        """
        k = top_k or self._config.rag.top_k

        if self._collection.count() == 0:
            return []

        # Clamp k to collection size
        k = min(k, self._collection.count())

        query_embedding = self._embed_query(text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Unpack ChromaDB's nested list structure
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            RetrievalResult(
                content=doc,
                metadata=meta,
                distance=dist,
            )
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def query_chunk(
        self,
        chunk: FunctionChunk,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Convenience wrapper: use a FunctionChunk's body as the query text.

        Parameters
        ----------
        chunk : FunctionChunk
            The chunk whose body will be embedded and used as the query.
        top_k : int | None
            Number of results to return.
        """
        return self.query(chunk.body, top_k=top_k)
