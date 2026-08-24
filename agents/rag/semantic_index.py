"""Semantic index using dedicated embedding server (:8001), not chat LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from utils.embeddings import embed_via_server, sanitize_embed_text

if TYPE_CHECKING:
    from agents.rag.store import DocumentChunk


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def embed_texts(
    texts: list[str],
    *,
    embedding_base_url: str | None = None,
    model: str | None = None,
    batch_size: int = 1,
    timeout: float = 120.0,
) -> np.ndarray:
    """Create embeddings via OpenAI-compatible POST /embeddings on the embed server."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    cleaned = [sanitize_embed_text(t) for t in texts]
    vectors = embed_via_server(
        cleaned,
        embedding_base_url=embedding_base_url,
        model=model,
        batch_size=batch_size,
        timeout=timeout,
    )
    mat = np.asarray(vectors, dtype=np.float32)
    return _l2_normalize(mat)


class SemanticIndex:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        *,
        embedding_base_url: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.chunks = chunks
        self.embedding_base_url = embedding_base_url
        self.embedding_model = embedding_model
        self._texts = [sanitize_embed_text(f"{c.title}. {c.text}") for c in chunks]
        self.embeddings: np.ndarray | None = None
        self.backend = "empty" if not self._texts else "deferred"

    def _ensure_embeddings(self) -> np.ndarray:
        if self.embeddings is not None:
            return self.embeddings
        if not self._texts:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
            self.backend = "empty"
            return self.embeddings
        self.embeddings = embed_texts(
            self._texts,
            embedding_base_url=self.embedding_base_url,
            model=self.embedding_model,
        )
        self.backend = "embed_server"
        return self.embeddings

    def search(self, query: str, top_k: int = 8) -> list[tuple[int, float]]:
        matrix = self._ensure_embeddings()
        if matrix.size == 0:
            return []
        q = embed_texts(
            [query],
            embedding_base_url=self.embedding_base_url,
            model=self.embedding_model,
        )[0]
        scores = matrix @ q
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k]]
