"""Hybrid retriever: BM25 + semantic with reciprocal rank fusion."""

from __future__ import annotations

from agents.contracts import RagHit
from agents.rag.bm25_index import BM25Index
from agents.rag.semantic_index import SemanticIndex
from agents.rag.store import DocumentChunk, DocumentStore


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _score) in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        store: DocumentStore | None = None,
        *,
        embedding_base_url: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.store = store or DocumentStore()
        self.embedding_base_url = embedding_base_url
        self.embedding_model = embedding_model
        self.chunks: list[DocumentChunk] = self.store.load_chunks()
        self.bm25 = BM25Index(self.chunks)
        self.semantic = SemanticIndex(
            self.chunks,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
        )

    def refresh(self) -> None:
        self.chunks = self.store.load_chunks()
        self.bm25 = BM25Index(self.chunks)
        self.semantic = SemanticIndex(
            self.chunks,
            embedding_base_url=self.embedding_base_url,
            embedding_model=self.embedding_model,
        )

    def search(self, query: str, top_k: int = 6) -> list[RagHit]:
        if not self.chunks:
            return []
        bm25_hits = self.bm25.search(query, top_k=top_k * 2)
        sem_hits = self.semantic.search(query, top_k=top_k * 2)
        fused = reciprocal_rank_fusion([bm25_hits, sem_hits])[:top_k]
        results: list[RagHit] = []
        for idx, score in fused:
            chunk = self.chunks[idx]
            results.append(
                RagHit(
                    doc_id=chunk.chunk_id,
                    title=chunk.title,
                    snippet=chunk.text[:320],
                    source=chunk.source,
                    score=round(float(score), 6),
                    url=chunk.url,
                )
            )
        return results
