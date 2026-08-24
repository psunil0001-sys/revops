"""ChromaDB persistent store using the dedicated embed server (:8001)."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from agents.contracts import MemoryHit
from utils.config import CHROMA_DATA_DIR
from utils.embeddings import embed_via_server, sanitize_embed_text

RESEARCH_COLLECTION = "research_docs"
MEMORY_COLLECTION = "memories"


class _ServerEmbeddingFunction:
    """Chroma-compatible embedding function backed by llama-server :8001."""

    def __init__(
        self,
        *,
        embedding_base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.embedding_base_url = embedding_base_url
        self.model = model

    def name(self) -> str:
        return "llama-server-embed"

    def get_config(self) -> dict:
        return {
            "embedding_base_url": self.embedding_base_url or "",
            "model": self.model or "",
        }

    @staticmethod
    def build_from_config(config: dict) -> "_ServerEmbeddingFunction":
        return _ServerEmbeddingFunction(
            embedding_base_url=config.get("embedding_base_url") or None,
            model=config.get("model") or None,
        )

    def __call__(self, input: list[str]) -> list[list[float]]:
        return embed_via_server(
            input,
            embedding_base_url=self.embedding_base_url,
            model=self.model,
            batch_size=1,
        )

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


class ChromaMemoryStore:
    def __init__(
        self,
        *,
        persist_dir: str | None = None,
        embedding_base_url: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        self._lock = threading.RLock()
        self.embedding_base_url = embedding_base_url
        self.embedding_model = embedding_model
        self.persist_dir = str(persist_dir or CHROMA_DATA_DIR)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self._ef = _ServerEmbeddingFunction(
            embedding_base_url=embedding_base_url,
            model=embedding_model,
        )
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.research = self._client.get_or_create_collection(
            name=RESEARCH_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
        self.memories = self._client.get_or_create_collection(
            name=MEMORY_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )

    @staticmethod
    def _doc_id(*parts: str) -> str:
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
        return f"doc_{digest}"

    def upsert_documents(
        self,
        documents: list[dict[str, Any]],
        *,
        run_id: str,
        agent_id: str,
    ) -> list[str]:
        if not documents:
            return []
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for doc in documents:
            title = sanitize_embed_text(str(doc.get("title") or "untitled"), max_chars=200)
            text = sanitize_embed_text(str(doc.get("text") or ""), max_chars=1000)
            body = sanitize_embed_text(f"{title}. {text}", max_chars=1200)
            doc_id = self._doc_id(run_id, agent_id, title, body[:200])
            ids.append(doc_id)
            texts.append(body)
            metadatas.append(
                {
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "title": title[:200],
                    "source": str(doc.get("source") or agent_id)[:200],
                    "url": str(doc.get("url") or "")[:500],
                    "collection": RESEARCH_COLLECTION,
                }
            )
        # Upsert one-at-a-time so a single bad doc doesn't fail the whole agent batch
        written: list[str] = []
        with self._lock:
            for doc_id, body, meta in zip(ids, texts, metadatas):
                self.research.upsert(
                    ids=[doc_id],
                    documents=[body],
                    metadatas=[meta],
                )
                written.append(doc_id)
        return written

    def upsert_memory(
        self,
        *,
        run_id: str,
        session: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        doc_id = self._doc_id("memory", run_id, summary[:120])
        body = sanitize_embed_text(summary, max_chars=2000)
        meta = {
            "run_id": run_id,
            "session": session,
            "title": f"Run summary {run_id}",
            "collection": MEMORY_COLLECTION,
            **{k: str(v)[:200] for k, v in (metadata or {}).items()},
        }
        with self._lock:
            self.memories.upsert(
                ids=[doc_id],
                documents=[body],
                metadatas=[meta],
            )
        return doc_id

    def query_related(self, query: str, n: int = 8) -> list[MemoryHit]:
        if not query.strip():
            return []
        query = sanitize_embed_text(query, max_chars=800)
        per = max(1, (n + 1) // 2)
        hits: list[MemoryHit] = []
        with self._lock:
            for collection, name in (
                (self.research, RESEARCH_COLLECTION),
                (self.memories, MEMORY_COLLECTION),
            ):
                try:
                    count = collection.count()
                except Exception as exc:
                    raise RuntimeError(f"Chroma count failed ({name}): {exc}") from exc
                if count <= 0:
                    continue
                result = collection.query(
                    query_texts=[query],
                    n_results=min(per, count),
                    include=["documents", "metadatas", "distances"],
                )
                docs = (result.get("documents") or [[]])[0]
                metas = (result.get("metadatas") or [[]])[0]
                dists = (result.get("distances") or [[]])[0]
                ids = (result.get("ids") or [[]])[0]
                for i, text in enumerate(docs):
                    meta = metas[i] if i < len(metas) else {}
                    dist = dists[i] if i < len(dists) else None
                    score = None if dist is None else float(1.0 / (1.0 + float(dist)))
                    hits.append(
                        MemoryHit(
                            doc_id=ids[i] if i < len(ids) else f"{name}_{i}",
                            collection=name,
                            title=str((meta or {}).get("title") or name),
                            text=str(text or "")[:2000],
                            score=score,
                            metadata=dict(meta or {}),
                        )
                    )
        hits.sort(key=lambda h: h.score or 0.0, reverse=True)
        return hits[:n]
