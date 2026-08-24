"""Simple JSONL document store with chunking."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_id: str
    title: str
    text: str
    source: str
    url: str | None
    created_at: str


class DocumentStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path("data/rag")
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs_path = self.root / "documents.jsonl"

    def _chunk_text(self, text: str, size: int = 500, overlap: int = 80) -> list[str]:
        words = re.findall(r"\S+", text)
        if not words:
            return []
        chunks: list[str] = []
        step = max(size - overlap, 1)
        for i in range(0, len(words), step):
            piece = " ".join(words[i : i + size]).strip()
            if piece:
                chunks.append(piece)
            if i + size >= len(words):
                break
        return chunks

    def add_document(
        self,
        *,
        title: str,
        text: str,
        source: str,
        url: str | None = None,
        doc_id: str | None = None,
    ) -> list[DocumentChunk]:
        body = (text or "").strip()
        if not body:
            return []
        digest = hashlib.sha1((url or title or body[:80]).encode("utf-8")).hexdigest()[:16]
        doc_id = doc_id or digest
        created = datetime.now(timezone.utc).isoformat()
        chunks: list[DocumentChunk] = []
        for idx, piece in enumerate(self._chunk_text(body)):
            chunk = DocumentChunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}:{idx}",
                title=title or "untitled",
                text=piece,
                source=source,
                url=url,
                created_at=created,
            )
            chunks.append(chunk)
        if not chunks:
            return []
        with self.docs_path.open("a", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=True) + "\n")
        return chunks

    def load_chunks(self) -> list[DocumentChunk]:
        if not self.docs_path.exists():
            return []
        chunks: list[DocumentChunk] = []
        with self.docs_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                chunks.append(DocumentChunk(**payload))
        return chunks

    def ingest_local_folder(self, folder: Path) -> int:
        folder = Path(folder)
        if not folder.exists():
            return 0
        count = 0
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md", ".csv"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            added = self.add_document(
                title=path.name,
                text=text,
                source=f"local:{path}",
                url=None,
                doc_id=hashlib.sha1(str(path).encode()).hexdigest()[:16],
            )
            count += len(added)
        return count
