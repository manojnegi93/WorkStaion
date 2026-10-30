"""Hybrid (dense + sparse/BM25) retrieval on Qdrant.

Qdrant (Apache-2.0) stores two named vectors per chunk:
  * ``dense``  - embedding from the configured open-source model (nomic-embed-text, ...)
  * ``sparse`` - BM25-style term weights; Qdrant applies IDF server-side (Modifier.IDF)

At query time both retrievers run in one batch request and results are merged
with weighted Reciprocal Rank Fusion. Fusion is done client-side so that every
hit keeps its per-retriever rank, which the UI trace and the eval report show.

Works with a Qdrant server (QDRANT_URL) or fully embedded (on-disk / in-memory).
"""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.config import Settings, get_settings
from app.ingestion.chunker import Chunk
from app.textutils import content_tokens

log = logging.getLogger(__name__)

SearchMode = Literal["hybrid", "dense", "sparse"]
_K1 = 1.2
_RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    section: str
    title: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "section": self.section,
            "title": self.title,
            "score": round(self.score, 5),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
        }


def _term_index(term: str) -> int:
    return int.from_bytes(hashlib.md5(term.encode()).digest()[:4], "little") & 0x7FFFFFFF


def sparse_encode(text: str, is_query: bool = False) -> qm.SparseVector:
    """BM25 term-frequency component (IDF is applied by Qdrant)."""
    counts = Counter(content_tokens(text))
    if not counts:
        return qm.SparseVector(indices=[], values=[])
    merged: dict[int, float] = {}
    for term, tf in counts.items():
        weight = 1.0 if is_query else tf * (_K1 + 1) / (tf + _K1)
        idx = _term_index(term)
        merged[idx] = merged.get(idx, 0.0) + weight
    return qm.SparseVector(indices=list(merged), values=list(merged.values()))


class HybridVectorStore:
    def __init__(self, embeddings: Embeddings, settings: Settings | None = None, client: QdrantClient | None = None):
        self.s = settings or get_settings()
        self.embeddings = embeddings
        self.collection = self.s.collection_name
        self._lock = threading.Lock()
        if client is not None:
            self.client = client
        elif self.s.qdrant_url:
            self.client = QdrantClient(url=self.s.qdrant_url)
        elif self.s.qdrant_path == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=self.s.qdrant_path)
        self._dim: int | None = None

    # ------------------------------------------------------------------ schema
    def ensure_collection(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            existing = vectors["dense"].size if isinstance(vectors, dict) else None
            if existing == dim:
                self._dim = dim
                return
            log.warning("Embedding dim changed (%s -> %s); recreating collection", existing, dim)
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={"dense": qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        if self.s.qdrant_url:  # payload indexes are a no-op in embedded mode
            self.client.create_payload_index(self.collection, "source", qm.PayloadSchemaType.KEYWORD)
        self._dim = dim

    # ------------------------------------------------------------------ writes
    def upsert(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        if not chunks:
            return 0
        with self._lock:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                vectors = self.embeddings.embed_documents([c.indexed_text for c in batch])
                if self._dim is None:
                    self.ensure_collection(len(vectors[0]))
                points = [
                    qm.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                        vector={"dense": v, "sparse": sparse_encode(c.indexed_text)},
                        payload=c.to_payload(),
                    )
                    for c, v in zip(batch, vectors, strict=True)
                ]
                self.client.upsert(self.collection, points=points)
        return len(chunks)

    def delete_source(self, source: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source))])
            ),
        )

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._dim = None

    # ------------------------------------------------------------------ reads
    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def list_sources(self) -> dict[str, int]:
        if not self.client.collection_exists(self.collection):
            return {}
        counts: Counter[str] = Counter()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=["source"], with_vectors=False
            )
            counts.update(p.payload.get("source", "?") for p in points)
            if offset is None:
                break
        return dict(sorted(counts.items()))

    def search(
        self,
        query: str,
        k: int | None = None,
        mode: SearchMode = "hybrid",
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        source_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        k = k or self.s.top_k
        if not self.client.collection_exists(self.collection):
            return []
        flt = (
            qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchAny(any=source_filter))])
            if source_filter
            else None
        )
        n = max(self.s.candidate_k, k)
        requests: list[tuple[str, qm.QueryRequest]] = []
        if mode in ("hybrid", "dense"):
            requests.append(
                (
                    "dense",
                    qm.QueryRequest(
                        query=self.embeddings.embed_query(query), using="dense", limit=n, filter=flt, with_payload=True
                    ),
                )
            )
        if mode in ("hybrid", "sparse"):
            sv = sparse_encode(query, is_query=True)
            if sv.indices:
                requests.append(
                    ("sparse", qm.QueryRequest(query=sv, using="sparse", limit=n, filter=flt, with_payload=True))
                )
        if not requests:
            return []

        responses = self.client.query_batch_points(self.collection, requests=[r for _, r in requests])
        weights = {"dense": dense_weight, "sparse": sparse_weight}
        fused: dict[str, RetrievedChunk] = {}
        for (name, _), resp in zip(requests, responses, strict=True):
            for rank, point in enumerate(resp.points, start=1):
                p = point.payload or {}
                hit = fused.get(p["chunk_id"])
                if hit is None:
                    hit = RetrievedChunk(
                        chunk_id=p["chunk_id"],
                        text=p["text"],
                        source=p["source"],
                        section=p.get("section", ""),
                        title=p.get("title", ""),
                        score=0.0,
                    )
                    fused[p["chunk_id"]] = hit
                hit.score += weights[name] / (_RRF_K + rank)
                setattr(hit, f"{name}_rank", rank)

        ranked = sorted(fused.values(), key=lambda h: -h.score)
        return [h for h in ranked if h.score >= self.s.min_relevance][:k]
