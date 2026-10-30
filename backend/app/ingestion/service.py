"""Ingestion pipeline: load -> chunk -> embed -> upsert (idempotent per source)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.config import get_settings
from app.ingestion.chunker import SUPPORTED_EXTENSIONS, chunk_document, chunk_file, load_text
from app.retrieval.vector_store import HybridVectorStore

log = logging.getLogger(__name__)


def ingest_text(store: HybridVectorStore, text: str, source: str) -> int:
    s = get_settings()
    chunks = chunk_document(text, source, s.chunk_size, s.chunk_overlap)
    store.delete_source(source)  # re-ingesting a file replaces its previous chunks
    return store.upsert(chunks)


def ingest_bytes(store: HybridVectorStore, data: bytes, filename: str) -> int:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type {suffix!r}; allowed: {sorted(SUPPORTED_EXTENSIONS)}")
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(data)
            tmp.flush()
            text = load_text(Path(tmp.name))
    else:
        text = data.decode("utf-8", errors="ignore")
    return ingest_text(store, text, Path(filename).name)


def ingest_directory(store: HybridVectorStore, directory: str | Path) -> dict[str, int]:
    s = get_settings()
    results: dict[str, int] = {}
    for path in sorted(Path(directory).glob("**/*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            store.delete_source(path.name)
            results[path.name] = store.upsert(chunk_file(path, s.chunk_size, s.chunk_overlap))
    log.info("Ingested %d files (%d chunks)", len(results), sum(results.values()))
    return results
