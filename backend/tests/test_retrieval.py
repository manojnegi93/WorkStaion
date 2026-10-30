import pytest
from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingestion.chunker import chunk_document
from app.llm.offline import HashingEmbeddings
from app.retrieval.context import build_context, dedupe
from app.retrieval.vector_store import HybridVectorStore, RetrievedChunk


@pytest.fixture
def store():
    s = HybridVectorStore(HashingEmbeddings(256), get_settings(), client=QdrantClient(location=":memory:"))
    s.upsert(chunk_document("# Care\n\n## Wicks\nTrim the wick to about 5 mm before every burn.", "care.md"))
    s.upsert(chunk_document("# Melts\n\n## Warmers\nPlace one wax melt in a warmer dish.", "melts.md"))
    return s


@pytest.mark.parametrize("mode", ["hybrid", "dense", "sparse"])
def test_search_modes_find_the_right_source(store, mode):
    hits = store.search("how short should I trim the wick", k=2, mode=mode)
    assert hits and hits[0].source == "care.md"


def test_hybrid_hits_carry_per_retriever_ranks(store):
    top = store.search("wax melt warmer", k=1)[0]
    assert top.source == "melts.md"
    assert top.dense_rank == 1 and top.sparse_rank == 1


def test_reingest_replaces_source(store):
    store.delete_source("care.md")
    assert store.list_sources() == {"melts.md": 1}


def test_context_dedupes_and_numbers_blocks():
    h = RetrievedChunk("1", "Trim the wick to about 5 mm.", "a.md", "S", "T", 0.5)
    dup = RetrievedChunk("2", "Trim the wick to about 5 mm.", "a.md", "S", "T", 0.4)
    assert len(dedupe([h, dup])) == 1
    ctx, ordered = build_context([h, dup], 1000)
    assert "[1] source: a.md | S" in ctx and len(ordered) == 1
