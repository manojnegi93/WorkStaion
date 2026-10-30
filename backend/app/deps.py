"""Process-wide singletons (vector store, compiled graph)."""

from __future__ import annotations

from functools import lru_cache

from app.llm.providers import get_embeddings
from app.retrieval.vector_store import HybridVectorStore


@lru_cache
def get_store() -> HybridVectorStore:
    return HybridVectorStore(get_embeddings())


def reset_singletons() -> None:
    """Used by tests after changing settings."""
    from app.agent import graph
    from app.config import get_settings
    from app.llm import providers

    for fn in (get_store, get_settings, providers.get_chat_model, providers.get_embeddings, graph.get_graph):
        fn.cache_clear()
