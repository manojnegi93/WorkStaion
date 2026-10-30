"""LLM and embedding provider factory.

All providers are free / open-source:
  * Ollama            - runs Llama 3.1, Qwen 2.5, Mistral, etc. locally (tool calling supported)
  * OpenAI-compatible - any self-hosted server exposing /v1 (vLLM, LM Studio, llama.cpp, LocalAI)
  * offline           - deterministic extractive model + hashing embeddings (no downloads)
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings
from app.llm.offline import HashingEmbeddings, OfflineChatModel


class PrefixedEmbeddings(Embeddings):
    """nomic-embed-text (and e5/bge style models) perform better with task prefixes."""

    def __init__(self, inner: Embeddings, query_prefix: str = "", doc_prefix: str = ""):
        self.inner, self.query_prefix, self.doc_prefix = inner, query_prefix, doc_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_prefix + text)


def build_chat_model(settings: Settings | None = None, model: str | None = None) -> BaseChatModel:
    s = settings or get_settings()
    name = model or s.chat_model
    if s.llm_provider == "offline":
        return OfflineChatModel()
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=name, base_url=s.ollama_base_url, temperature=s.temperature)
    if s.llm_provider == "openai_compatible":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=name,
            base_url=s.openai_compatible_base_url,
            api_key=s.openai_compatible_api_key,
            temperature=s.temperature,
            stream_usage=True,
        )
    raise ValueError(f"Unknown LLM provider: {s.llm_provider}")


@lru_cache
def get_chat_model() -> BaseChatModel:
    return build_chat_model()


@lru_cache
def get_embeddings() -> Embeddings:
    s = get_settings()
    if s.embedding_provider == "hashing" or s.llm_provider == "offline":
        return HashingEmbeddings(dim=s.embedding_dim)
    if s.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        inner = OllamaEmbeddings(model=s.embedding_model, base_url=s.ollama_base_url)
        if "nomic" in s.embedding_model:
            return PrefixedEmbeddings(inner, "search_query: ", "search_document: ")
        return inner
    if s.embedding_provider == "openai_compatible":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=s.embedding_model,
            base_url=s.openai_compatible_base_url,
            api_key=s.openai_compatible_api_key,
            check_embedding_ctx_length=False,
        )
    raise ValueError(f"Unknown embedding provider: {s.embedding_provider}")
