"""Central configuration (12-factor style, read from env / .env).

Everything defaults to free, open-source, local components:
  * LLM + embeddings  -> Ollama (Llama 3.1 / Qwen 2.5, nomic-embed-text)
  * Vector store      -> Qdrant (embedded on-disk mode, or a Qdrant server)
  * Tracing           -> built-in step tracer + optional self-hosted Langfuse
  * Metrics           -> Prometheus /metrics, visualised in Grafana

`LLM_PROVIDER=offline` runs a deterministic, dependency-free model so the
whole system (and CI) works with no model downloaded at all.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    app_name: str = "Liora Assistant"
    environment: str = "local"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ---- LLM ---------------------------------------------------------------
    # ollama  : local open-source models (default)
    # openai_compatible : any OpenAI-compatible server (vLLM, LM Studio, llama.cpp, LocalAI)
    # offline : deterministic extractive model, no downloads (tests / CI / demo)
    llm_provider: Literal["ollama", "openai_compatible", "offline"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.1:8b"
    judge_model: str = ""  # model used for LLM-as-judge evaluation (defaults to chat_model)
    temperature: float = 0.1
    openai_compatible_base_url: str = "http://localhost:8001/v1"
    openai_compatible_api_key: str = "not-needed"

    # ---- Embeddings --------------------------------------------------------
    embedding_provider: Literal["ollama", "openai_compatible", "hashing"] = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768  # nomic-embed-text = 768; hashing embedder uses this too

    # ---- Vector store (Qdrant) --------------------------------------------
    qdrant_url: str = ""  # e.g. http://localhost:6333 ; empty -> embedded mode
    qdrant_path: str = str(BACKEND_DIR / ".qdrant")  # used in embedded mode; ":memory:" allowed
    collection_name: str = "knowledge_base"

    # ---- Retrieval / context engineering ----------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    candidate_k: int = 20  # per-retriever candidates before RRF fusion
    min_relevance: float = 0.0  # drop fused hits below this RRF score
    max_context_tokens: int = 2500

    # ---- Agent -------------------------------------------------------------
    max_agent_steps: int = 4

    # ---- Guardrails --------------------------------------------------------
    enable_guardrails: bool = True
    max_input_chars: int = 4000
    min_groundedness: float = 0.35

    # ---- Observability -----------------------------------------------------
    langfuse_host: str = ""  # self-hosted Langfuse, e.g. http://localhost:3001
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # ---- Data --------------------------------------------------------------
    docs_dir: str = str(BACKEND_DIR / "data" / "knowledge_base")
    auto_ingest_on_startup: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
