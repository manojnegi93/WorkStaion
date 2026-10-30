"""LLMOps: step-level tracing, token accounting and Prometheus metrics.

* ``RequestTrace`` collects one span per LangGraph node (latency + metadata) and
  token usage per LLM call. It is returned to the UI so every answer is explainable.
* Prometheus metrics are exposed on /metrics and visualised by the bundled
  Grafana dashboard (infra/grafana).
* If a self-hosted Langfuse is configured, a LangChain callback handler is added
  so full prompts/completions are also traced there.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from prometheus_client import Counter, Histogram

from app.config import get_settings

log = logging.getLogger(__name__)

REQUESTS = Counter("rag_requests_total", "Chat requests", ["status"])
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds", "End-to-end chat latency", buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
)
NODE_LATENCY = Histogram(
    "rag_node_latency_seconds",
    "Latency per agent step",
    ["node"],
    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
)
TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["model", "kind"])
GUARDRAIL_FLAGS = Counter("rag_guardrail_flags_total", "Guardrail triggers", ["flag"])
TOOL_CALLS = Counter("rag_tool_calls_total", "Agent tool invocations", ["tool"])
RETRIEVAL_HITS = Histogram("rag_retrieval_hits", "Chunks placed in context", buckets=(0, 1, 2, 3, 5, 8, 13))
GROUNDEDNESS = Histogram(
    "rag_answer_groundedness", "Output groundedness score", buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)


@dataclass
class Span:
    name: str
    start_ms: float
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class RequestTrace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.perf_counter)
    spans: list[Span] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    def add_usage(self, model: str, usage: dict | None) -> None:
        if not usage:
            return
        i, o = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        self.input_tokens += i
        self.output_tokens += o
        self.llm_calls += 1
        TOKENS.labels(model, "input").inc(i)
        TOKENS.labels(model, "output").inc(o)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_ms": round((time.perf_counter() - self.started) * 1000, 1),
            "llm_calls": self.llm_calls,
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.input_tokens + self.output_tokens,
            },
            # Local open-source models have no per-token bill; we still report compute usage.
            "cost_usd": 0.0,
            "spans": [
                {
                    "name": s.name,
                    "start_ms": round(s.start_ms, 1),
                    "duration_ms": round(s.duration_ms, 1),
                    "metadata": s.metadata,
                }
                for s in self.spans
            ],
        }


_current: contextvars.ContextVar[RequestTrace | None] = contextvars.ContextVar("rag_trace", default=None)


def start_trace() -> RequestTrace:
    trace = RequestTrace()
    _current.set(trace)
    return trace


def current_trace() -> RequestTrace | None:
    return _current.get()


def traced(name: str) -> Callable:
    """Decorator for LangGraph nodes: records a span and a Prometheus latency sample.

    A node may return ``{"_span": {...}}`` to attach metadata to its span.
    """

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = kwargs.get("config") or (args[1] if len(args) > 1 else None) or {}
            trace = (config.get("configurable") or {}).get("trace") or current_trace()
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            NODE_LATENCY.labels(name).observe(dt)
            meta = result.pop("_span", {}) if isinstance(result, dict) else {}
            if trace is not None:
                trace.spans.append(Span(name, (t0 - trace.started) * 1000, dt * 1000, meta))
            return result

        return wrapper

    return deco


def langfuse_callbacks() -> list:
    """Return a Langfuse LangChain callback if a self-hosted Langfuse is configured."""
    s = get_settings()
    if not (s.langfuse_host and s.langfuse_public_key and s.langfuse_secret_key):
        return []
    try:
        import os

        os.environ.setdefault("LANGFUSE_HOST", s.langfuse_host)
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key)
        from langfuse.langchain import CallbackHandler  # type: ignore

        return [CallbackHandler()]
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning("Langfuse configured but unavailable: %s", exc)
        return []
