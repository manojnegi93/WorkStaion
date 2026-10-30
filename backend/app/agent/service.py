"""Thin service layer between the API and the LangGraph workflow."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from langchain_core.messages import AIMessageChunk

from app.agent.graph import get_graph
from app.agent.prompts import PROMPT_VERSION
from app.config import get_settings
from app.observability.tracing import REQUEST_LATENCY, REQUESTS, RequestTrace, langfuse_callbacks


def _config(trace: RequestTrace) -> dict:
    return {
        "configurable": {"trace": trace},
        "callbacks": langfuse_callbacks(),
        "recursion_limit": 4 * get_settings().max_agent_steps + 8,
        "run_name": "liora-assistant",
        "metadata": {"trace_id": trace.trace_id, "prompt_version": PROMPT_VERSION},
    }


def _response(state: dict, trace: RequestTrace) -> dict:
    s = get_settings()
    return {
        "answer": state.get("answer", ""),
        "sources": state.get("sources", []),
        "flags": sorted(set(state.get("flags", []))),
        "blocked": bool(state.get("blocked")),
        "groundedness": state.get("groundedness"),
        "search_query": state.get("search_query"),
        "model": "offline" if s.llm_provider == "offline" else s.chat_model,
        "prompt_version": PROMPT_VERSION,
        "trace": trace.to_dict(),
    }


def run_chat(question: str, history: list[dict] | None = None) -> dict:
    trace = RequestTrace()
    t0 = time.perf_counter()
    try:
        state = get_graph().invoke({"question": question, "history": history or [], "flags": []}, _config(trace))
        REQUESTS.labels("blocked" if state.get("blocked") else "ok").inc()
        return _response(state, trace)
    except Exception:
        REQUESTS.labels("error").inc()
        raise
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - t0)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def stream_chat(question: str, history: list[dict] | None = None) -> Iterator[str]:
    """Server-Sent Events: `step` per graph node, `token` while the agent writes, then `final`."""
    trace = RequestTrace()
    t0 = time.perf_counter()
    state: dict = {"question": question, "history": history or [], "flags": []}
    status = "ok"
    try:
        for mode, payload in get_graph().stream(state, _config(trace), stream_mode=["updates", "messages"]):
            if mode == "messages":
                chunk, meta = payload
                if meta.get("langgraph_node") == "agent" and isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield _sse("token", {"text": chunk.content})
                continue
            for node, update in payload.items():
                if not update:
                    continue
                for key, value in update.items():
                    if key in ("flags",):
                        state.setdefault("flags", []).extend(value)
                    elif key != "messages":
                        state[key] = value
                span = trace.spans[-1].__dict__ if trace.spans else {}
                yield _sse("step", {"node": node, "span": span})
        if state.get("blocked"):
            status = "blocked"
        yield _sse("final", _response(state, trace))
    except Exception as exc:  # surface errors to the UI instead of a dropped connection
        status = "error"
        yield _sse("error", {"message": str(exc)})
    finally:
        REQUESTS.labels(status).inc()
        REQUEST_LATENCY.observe(time.perf_counter() - t0)
