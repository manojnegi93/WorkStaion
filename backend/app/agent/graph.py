"""The agentic RAG workflow, built with LangGraph.

    START
      │
  input_guard ──(blocked)──────────────────────────────┐
      │                                                │
  rewrite_query   (condense follow-ups using history)  │
      │                                                │
   retrieve       (hybrid dense+BM25 search, RRF,      │
      │            dedupe, token budget, reorder)      │
      ▼                                                │
    agent ◄────────┐  (LLM with tools bound)           │
      │            │                                   │
  (tool calls?)──► tools                               │
      │                                                │
  output_guard    (citations, PII, groundedness)       │
      │                                                │
     END ◄─────────────────────────────────────────────┘

Every node is wrapped with ``@traced`` so the response carries a step-level trace.
"""

from __future__ import annotations

import json
import operator
import re
from datetime import date
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent.catalog import WHATSAPP_NUMBER
from app.agent.prompts import PROMPT_VERSION, REWRITE_PROMPT, SYSTEM_PROMPT
from app.agent.tools import TOOLS
from app.config import get_settings
from app.deps import get_store
from app.guardrails.rails import check_input, check_output
from app.llm.providers import get_chat_model
from app.observability.tracing import GROUNDEDNESS, GUARDRAIL_FLAGS, RETRIEVAL_HITS, traced
from app.retrieval.context import build_context
from app.textutils import content_tokens


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}]
    search_query: str
    context: str
    sources: list[dict]
    messages: Annotated[list[BaseMessage], add_messages]  # agent <-> tool loop only
    answer: str
    blocked: bool
    steps: int
    groundedness: float
    flags: Annotated[list[str], operator.add]


def _trace(config: RunnableConfig):
    return (config.get("configurable") or {}).get("trace")


def _model_name() -> str:
    s = get_settings()
    return "offline" if s.llm_provider == "offline" else s.chat_model


def _history_messages(history: list[dict], limit: int = 6) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for turn in history[-limit:]:
        cls = HumanMessage if turn.get("role") == "user" else AIMessage
        out.append(cls(content=turn.get("content", "")))
    return out


# --------------------------------------------------------------------------- nodes
@traced("input_guard")
def input_guard(state: AgentState, config: RunnableConfig) -> dict:
    res = check_input(state["question"])
    for f in res.flags:
        GUARDRAIL_FLAGS.labels(f.split(":")[0]).inc()
    if not res.allowed:
        return {
            "blocked": True,
            "answer": res.text,
            "flags": res.flags,
            "sources": [],
            "_span": {"allowed": False, "flags": res.flags},
        }
    return {"blocked": False, "question": res.text, "flags": res.flags, "_span": {"allowed": True, "flags": res.flags}}


@traced("rewrite_query")
def rewrite_query(state: AgentState, config: RunnableConfig) -> dict:
    history = state.get("history") or []
    question = state["question"]
    if not history:
        return {"search_query": question, "_span": {"rewritten": False}}
    if get_settings().llm_provider == "offline":
        # Heuristic: only short or anaphoric messages are follow-ups that need history.
        follow_up = (
            re.match(r"\s*(and|what about|how about|also|what if|same for)\b", question, re.I)
            or len(content_tokens(question)) <= 2
        )
        if not follow_up:
            return {"search_query": question, "_span": {"rewritten": False}}
        last_user = next((t["content"] for t in reversed(history) if t.get("role") == "user"), "")
        query = f"{last_user} {question}".strip()
    else:
        convo = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
        msg = get_chat_model().invoke(REWRITE_PROMPT.format(history=convo, question=question), config)
        if (t := _trace(config)) is not None:
            t.add_usage(_model_name(), getattr(msg, "usage_metadata", None))
        query = str(msg.content).strip().strip('"') or question
    return {"search_query": query, "_span": {"rewritten": True, "query": query}}


@traced("retrieve")
def retrieve(state: AgentState, config: RunnableConfig) -> dict:
    s = get_settings()
    hits = get_store().search(state["search_query"], k=s.top_k)
    context, ordered = build_context(hits, s.max_context_tokens)
    RETRIEVAL_HITS.observe(len(ordered))
    sources = [{"id": i, **h.to_dict()} for i, h in enumerate(ordered, start=1)]
    return {
        "context": context,
        "sources": sources,
        "_span": {
            "query": state["search_query"],
            "candidates": len(hits),
            "in_context": len(ordered),
            "top": [
                {
                    "source": h.source,
                    "section": h.section,
                    "score": round(h.score, 4),
                    "dense_rank": h.dense_rank,
                    "sparse_rank": h.sparse_rank,
                }
                for h in ordered
            ],
        },
    }


def _system_prompt(state: AgentState) -> str:
    return SYSTEM_PROMPT.format(today=date.today().isoformat(), context=state.get("context", "<context>\n</context>"))


@traced("agent")
def agent(state: AgentState, config: RunnableConfig) -> dict:
    s = get_settings()
    steps = state.get("steps", 0) + 1
    llm = get_chat_model()
    # On the last allowed step, call the model without tools to force a final answer.
    if steps < s.max_agent_steps:
        llm = llm.bind_tools(TOOLS)
    messages = [
        SystemMessage(_system_prompt(state)),
        *_history_messages(state.get("history") or []),
        HumanMessage(state["question"]),
        *state.get("messages", []),
    ]
    response: AIMessage = llm.invoke(messages, config)
    usage = getattr(response, "usage_metadata", None)
    if (t := _trace(config)) is not None:
        t.add_usage(_model_name(), usage)
    calls = [{"name": c["name"], "args": c["args"]} for c in (response.tool_calls or [])]
    return {
        "messages": [response],
        "steps": steps,
        "answer": "" if calls else str(response.content),
        "_span": {"step": steps, "tool_calls": calls, "usage": usage or {}, "prompt_version": PROMPT_VERSION},
    }


_tool_node = ToolNode(TOOLS)


@traced("tools")
def tools(state: AgentState, config: RunnableConfig) -> dict:
    result = _tool_node.invoke(state, config)
    outputs = []
    for m in result["messages"]:
        try:
            content: Any = json.loads(m.content)
        except (json.JSONDecodeError, TypeError):
            content = str(m.content)[:400]
        outputs.append({"tool": m.name, "output": content})
    return {"messages": result["messages"], "_span": {"results": outputs}}


@traced("output_guard")
def output_guard(state: AgentState, config: RunnableConfig) -> dict:
    tool_text = "\n".join(str(m.content) for m in state.get("messages", []) if isinstance(m, ToolMessage))
    evidence = f"{state.get('context', '')}\n{tool_text}\n{WHATSAPP_NUMBER}"
    res = check_output(state.get("answer", ""), evidence, len(state.get("sources", [])), _system_prompt(state))
    for f in res.flags:
        GUARDRAIL_FLAGS.labels(f.split(":")[0]).inc()
    g = res.details.get("groundedness")
    if g is not None:
        GROUNDEDNESS.observe(g)
    return {
        "answer": res.text,
        "flags": res.flags,
        "groundedness": g if g is not None else 0.0,
        "_span": {"flags": res.flags, **res.details},
    }


# --------------------------------------------------------------------------- edges
def _after_input(state: AgentState) -> str:
    return END if state.get("blocked") else "rewrite_query"


def _after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "output_guard"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("input_guard", input_guard)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("retrieve", retrieve)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("output_guard", output_guard)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", _after_input, ["rewrite_query", END])
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("retrieve", "agent")
    g.add_conditional_edges("agent", _after_agent, ["tools", "output_guard"])
    g.add_edge("tools", "agent")
    g.add_edge("output_guard", END)
    return g.compile()


@lru_cache
def get_graph():
    return build_graph()
