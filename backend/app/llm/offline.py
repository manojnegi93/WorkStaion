"""Deterministic, download-free stand-ins for an LLM and an embedding model.

They let the full LangGraph pipeline (tool calling, guardrails, evaluation, tracing)
run in CI or on a laptop with no model installed. They are *not* meant to be smart:
the chat model answers extractively from the retrieved context and calls tools
using simple pattern rules, which is exactly what the tests need to be reproducible.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.catalog import base_name, guess_category, match_products
from app.textutils import content_tokens, estimate_tokens, split_sentences, tokenize

# --------------------------------------------------------------------------- embeddings


class HashingEmbeddings(Embeddings):
    """Feature-hashing bag of (stemmed) unigrams + bigrams, L2-normalised."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = content_tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# --------------------------------------------------------------------------- chat model

_CTX_BLOCK = re.compile(r"\[(\d+)\] source: ([^\n]*)\n(.*?)(?=\n\[\d+\] source:|\n?</context>)", re.S)

_TOTAL = re.compile(r"\b(total|altogether|in all|add up)\b", re.I)
_BUDGET = re.compile(
    r"\b(?:under|below|less than|within|up ?to|budget(?: of)?)\s*(?:₹|rs\.?\s*|inr\s*)?(\d[\d,]*)", re.I
)
_BROWSE = re.compile(r"\b(show|list|browse|do you (have|sell)|what\b.*\b(have|sell|available))\b", re.I)
_ORDER_SPLIT = re.compile(r",|\bplus\b|\band\b(?! daisy)", re.I)


def _order_items(m: re.Match[str]) -> dict | None:
    lines = []
    for part in _ORDER_SPLIT.split(m.string):
        matches = match_products(part)
        if len(matches) == 1:
            qty = re.search(r"\b(\d+)\s*(?:x|×)?\s+[a-z]", part, re.I)
            lines.append(f"{qty.group(1) if qty else 1} x {matches[0]['name']}")
    return {"items": ", ".join(lines)} if lines else None


def _budget_args(m: re.Match[str]) -> dict:
    return {"max_price": int(m.group(1).replace(",", "")), "category": guess_category(m.string)}


def _named_product(m: re.Match[str]) -> dict | None:
    matches = match_products(m.string)
    if not matches:
        return None
    return {"product": matches[0]["name"] if len(matches) == 1 else base_name(matches[0]["name"])}


def _browse_args(m: re.Match[str]) -> dict | None:
    category = guess_category(m.string)
    return {"category": category} if category else None


# (tool name, regex on the user question, arg builder returning None to skip the rule)
_TOOL_RULES: list[tuple[str, re.Pattern[str], Any]] = [
    ("calculate_order_total", _TOTAL, _order_items),
    ("find_products", _BUDGET, _budget_args),
    ("get_product_details", re.compile(r".+", re.S), _named_product),
    ("find_products", _BROWSE, _browse_args),
]


def _last_question(messages: list[BaseMessage]) -> tuple[str, list[ToolMessage]]:
    """Return the latest human question and any tool results produced after it."""
    tool_msgs: list[ToolMessage] = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_msgs.append(msg)
        elif isinstance(msg, HumanMessage):
            return str(msg.content), list(reversed(tool_msgs))
    return "", tool_msgs


class OfflineChatModel(BaseChatModel):
    """Extractive answerer with rule-based tool calling. Deterministic by design."""

    bound_tools: list[str] = []
    max_sentences: int = 3

    @property
    def _llm_type(self) -> str:
        return "offline-extractive"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> OfflineChatModel:  # type: ignore[override]
        names = [getattr(t, "name", None) or t.get("name") for t in tools]
        return self.model_copy(update={"bound_tools": [n for n in names if n]})

    # -- core ---------------------------------------------------------------
    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        question, tool_results = _last_question(messages)
        system = next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")

        msg: AIMessage
        if self.bound_tools and not tool_results:
            call = self._maybe_tool_call(question)
            if call:
                msg = AIMessage(content="", tool_calls=[call])
                return self._result(msg, messages)

        msg = AIMessage(content=self._answer(question, system, tool_results))
        return self._result(msg, messages)

    def _maybe_tool_call(self, question: str) -> dict | None:
        for name, pattern, build in _TOOL_RULES:
            if name not in self.bound_tools:
                continue
            m = pattern.search(question)
            args = build(m) if m else None
            if args:
                return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}
        return None

    def _answer(self, question: str, system: str, tool_results: list[ToolMessage]) -> str:
        if "Respond with JSON" in system:  # used by LLM-as-judge prompts in offline mode
            return json.dumps({"score": 0.5, "reason": "offline model cannot judge"})

        parts: list[str] = []
        for tm in tool_results:
            parts.append(_summarise_tool(tm))

        q_tokens = set(content_tokens(question))
        wants_number = bool(re.match(r"\s*(?:and\s+)?(how (long|much|many|often|far)|when|what time)", question, re.I))
        scored: list[tuple[float, int, int, str]] = []
        # a catalogue tool result is the answer, so don't pad it with loosely related passages
        for idx_s, _src, body in [] if tool_results else _CTX_BLOCK.findall(system):
            for pos, sent in enumerate(split_sentences(body)):
                if sent.lstrip().startswith("#"):
                    continue
                s_tokens = set(content_tokens(sent))
                if not s_tokens:
                    continue
                hits = len(q_tokens & s_tokens)
                if hits == 0:
                    continue
                # one shared word out of many ("cup" in a football question) is not an answer
                if hits / len(q_tokens) < 1 / 3:
                    continue
                score = hits / len(s_tokens) ** 0.3 + 0.01 / (pos + 1)
                if wants_number and re.search(r"\d", sent):
                    score += 0.5
                scored.append((score, int(idx_s), pos, sent))

        scored.sort(key=lambda x: -x[0])
        picked = scored[: self.max_sentences]
        picked.sort(key=lambda x: (x[1], x[2]))
        for _, idx, _, sent in picked:
            sent = sent.rstrip()
            if not sent.endswith((".", "!", "?")):
                sent += "."
            parts.append(f"{sent} [{idx}]")

        if not parts:
            return "I don't have enough information in the knowledge base to answer that."
        return " ".join(parts)

    @staticmethod
    def _result(msg: AIMessage, messages: list[BaseMessage]) -> ChatResult:
        prompt_tokens = sum(estimate_tokens(str(m.content)) for m in messages)
        completion_tokens = estimate_tokens(str(msg.content) or json.dumps(msg.tool_calls))
        msg.usage_metadata = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _product_line(product: dict) -> str:
    code = f" ({product['code']})" if product["code"] else ""
    return f"{product['name']}{code} at {product['price']}"


def _summarise_tool(tm: ToolMessage) -> str:
    try:
        data = json.loads(str(tm.content))
    except (json.JSONDecodeError, TypeError):
        return str(tm.content)
    if isinstance(data, dict) and "error" in data:
        return f"Tool {tm.name} reported: {data['error']}."
    if isinstance(data, dict) and "products" in data:
        if not data["products"]:
            return "I couldn't find a product matching that."
        return "Here is what I found: " + "; ".join(_product_line(p) for p in data["products"]) + "."
    if isinstance(data, dict) and "product" in data:
        return f"{_product_line(data['product'])}. {data['product']['description']}"
    if isinstance(data, dict) and "lines" in data:
        return (
            f"{'; '.join(data['lines'])}. Subtotal before shipping: {data['subtotal']}. "
            f"To order, send {data['order_codes']} on WhatsApp to {data['whatsapp']}."
        )
    if isinstance(data, dict):
        fields = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in data.items() if not isinstance(v, (dict, list)))
        return f"From {tm.name.replace('_', ' ')} — {fields}."
    return f"From {tm.name}: {data}"


__all__ = ["HashingEmbeddings", "OfflineChatModel", "tokenize"]
