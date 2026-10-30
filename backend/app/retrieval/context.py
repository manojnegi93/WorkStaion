"""Context engineering: turn raw hits into a compact, citable context window.

Steps
  1. drop near-duplicate chunks (overlapping windows from the chunker)
  2. keep the best hits that fit inside the token budget
  3. "lost in the middle" mitigation: strongest evidence at the edges
  4. render numbered blocks the LLM cites as [n]
"""

from __future__ import annotations

from app.retrieval.vector_store import RetrievedChunk
from app.textutils import estimate_tokens, overlap_ratio


def dedupe(hits: list[RetrievedChunk], threshold: float = 0.85) -> list[RetrievedChunk]:
    kept: list[RetrievedChunk] = []
    for h in hits:
        if any(overlap_ratio(h.text, k.text) >= threshold for k in kept):
            continue
        kept.append(h)
    return kept


def fit_budget(hits: list[RetrievedChunk], max_tokens: int) -> list[RetrievedChunk]:
    out, used = [], 0
    for h in hits:
        t = estimate_tokens(h.text)
        if used + t > max_tokens and out:
            break
        out.append(h)
        used += t
    return out


def edge_order(hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Place rank 1 first, rank 2 last, rank 3 second, ... (LongContextReorder)."""
    front, back = [], []
    for i, h in enumerate(hits):
        (front if i % 2 == 0 else back).append(h)
    return front + back[::-1]


def build_context(hits: list[RetrievedChunk], max_tokens: int) -> tuple[str, list[RetrievedChunk]]:
    selected = fit_budget(dedupe(hits), max_tokens)
    ordered = edge_order(selected)
    blocks = []
    for i, h in enumerate(ordered, start=1):
        label = f"{h.source} | {h.section}" if h.section else h.source
        blocks.append(f"[{i}] source: {label}\n{h.text}")
    return "<context>\n" + "\n".join(blocks) + "\n</context>", ordered
