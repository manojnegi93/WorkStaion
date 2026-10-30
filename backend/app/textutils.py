"""Small, dependency-free text helpers shared by retrieval, guardrails and evaluation."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}|\n(?=[-*•]|\d+\.)")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "could",
        "do",
        "does",
        "did",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "please",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "any",
        "all",
        "also",
        "am",
        "been",
        "being",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "too",
        "very",
        "just",
        "tell",
        "know",
        "get",
        "got",
        "us",
    ]
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# Light domain normalisation so lexical matching treats synonyms alike.
SYNONYMS = {
    "fragrance": "scent",
    "aroma": "scent",
    "smell": "scent",
    "perfume": "scent",
    "centerpiece": "centrepiece",
    "cost": "price",
    "closet": "wardrobe",
    "cupboard": "wardrobe",
    "burner": "warmer",
    "tealight": "tea-light",
    "stor": "store",
}


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed, a light plural/verb stem and synonym folding."""
    out = []
    for t in tokenize(text):
        if t in STOPWORDS or len(t) < 2:
            continue
        t = stem(t)
        out.append(SYNONYMS.get(t, t))
    return out


def stem(t: str) -> str:
    if len(t) <= 4 or t.endswith("ss"):
        return t
    if t.endswith("ies"):
        return t[:-3] + "y"
    if t.endswith(("ches", "shes", "sses", "xes", "zes")):
        return t[:-2]
    if t.endswith("ing") and len(t) > 6:
        return t[:-3]
    if t.endswith("ed") and len(t) > 5:
        return t[:-2]
    if t.endswith("s"):
        return t[:-1]
    return t


def split_sentences(text: str) -> list[str]:
    parts = [p.strip(" \n-*•") for p in _SENT_RE.split(text)]
    return [p for p in parts if len(p) > 3]


def estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic; good enough for budgeting and cost estimates."""
    return max(1, len(text) // 4)


def overlap_ratio(a: str, b: str) -> float:
    """Share of content tokens in `a` that also appear in `b`."""
    ta = set(content_tokens(a))
    if not ta:
        return 0.0
    tb = set(content_tokens(b))
    return len(ta & tb) / len(ta)
