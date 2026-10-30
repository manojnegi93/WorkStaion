"""Input and output guardrails (pure Python, no external service).

Input rails  : size limit, prompt-injection / jailbreak detection, PII redaction, abuse filter
Output rails : PII redaction, system-prompt leak check, citation validation, groundedness check,
               unsupported scent claims

Each rail is a small function so they are easy to unit-test and to extend
(see the ``add-guardrail`` Claude Skill in .claude/skills).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import get_settings
from app.textutils import content_tokens, overlap_ratio, split_sentences

REFUSAL_FALLBACK = (
    "I couldn't find a reliable answer to that in the knowledge base. "
    "Please rephrase, or message Liora on WhatsApp at 85913 53636."
)


@dataclass
class GuardrailResult:
    allowed: bool
    text: str
    flags: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- PII
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "aadhaar": re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3,5}\)?[ -]?)\d{3,4}[ -]?\d{4}(?!\w)"),
}


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def redact_pii(text: str, allow: str = "") -> tuple[str, list[str]]:
    """Mask PII. Values that literally appear in ``allow`` (trusted evidence such as a
    phone number returned by a tool) are kept."""
    found: list[str] = []

    def _sub(kind: str):
        def repl(m: re.Match[str]) -> str:
            if kind == "credit_card" and not _luhn_ok(m.group(0)):
                return m.group(0)
            if allow and m.group(0) in allow:
                return m.group(0)
            found.append(kind)
            return f"[REDACTED_{kind.upper()}]"

        return repl

    for kind, pattern in _PII_PATTERNS.items():
        text = pattern.sub(_sub(kind), text)
    return text, sorted(set(found))


# ----------------------------------------------------------------------------- injection / abuse
_INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) (instructions|prompts|rules)",
    r"disregard (your|the|all) (instructions|rules|guidelines)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system|hidden|initial) (prompt|instructions)",
    r"you are now (dan|in developer mode|jailbroken)",
    r"\bdo anything now\b",
    r"pretend (you have|there are) no (rules|restrictions|guidelines)",
    r"act as (an? )?(unfiltered|uncensored)",
    r"</?(system|assistant)>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.I)
_ABUSE_RE = re.compile(r"\b(kill yourself|build a bomb|make a weapon)\b", re.I)


def check_input(text: str) -> GuardrailResult:
    s = get_settings()
    if not s.enable_guardrails:
        return GuardrailResult(True, text)
    if not text.strip():
        return GuardrailResult(False, "Please enter a question.", ["empty_input"])
    if len(text) > s.max_input_chars:
        return GuardrailResult(
            False, f"Your message is too long (max {s.max_input_chars} characters).", ["input_too_long"]
        )
    if _INJECTION_RE.search(text):
        return GuardrailResult(
            False,
            "I can't follow instructions that try to override my configuration. "
            "Ask me anything about Liora's candles or candle care.",
            ["prompt_injection"],
        )
    if _ABUSE_RE.search(text):
        return GuardrailResult(False, "I can't help with that request.", ["unsafe_content"])
    redacted, pii = redact_pii(text)
    flags = [f"pii_redacted:{k}" for k in pii]
    return GuardrailResult(True, redacted, flags, {"pii": pii})


# ----------------------------------------------------------------------------- scent claims
SCENT_FALLBACK = (
    "Individual scents are not listed in the catalogue, so I don't want to guess. "
    "Please ask Liora on WhatsApp at 85913 53636 which scents are available."
)
_SCENT_NOUN = r"(?:scents?|fragrances?|aromas?)"
# "a sweet and subtle scent": the words directly describing a scent
_SCENT_DESCRIPTION_RE = re.compile(rf"\b(?:([a-z-]+)(?:,\s+|\s+and\s+))?([a-z-]+)\s+({_SCENT_NOUN})\b", re.I)
_SCENT_RANGE_RE = re.compile(rf"\b(?:variety|range|selection|lots|plenty) of {_SCENT_NOUN}\b", re.I)
_SMELLS_LIKE_RE = re.compile(r"\bsmells? (?:of|like)\b", re.I)
_SCENT_NOTES_RE = re.compile(
    r"\b(lavender|vanilla|jasmine|sandalwood|citrus|lemongrass|oud|musk|cinnamon|coffee|mogra|patchouli|bergamot)\b",
    re.I,
)
# Words next to "scent" that don't describe how anything smells.
_NEUTRAL_SCENT_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "about",
        "all",
        "available",
        "be",
        "by",
        "can",
        "custom",
        "different",
        "do",
        "does",
        "each",
        "every",
        "favourite",
        "favorite",
        "for",
        "from",
        "have",
        "has",
        "here",
        "if",
        "in",
        "individual",
        "is",
        "it",
        "its",
        "know",
        "list",
        "listed",
        "many",
        "more",
        "my",
        "new",
        "no",
        "not",
        "of",
        "on",
        "or",
        "other",
        "our",
        "own",
        "please",
        "preferred",
        "preserve",
        "same",
        "see",
        "some",
        "specific",
        "tell",
        "that",
        "the",
        "their",
        "these",
        "this",
        "those",
        "to",
        "us",
        "want",
        "what",
        "which",
        "with",
        "your",
        "you",
    ]
)


def _normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def unsupported_scent_claim(answer: str, evidence: str) -> str | None:
    """Return the first scent claim in ``answer`` that the evidence doesn't contain. Product descriptions
    are about looks, so a model can easily invent a fragrance that overlap scoring won't catch."""
    text, source = _normalise_space(_CITATION_RE.sub("", answer)), _normalise_space(evidence)
    for pattern in (_SCENT_RANGE_RE, _SMELLS_LIKE_RE, _SCENT_NOTES_RE):
        for m in pattern.finditer(text):
            if m.group(0) not in source:
                return m.group(0)
    for m in _SCENT_DESCRIPTION_RE.finditer(text):
        first, second, noun = m.groups()
        for word in filter(None, (first, second)):
            if word not in _NEUTRAL_SCENT_WORDS and f"{word} {noun}" not in source and m.group(0) not in source:
                return m.group(0)
    return None


# ----------------------------------------------------------------------------- output
_CITATION_RE = re.compile(r"\[(\d+)\]")


def groundedness(answer: str, evidence: str) -> float:
    """Mean share of each answer sentence's content words found in the evidence."""
    sentences = [s for s in split_sentences(_CITATION_RE.sub("", answer)) if content_tokens(s)]
    if not sentences:
        return 1.0
    return sum(overlap_ratio(s, evidence) for s in sentences) / len(sentences)


def check_output(answer: str, evidence: str, n_sources: int, system_prompt: str = "") -> GuardrailResult:
    s = get_settings()
    if not s.enable_guardrails:
        return GuardrailResult(True, answer)
    flags: list[str] = []

    # system prompt leakage: long verbatim runs of the system prompt in the answer
    if system_prompt:
        head = system_prompt.strip()[:120]
        if len(head) > 40 and head in answer:
            return GuardrailResult(False, REFUSAL_FALLBACK, ["system_prompt_leak"])

    # citation validation: strip references to sources that don't exist
    def _fix(m: re.Match[str]) -> str:
        return m.group(0) if 1 <= int(m.group(1)) <= n_sources else ""

    cleaned = _CITATION_RE.sub(_fix, answer)
    if cleaned != answer:
        flags.append("invalid_citation_removed")

    cleaned, pii = redact_pii(cleaned, allow=evidence)
    flags += [f"pii_redacted:{k}" for k in pii]

    is_refusal = "don't have enough information" in cleaned.lower() or "couldn't find" in cleaned.lower()
    score = 1.0 if is_refusal else groundedness(cleaned, evidence)
    details = {"groundedness": round(score, 3)}
    claim = unsupported_scent_claim(cleaned, evidence)
    if claim:
        details["unsupported_scent_claim"] = claim
        return GuardrailResult(True, SCENT_FALLBACK, flags + ["unsupported_scent_claim"], details)
    if score < s.min_groundedness / 2:
        return GuardrailResult(True, REFUSAL_FALLBACK, flags + ["ungrounded_answer_blocked"], details)
    if score < s.min_groundedness:
        flags.append("low_groundedness")
        cleaned += "\n\n_Note: parts of this answer may not be fully supported by the knowledge base._"
    return GuardrailResult(True, cleaned.strip(), flags, details)
