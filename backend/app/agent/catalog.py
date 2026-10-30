"""Liora's product catalogue (``data/catalog.json``) and the name matching around it.

The catalogue is the system of record for prices. Tools read it directly so a price
in an answer never depends on what a model recalled or what retrieval happened to find.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from app.config import BACKEND_DIR
from app.textutils import content_tokens

CATALOG_PATH = BACKEND_DIR / "data" / "catalog.json"
WHATSAPP_NUMBER = "85913 53636"

# Words customers use for each catalogue category, checked in this order.
CATEGORY_ALIASES = {
    "Diwali Hampers": ("hamper", "gift set"),
    "Candle Bouquets": ("bouquet",),
    "Scented Jar Candles": ("jar candle", "glass jar", "ocean"),
    "Wax & Tea Lights": ("wax melt", "wax sachet", "sachet", "tea light", "tealight"),
    "Urlis & Tapers": ("urli", "taper"),
    "Roses & Peonies": ("rose", "peony", "peonies"),
    "Shaped Candles": ("shaped candle", "pillar", "daisy", "sunflower", "tulip", "ladoo"),
}
_CODE_RE = re.compile(r"\bLS[ -]?(\d{3})\b", re.I)


@lru_cache
def load_catalog() -> tuple[dict, ...]:
    return tuple(json.loads(CATALOG_PATH.read_text(encoding="utf-8")))


def format_price(price_inr: int) -> str:
    return f"₹{price_inr:,}"


def public_view(product: dict) -> dict:
    return {"image": f"/products/{product['code']}.jpg", **product, "price": format_price(product["price_inr"])}


def base_name(name: str) -> str:
    return name.split(" · ")[0]


def _name_parts(name: str) -> tuple[set[str], set[str]]:
    """Split "Glass Jar Candle · 400 ml" into its base tokens and its variant tokens."""
    base, _, variant = name.partition(" · ")
    return set(content_tokens(base)), set(content_tokens(variant))


def match_products(text: str, min_base_share: float = 1.0) -> list[dict]:
    """Products named in ``text``, by code or by name. A shared base name with no variant
    ("glass jar candle") returns every size; a variant in the text ("400 ml") narrows it."""
    code = _CODE_RE.search(text)
    if code:
        return [p for p in load_catalog() if p["code"] == f"LS{code.group(1)}"]
    tokens = set(content_tokens(text))
    scored = []
    for product in load_catalog():
        base, variant = _name_parts(product["name"])
        share = len(base & tokens) / len(base)
        if share >= min_base_share:
            scored.append(((share, len(variant & tokens), len(base)), product))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [p for score, p in scored if score == best]


def guess_category(text: str) -> str:
    lowered = text.lower()
    for category, aliases in CATEGORY_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return category
    return ""


def resolve_category(text: str) -> str | None:
    """Map free text to a catalogue category; None when it names nothing we sell."""
    if not text.strip():
        return ""
    for product in load_catalog():
        if product["category"].lower() == text.strip().lower():
            return product["category"]
    return guess_category(text) or None
