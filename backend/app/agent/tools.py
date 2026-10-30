"""Tools the agent can call. Each tool is a plain, typed function with a docstring
that becomes the tool schema the LLM sees (LangChain ``@tool``).

They read the product catalogue in ``data/catalog.json``. Liora takes orders on WhatsApp, so
there are no stock or order-status tools; add them if a system of record appears
(see the ``add-agent-tool`` Claude Skill).
"""

from __future__ import annotations

import json
import re

from langchain_core.tools import tool

from app.agent.catalog import WHATSAPP_NUMBER, format_price, load_catalog, match_products, public_view, resolve_category
from app.observability.tracing import TOOL_CALLS
from app.textutils import content_tokens

MAX_RESULTS = 8
_ORDER_LINE = re.compile(r"^\s*(?:(\d+)\s*(?:x|×|\*)?\s+)?(.+?)\s*$", re.I)


def _out(name: str, payload: dict) -> str:
    TOOL_CALLS.labels(name).inc()
    return json.dumps(payload, ensure_ascii=False)


@tool
def find_products(query: str = "", category: str = "", max_price: int = 0) -> str:
    """Browse Liora's products, cheapest first. All arguments are optional: `query` is keywords such as
    "rose" or "return gifts", `category` is a range such as "Diwali Hampers" or "Urlis & Tapers", and
    `max_price` is a budget in rupees (0 means no limit). Use for "what do you have" and
    "something under ₹300" questions."""
    resolved = resolve_category(category)
    if resolved is None:
        known = sorted({p["category"] for p in load_catalog()})
        return _out("find_products", {"error": f"no category called {category}", "known_categories": known})
    terms = set(content_tokens(query))
    matches = []
    for product in load_catalog():
        if resolved and product["category"] != resolved:
            continue
        if max_price and product["price_inr"] > max_price:
            continue
        haystack = content_tokens(f"{product['name']} {product['category']} {product['description']}")
        if terms and not terms & set(haystack):
            continue
        matches.append(product)
    matches.sort(key=lambda p: p["price_inr"])
    return _out(
        "find_products",
        {"count": len(matches), "products": [public_view(p) for p in matches[:MAX_RESULTS]]},
    )


@tool
def get_product_details(product: str) -> str:
    """Get the exact price, product code and description of a product by name or code, e.g. "Wax Sachet",
    "Glass Jar Candle 400 ml" or "LS014". Hampers list what is inside. Always use this rather than
    guessing a price."""
    matches = match_products(product, min_base_share=0.6)
    if not matches:
        names = [p["name"] for p in load_catalog()]
        return _out("get_product_details", {"error": f"no product called {product}", "known_products": names})
    if len(matches) > 1:
        return _out(
            "get_product_details",
            {"note": "this name covers several products", "products": [public_view(p) for p in matches]},
        )
    return _out("get_product_details", {"product": public_view(matches[0])})


@tool
def calculate_order_total(items: str) -> str:
    """Add up an order before shipping and prepare the WhatsApp order message. `items` is comma-separated
    "quantity x product name or code", e.g. "2 x Wax Melts, 1 x LS001". Use this for any sum; never add
    prices yourself."""
    lines, order, subtotal = [], [], 0
    for raw in filter(str.strip, items.split(",")):
        qty_text, name = _ORDER_LINE.match(raw).groups()
        qty = int(qty_text) if qty_text else 1
        matches = match_products(name, min_base_share=0.6)
        if qty < 1:
            return _out("calculate_order_total", {"error": f"quantity must be at least 1 for {name}"})
        if not matches:
            return _out("calculate_order_total", {"error": f"no product called {name}"})
        if len(matches) > 1:
            options = [p["name"] for p in matches]
            return _out("calculate_order_total", {"error": f"{name} is ambiguous", "choose_one_of": options})
        product = matches[0]
        amount = qty * product["price_inr"]
        subtotal += amount
        lines.append(f"{qty} x {product['name']} = {format_price(amount)}")
        order.append(f"{product['code'] or product['name']} x {qty}")
    if not lines:
        return _out("calculate_order_total", {"error": "no items given"})
    return _out(
        "calculate_order_total",
        {
            "lines": lines,
            "subtotal": format_price(subtotal),
            "subtotal_inr": subtotal,
            "excludes": "shipping",
            "order_codes": ", ".join(order),
            "whatsapp": WHATSAPP_NUMBER,
        },
    )


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base (products, candle care, ordering, gifting) again with a different or more
    specific query. Use when the provided context does not contain the answer."""
    from app.deps import get_store

    hits = get_store().search(query, k=3)
    TOOL_CALLS.labels("search_knowledge_base").inc()
    return "\n\n".join(f"(source: {h.source} | {h.section})\n{h.text}" for h in hits) or "No results."


TOOLS = [search_knowledge_base, find_products, get_product_details, calculate_order_total]
