"""Alternative orchestrator built with Microsoft Semantic Kernel (MIT licence).

The same knowledge base and catalogue tools are exposed as a Semantic Kernel
*plugin*; the model decides which kernel functions to call via
``FunctionChoiceBehavior.Auto()``. This shows the same agent design in both
LangGraph (default, /api/chat) and Semantic Kernel (/api/sk/chat), running on
local Ollama models or any OpenAI-compatible open-source server.
"""

from __future__ import annotations

from typing import Annotated

from app.agent import tools as t
from app.config import get_settings
from app.deps import get_store
from app.guardrails.rails import check_input, check_output
from app.retrieval.context import build_context

try:
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
    from semantic_kernel.contents import ChatHistory
    from semantic_kernel.functions import kernel_function

    SK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SK_AVAILABLE = False

    def kernel_function(*_a, **_k):  # type: ignore[no-redef]
        return lambda f: f


class CatalogPlugin:
    """Semantic Kernel plugin wrapping the knowledge base and product catalogue."""

    @kernel_function(
        name="search_knowledge_base", description="Search Liora's products, candle care, ordering and gifting notes."
    )
    def search_knowledge_base(self, query: Annotated[str, "search query"]) -> str:
        hits = get_store().search(query, k=get_settings().top_k)
        context, _ = build_context(hits, get_settings().max_context_tokens)
        return context

    @kernel_function(
        name="find_products", description="Find products by keywords, category and maximum price in rupees."
    )
    def find_products(
        self,
        query: Annotated[str, "keywords, may be empty"] = "",
        category: Annotated[str, "category name, may be empty"] = "",
        max_price: Annotated[int, "budget in rupees, 0 for no limit"] = 0,
    ) -> str:
        return t.find_products.invoke({"query": query, "category": category, "max_price": max_price})

    @kernel_function(name="get_product_details", description="Get the price, code and description of a product.")
    def get_product_details(self, product: Annotated[str, "product name"]) -> str:
        return t.get_product_details.invoke({"product": product})

    @kernel_function(
        name="calculate_order_total",
        description="Add up an order before shipping. Items look like '2 x Wax Melts, 1 x Wax Sachet'.",
    )
    def calculate_order_total(self, items: Annotated[str, "comma-separated quantity x product"]) -> str:
        return t.calculate_order_total.invoke({"items": items})


SK_SYSTEM = (
    "You are the Liora Assistant for Liora By Srijana, a brand of hand-poured scented candles. "
    "Always call search_knowledge_base before answering questions about candle care, ordering or gifting, "
    "and use the other functions for product prices, browsing and order totals. "
    "Answer only from function results; if they don't contain the answer, say you don't have enough information."
)


def _build_service_and_settings():
    s = get_settings()
    if s.llm_provider == "ollama":
        from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings

        service = OllamaChatCompletion(ai_model_id=s.chat_model, host=s.ollama_base_url)
        settings = OllamaChatPromptExecutionSettings(options={"temperature": s.temperature})
    elif s.llm_provider == "openai_compatible":
        from openai import AsyncOpenAI
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings

        client = AsyncOpenAI(base_url=s.openai_compatible_base_url, api_key=s.openai_compatible_api_key)
        service = OpenAIChatCompletion(ai_model_id=s.chat_model, async_client=client)
        settings = OpenAIChatPromptExecutionSettings(temperature=s.temperature)
    else:
        raise RuntimeError("The Semantic Kernel orchestrator needs LLM_PROVIDER=ollama or openai_compatible.")
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    return service, settings


def build_kernel() -> Kernel:
    if not SK_AVAILABLE:
        raise RuntimeError("semantic-kernel is not installed: pip install 'semantic-kernel[ollama]'")
    kernel = Kernel()
    kernel.add_plugin(CatalogPlugin(), plugin_name="catalog")
    return kernel


async def sk_chat(question: str, history: list[dict] | None = None) -> dict:
    guard = check_input(question)
    if not guard.allowed:
        return {"answer": guard.text, "flags": guard.flags, "blocked": True, "orchestrator": "semantic-kernel"}
    kernel = build_kernel()
    service, settings = _build_service_and_settings()
    chat = ChatHistory(system_message=SK_SYSTEM)
    for turn in (history or [])[-6:]:
        (chat.add_user_message if turn["role"] == "user" else chat.add_assistant_message)(turn["content"])
    chat.add_user_message(guard.text)

    result = await service.get_chat_message_content(chat, settings, kernel=kernel)
    answer = str(result) if result else ""
    # Evidence = every function result produced during auto function calling.
    evidence = "\n".join(
        str(getattr(item, "result", ""))
        for msg in chat.messages
        for item in msg.items
        if type(item).__name__ == "FunctionResultContent"
    )
    out = check_output(answer, evidence, n_sources=99)
    return {
        "answer": out.text,
        "flags": guard.flags + out.flags,
        "blocked": False,
        "groundedness": out.details.get("groundedness"),
        "orchestrator": "semantic-kernel",
    }
