---
name: add-agent-tool
description: Add a new tool the LangGraph agent (and the Semantic Kernel plugin) can call, with tests and eval coverage. Use when asked to connect the assistant to a new system, API or calculation.
---

# Add an agent tool

1. In `backend/app/agent/tools.py`, write a typed function decorated with `@tool`.
   - The docstring is the tool description the LLM sees. Say what it does, when to use it, and
     the exact argument formats (e.g. "product name such as Wax Sachet", "budget in rupees").
   - Return a JSON string. Return `{"error": "..."}` instead of raising, so the agent can recover.
   - Call `TOOL_CALLS.labels("<name>").inc()` (use the `_out` helper) for metrics.
2. Append the tool to `TOOLS` at the bottom of the file.
3. Mirror it in `backend/app/sk/orchestrator.py` as a `@kernel_function` on `CatalogPlugin`.
4. For offline/CI support, add a pattern rule to `_TOOL_RULES` in `backend/app/llm/offline.py`.
5. Add a golden example with `"category": "tool"` and `"expected_tool"` to
   `backend/data/eval/golden.jsonl`, and an API test in `backend/tests/test_agent_api.py`.
6. If the frontend should label the tool step specially, update `detail()` in
   `frontend/src/components/RouteTrace.tsx`.
7. Run `make test` and `make eval`.
