# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is
Liora Assistant: an agentic RAG assistant for customers of Liora By Srijana, a hand-poured candle brand. Python
FastAPI + LangGraph backend, React + TypeScript frontend. The stack is free and open source: Ollama (local LLMs and
embeddings), Qdrant (hybrid dense + BM25 search), LangChain, LangGraph, Semantic Kernel, Prometheus and Grafana.

## Layout
- `backend/data/catalog.json` — the system of record for products and prices: the 2026 catalogue PDF, with online shop prices taking priority where both list a product. Shop-only products have `code: null` and are ordered by name
- `backend/data/knowledge_base/` — Markdown the assistant retrieves from; `product-catalog.md` must match `catalog.json`
- `backend/app/agent/graph.py` — LangGraph workflow: input_guard → rewrite_query → retrieve → agent ⇄ tools → output_guard
- `backend/app/agent/catalog.py` — catalogue loading and product name matching
- `backend/app/agent/tools.py` — tools the agent can call (`@tool` functions)
- `backend/app/agent/prompts.py` — versioned prompts; bump `PROMPT_VERSION` on every prompt change
- `backend/app/retrieval/` — Qdrant hybrid store (RRF fusion) and context engineering
- `backend/app/guardrails/rails.py` — input and output guardrails
- `backend/app/evaluation/` — metrics and release gate; dataset in `backend/data/eval/golden.jsonl`
- `backend/app/sk/orchestrator.py` — the same agent built on Semantic Kernel
- `backend/app/llm/offline.py` — deterministic model used by tests and CI
- `frontend/src/` — chat UI; `components/RouteTrace.tsx` draws the agent trace, `components/ProductCards.tsx` shows tool results

## Commands
- `make test` — pytest (always offline, deterministic, under 5 seconds)
- `make eval` — evaluation suite and release gate; it must pass before you finish a change
- `make lint` — ruff and the TypeScript check
- `make demo` — run the app with no model downloaded (`./run-demo.ps1` on Windows without make)

## Rules
- Run `make test` and `make eval` after every backend change. Don't lower thresholds in
  `data/eval/thresholds.yaml` to make a change pass. Fix the cause, or explain why to the user.
- New behaviour needs a golden example in `golden.jsonl` and a unit test.
- Never invent product facts, prices or policies. Everything in `data/` must come from Liora's own material.
- When prices change, update `catalog.json` and `product-catalog.md` together; a test checks they agree.
- Keep everything runnable offline. Don't add paid APIs or closed-source dependencies.
- Prompt changes: edit `prompts.py`, bump `PROMPT_VERSION`, run `make eval-judge` if Ollama is available.
- Graph nodes must stay wrapped in `@traced(...)` so the UI trace and Prometheus metrics keep working.

## Project skills
See `.claude/skills/`: `run-rag-eval`, `add-agent-tool`, `add-guardrail`, `ingest-knowledge`.
