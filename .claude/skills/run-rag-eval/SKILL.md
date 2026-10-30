---
name: run-rag-eval
description: Run the RAG evaluation suite, read the release-gate report and diagnose regressions in retrieval, generation, tool use or guardrails. Use after any change to prompts, retrieval, chunking, tools or guardrails, or when asked about answer quality.
---

# Run and interpret the RAG evaluation

1. From `backend/`, run the offline gate:
   `LLM_PROVIDER=offline QDRANT_PATH=:memory: python -m scripts.run_eval --out reports`
   If Ollama is running and the change affects generation, also run
   `python -m scripts.run_eval --reindex --judge --thresholds data/eval/thresholds.llm.yaml`.
2. Read `reports/eval_report.md`. The process exits 1 when any gate metric is below threshold.
3. Diagnose by which metric failed:
   - `retrieval_hit_rate` / `mrr` / `context_recall` down → retrieval problem. Compare the
     "dense vs BM25 vs hybrid" table. Inspect a failing question with
     `curl -s localhost:8000/api/search -d '{"query":"...","mode":"hybrid"}' -H 'content-type: application/json'`.
     Check chunk size/overlap in `config.py` and the synonym map in `textutils.py`.
   - `answer_correctness` down while retrieval is fine → generation problem: prompt, context
     ordering (`retrieval/context.py`), or `max_context_tokens` too small.
   - `faithfulness` down → the model is adding unsupported claims; tighten prompt rule 1.
   - `tool_accuracy` down → tool docstrings are ambiguous or a tool was renamed.
   - `guardrail_accuracy` down → see `guardrails/rails.py`.
4. Look at "Examples to review" in the report for the specific failing questions.
5. Report to the user: the gate result, metric deltas versus the previous report if one exists,
   the root cause and the fix. Never lower a threshold without the user's agreement.
