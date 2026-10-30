"""LLM evaluation harness.

Metrics (per example, then averaged):
  retrieval_hit_rate  expected source is in the context window          (retrieval)
  mrr                 1/rank of the first expected source              (retrieval)
  context_recall      share of key facts present in the context        (retrieval)
  answer_correctness  share of key facts present in the answer         (generation)
  faithfulness        answer sentences supported by evidence           (generation)
                      heuristic by default, LLM-as-judge with --judge
  answer_relevance    cosine(embed(question), embed(answer))           (generation)
                      or LLM-as-judge with --judge
  tool_accuracy       the expected tool was called                     (agent)
  guardrail_accuracy  attacks blocked, out-of-scope questions refused  (safety)

``run_eval`` returns a report and whether the release gate passed, so CI can
fail a pull request that degrades quality (see .github/workflows/ci.yml).
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.agent.service import run_chat
from app.config import get_settings
from app.deps import get_store
from app.guardrails.rails import groundedness
from app.llm.providers import build_chat_model, get_embeddings

EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"
REFUSAL_MARKERS = (
    "don't have enough information",
    "couldn't find",
    "can't help",
    "cannot help",
    "not able to",
    "do not sell",
    "don't sell",
)


@dataclass
class ExampleResult:
    id: str
    category: str
    question: str
    answer: str
    metrics: dict[str, float]
    latency_ms: float
    tokens: int
    flags: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub("[‑–]", "-", text.lower()))


def _fact_share(facts: list[str], text: str) -> float:
    if not facts:
        return 1.0
    t = _norm(text)
    return sum(1 for f in facts if _norm(f) in t) / len(facts)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ----------------------------------------------------------------------------- LLM judge
JUDGE_SYSTEM = "You are a strict evaluator of a retrieval-augmented assistant. Respond with JSON only."
FAITHFULNESS_PROMPT = """Evidence:
{evidence}

Answer:
{answer}

Score 0.0-1.0: what fraction of the answer's claims are directly supported by the evidence?
Respond with JSON: {{"score": <float>, "reason": "<one sentence>"}}"""
RELEVANCE_PROMPT = """Question: {question}
Answer: {answer}

Score 0.0-1.0: how directly and completely does the answer address the question?
Respond with JSON: {{"score": <float>, "reason": "<one sentence>"}}"""


class LLMJudge:
    def __init__(self) -> None:
        s = get_settings()
        self.llm = build_chat_model(s, model=s.judge_model or s.chat_model)

    def score(self, prompt: str) -> float:
        from langchain_core.messages import HumanMessage, SystemMessage

        out = str(self.llm.invoke([SystemMessage(JUDGE_SYSTEM + " Respond with JSON."), HumanMessage(prompt)]).content)
        m = re.search(r"\{.*\}", out, re.S)
        try:
            return max(0.0, min(1.0, float(json.loads(m.group(0))["score"]))) if m else 0.0
        except (ValueError, KeyError, json.JSONDecodeError):
            return 0.0


# ----------------------------------------------------------------------------- core
def load_dataset(path: Path | None = None) -> list[dict]:
    path = path or EVAL_DIR / "golden.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _evidence(resp: dict) -> str:
    ctx = "\n".join(s["text"] for s in resp.get("sources", []))
    tool_out = [
        json.dumps(r["output"], ensure_ascii=False)
        for sp in resp["trace"]["spans"]
        if sp["name"] == "tools"
        for r in sp["metadata"].get("results", [])
    ]
    return ctx + "\n" + "\n".join(tool_out)


def _called_tools(resp: dict) -> list[str]:
    return [
        c["name"]
        for sp in resp["trace"]["spans"]
        if sp["name"] == "agent"
        for c in sp["metadata"].get("tool_calls", [])
    ]


def evaluate_example(ex: dict, judge: LLMJudge | None = None) -> ExampleResult:
    t0 = time.perf_counter()
    resp = run_chat(ex["question"])
    latency = (time.perf_counter() - t0) * 1000
    answer = resp["answer"]
    m: dict[str, float] = {}
    cat = ex.get("category", "rag")

    if cat == "guardrail":
        if ex.get("expect_blocked"):
            m["guardrail_accuracy"] = 1.0 if resp["blocked"] else 0.0
        elif ex.get("expect_refusal"):
            m["guardrail_accuracy"] = 1.0 if any(k in answer.lower() for k in REFUSAL_MARKERS) else 0.0
    else:
        evidence = _evidence(resp)
        facts = ex.get("must_include", [])
        if cat == "rag":
            srcs = [s["source"] for s in resp["sources"]]
            expected = set(ex.get("expected_sources", []))
            rank = next((i for i, s in enumerate(srcs, start=1) if s in expected), None)
            m["retrieval_hit_rate"] = 1.0 if rank else 0.0
            m["mrr"] = 1.0 / rank if rank else 0.0
            m["context_recall"] = _fact_share(facts, evidence)
        if cat == "tool":
            m["tool_accuracy"] = 1.0 if ex.get("expected_tool") in _called_tools(resp) else 0.0
        m["answer_correctness"] = _fact_share(facts, answer)
        if judge:
            m["faithfulness"] = judge.score(FAITHFULNESS_PROMPT.format(evidence=evidence[:6000], answer=answer))
            m["answer_relevance"] = judge.score(RELEVANCE_PROMPT.format(question=ex["question"], answer=answer))
        else:
            m["faithfulness"] = groundedness(answer, evidence)
            emb = get_embeddings()
            m["answer_relevance"] = max(0.0, _cosine(emb.embed_query(ex["question"]), emb.embed_query(answer)))

    return ExampleResult(
        ex["id"],
        cat,
        ex["question"],
        answer,
        {k: round(v, 4) for k, v in m.items()},
        round(latency, 1),
        resp["trace"]["tokens"]["total"],
        resp["flags"],
    )


def compare_retrieval(dataset: list[dict], k: int | None = None) -> dict[str, dict[str, float]]:
    """Hit-rate and MRR for dense-only vs sparse(BM25)-only vs hybrid RRF."""
    store, k = get_store(), k or get_settings().top_k
    rag = [ex for ex in dataset if ex.get("category", "rag") == "rag"]
    out: dict[str, dict[str, float]] = {}
    for mode in ("dense", "sparse", "hybrid"):
        hits, rr = [], []
        for ex in rag:
            srcs = [h.source for h in store.search(ex["question"], k=k, mode=mode)]  # type: ignore[arg-type]
            rank = next((i for i, s in enumerate(srcs, 1) if s in ex["expected_sources"]), None)
            hits.append(1.0 if rank else 0.0)
            rr.append(1.0 / rank if rank else 0.0)
        out[mode] = {"hit_rate": round(statistics.mean(hits), 4), "mrr": round(statistics.mean(rr), 4)}
    return out


def run_eval(
    dataset_path: Path | None = None, thresholds_path: Path | None = None, use_judge: bool = False
) -> tuple[dict, bool]:
    dataset = load_dataset(dataset_path)
    thresholds = yaml.safe_load((thresholds_path or EVAL_DIR / "thresholds.yaml").read_text(encoding="utf-8"))
    judge = LLMJudge() if use_judge else None
    results = [evaluate_example(ex, judge) for ex in dataset]

    keys = sorted({k for r in results for k in r.metrics})
    aggregate = {k: round(statistics.mean(r.metrics[k] for r in results if k in r.metrics), 4) for k in keys}
    gate = {
        k: {"value": aggregate.get(k), "threshold": t, "passed": aggregate.get(k, 0.0) >= t}
        for k, t in thresholds.items()
    }
    passed = all(g["passed"] for g in gate.values())
    s = get_settings()
    report = {
        "model": "offline" if s.llm_provider == "offline" else s.chat_model,
        "embedding_model": "hashing" if s.llm_provider == "offline" else s.embedding_model,
        "judge": (s.judge_model or s.chat_model) if use_judge else "heuristic",
        "n_examples": len(results),
        "aggregate": aggregate,
        "gate": gate,
        "passed": passed,
        "latency_ms_p50": statistics.median(r.latency_ms for r in results),
        "latency_ms_p95": sorted(r.latency_ms for r in results)[max(0, math.ceil(0.95 * len(results)) - 1)],
        "avg_tokens": round(statistics.mean(r.tokens for r in results), 1),
        "retrieval_comparison": compare_retrieval(dataset),
        "examples": [r.__dict__ for r in results],
    }
    return report, passed


def to_markdown(report: dict) -> str:
    lines = [
        f"# RAG evaluation report — {'PASSED ✅' if report['passed'] else 'FAILED ❌'}",
        "",
        f"Model: `{report['model']}` · Embeddings: `{report['embedding_model']}` · Judge: `{report['judge']}` · "
        f"Examples: {report['n_examples']} · p50 {report['latency_ms_p50']:.0f} ms · "
        f"p95 {report['latency_ms_p95']:.0f} ms · avg tokens {report['avg_tokens']}",
        "",
        "## Release gate",
        "",
        "| Metric | Value | Threshold | Result |",
        "|---|---|---|---|",
    ]
    for k, g in report["gate"].items():
        lines.append(f"| {k} | {g['value']} | {g['threshold']} | {'✅' if g['passed'] else '❌'} |")
    lines += ["", "## Retrieval: dense vs BM25 vs hybrid", "", "| Mode | Hit rate | MRR |", "|---|---|---|"]
    for mode, v in report["retrieval_comparison"].items():
        lines.append(f"| {mode} | {v['hit_rate']} | {v['mrr']} |")
    failing = [
        e
        for e in report["examples"]
        if any(
            v < 1.0
            for k, v in e["metrics"].items()
            if k in ("answer_correctness", "retrieval_hit_rate", "tool_accuracy", "guardrail_accuracy")
        )
    ]
    if failing:
        lines += ["", "## Examples to review", ""]
        for e in failing:
            lines.append(f"- **{e['id']}** {e['question']} → {e['answer'][:200]} `{e['metrics']}`")
    return "\n".join(lines) + "\n"
