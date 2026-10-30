"""CLI: run the evaluation suite and enforce the release gate.

    python -m scripts.run_eval                 # heuristic metrics (fast, used in CI)
    python -m scripts.run_eval --judge         # LLM-as-judge with the local Ollama model
    python -m scripts.run_eval --thresholds data/eval/thresholds.llm.yaml --judge

Exit code 1 when any metric is below its threshold, so CI blocks the merge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.deps import get_store  # noqa: E402
from app.evaluation.evaluator import run_eval, to_markdown  # noqa: E402
from app.ingestion.service import ingest_directory  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--thresholds", type=Path)
    ap.add_argument("--judge", action="store_true", help="use the LLM as a judge for faithfulness/relevance")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--reindex", action="store_true", help="rebuild the index from data/knowledge_base first")
    args = ap.parse_args()

    store = get_store()
    if args.reindex or store.count() == 0:
        store.reset()
        ingest_directory(store, get_settings().docs_dir)

    report, passed = run_eval(args.dataset, args.thresholds, args.judge)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(report)
    (args.out / "eval_report.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # the report has emoji; Windows consoles default to cp1252
    print(md)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
