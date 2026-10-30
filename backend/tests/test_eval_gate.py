import json
import sys

from app.evaluation.evaluator import load_dataset, run_eval
from scripts.run_eval import main


def test_release_gate_passes_offline(client):
    report, passed = run_eval()
    assert passed, report["gate"]
    assert set(report["retrieval_comparison"]) == {"dense", "sparse", "hybrid"}


def test_dataset_is_read_as_utf8(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_bytes('{"question": "Is the price ₹500?"}\n'.encode())
    assert load_dataset(path) == [{"question": "Is the price ₹500?"}]


def test_reports_are_written_as_utf8(client, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_eval", "--out", str(tmp_path)])
    assert main() == 0
    json.loads((tmp_path / "eval_report.json").read_bytes().decode("utf-8"))
    assert "✅" in (tmp_path / "eval_report.md").read_bytes().decode("utf-8")
