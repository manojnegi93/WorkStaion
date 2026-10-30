---
name: add-guardrail
description: Add or tune an input or output guardrail (prompt injection, PII, topic, groundedness, citation rules) with tests. Use when asked to block, redact or flag a kind of content.
---

# Add a guardrail

1. Decide where it belongs in `backend/app/guardrails/rails.py`:
   - Input rail (`check_input`): blocks or rewrites the question before any model call.
     Return `GuardrailResult(False, <user-facing message>, ["<flag>"])` to block.
   - Output rail (`check_output`): validates the answer against the evidence.
2. Use a short snake_case flag name. Flags flow automatically to the API response,
   the Prometheus counter `rag_guardrail_flags_total` and the UI.
3. Add a plain-language description of the flag to `FLAG_TEXT` in
   `frontend/src/components/Answer.tsx`. Write it as what happened, e.g. "Hid personal data".
4. Add unit tests in `backend/tests/test_guardrails.py`: at least one case that must trigger and
   one realistic case that must not (false positives hurt staff trust).
5. Add a `"category": "guardrail"` example to `golden.jsonl` with `expect_blocked` or `expect_refusal`.
6. Run `make test` and `make eval`.
