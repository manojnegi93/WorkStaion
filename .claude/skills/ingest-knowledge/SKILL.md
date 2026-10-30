---
name: ingest-knowledge
description: Add or update documents in the assistant's knowledge base and verify they are retrievable. Use when given new policies, manuals or FAQs to load.
---

# Ingest knowledge

1. Prefer Markdown with `#` title and `##` sections. The chunker uses headings as the citation
   label, so give each section a specific, descriptive heading. `.txt` and `.pdf` also work.
2. Put permanent documents in `backend/data/knowledge_base/`. For a running server, upload with
   `curl -F files=@path/to/doc.md localhost:8000/api/documents` or the Knowledge base tab in the UI.
   Re-uploading a file with the same name replaces its old passages.
3. Verify retrieval for two or three real questions:
   `curl -s localhost:8000/api/search -H 'content-type: application/json' -d '{"query":"...","k":5}'`
   The new file should be in the top results with both `dense_rank` and `sparse_rank` set.
4. Add one golden example per document to `backend/data/eval/golden.jsonl` with
   `expected_sources` and the key fact in `must_include`, then run `make eval`.
5. If a document uses vocabulary customers don't (e.g. "fragrance" vs "scent"), extend `SYNONYMS` in
   `backend/app/textutils.py`.
