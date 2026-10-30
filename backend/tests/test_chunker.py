from app.ingestion.chunker import chunk_document

DOC = """# Policy

## Wicks
Trim the wick before every burn.

## Safety
### Flames
Never leave a flame unattended.
"""


def test_sections_become_heading_paths():
    chunks = chunk_document(DOC, "policy.md")
    assert [c.section for c in chunks] == ["Wicks", "Safety > Flames"]
    assert all(c.title == "Policy" for c in chunks)
    assert chunks[0].indexed_text.startswith("Policy > Wicks")


def test_long_sections_split_with_overlap_and_stable_ids():
    body = "# T\n\n## S\n" + " ".join(f"Sentence number {i} about candles." for i in range(200))
    a = chunk_document(body, "t.md", chunk_size=300, overlap=60)
    b = chunk_document(body, "t.md", chunk_size=300, overlap=60)
    assert len(a) > 5
    assert all(len(c.text) <= 300 for c in a)
    assert [c.id for c in a] == [c.id for c in b]
