"""Document loading and structure-aware chunking.

* Markdown is split on headings first, so each chunk keeps its heading path
  (``section``) — this becomes part of the citation and of the embedded text.
* Long sections are split recursively (paragraph -> line -> sentence -> word)
  with a character overlap so facts spanning a boundary are not lost.
* Each chunk gets a *contextual header* ("Document > Section") prepended to the
  text that is embedded/indexed. This cheap form of contextual retrieval
  noticeably improves recall for short, ambiguous chunks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str
    section: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def indexed_text(self) -> str:
        header = f"{self.title} > {self.section}" if self.section else self.title
        return f"{header}\n{self.text}"

    def to_payload(self) -> dict:
        return {
            "chunk_id": self.id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _title_from(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else fallback.replace("-", " ").replace("_", " ").title()


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading_path, body) pairs. Content before the first heading has path ''."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        level, heading = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            path = " > ".join(h for lvl, h in stack if lvl > 1) or heading
            sections.append((path, body))
    return sections


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, size: int, overlap: int, seps: list[str] | None = None) -> list[str]:
    seps = _SEPARATORS if seps is None else seps
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size - overlap)]
    sep, rest = seps[0], seps[1:]
    pieces = [p for p in text.split(sep) if p.strip()]
    if len(pieces) == 1:
        return _recursive_split(text, size, overlap, rest)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(piece) > size:
            chunks.extend(_recursive_split(piece, size, overlap, rest))
            current = ""
        else:
            tail = current[-overlap:] if overlap and current else ""
            # start the overlap on a word boundary
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            current = f"{tail}{sep}{piece}" if tail else piece
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def chunk_document(text: str, source: str, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    title = _title_from(text, Path(source).stem)
    chunks: list[Chunk] = []
    for section, body in _split_sections(text):
        for piece in _recursive_split(body, chunk_size, overlap):
            idx = len(chunks)
            cid = hashlib.sha1(f"{source}::{idx}::{piece[:64]}".encode()).hexdigest()[:16]
            chunks.append(Chunk(cid, piece, source, title, section, idx))
    return chunks


def chunk_file(path: Path, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    return chunk_document(load_text(path), path.name, chunk_size, overlap)
