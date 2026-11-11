"""Document chunking strategies.

Three strategies are implemented so their effect on retrieval quality can be
measured instead of guessed:

* ``fixed``     - sliding window over tokens with overlap
* ``recursive`` - split on paragraphs, then sentences, then words, merging up to a size limit
* ``heading``   - one chunk per markdown section, with the document title and
                  section heading kept as metadata and prepended to the text

"Tokens" are whitespace-separated words, a deliberate approximation that keeps
the project dependency-free. Swap ``count_tokens`` for ``tiktoken`` if you need
exact model token counts.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str = ""
    metadata: dict = field(default_factory=dict, compare=False)


def count_tokens(text: str) -> int:
    return len(text.split())


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def chunk_fixed(doc: Document, size: int = 60, overlap: int = 15) -> list[Chunk]:
    """Sliding window of ``size`` tokens advancing by ``size - overlap``."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = doc.text.split()
    step = size - overlap
    chunks: list[Chunk] = []
    for n, start in enumerate(range(0, max(len(words), 1), step)):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(Chunk(f"{doc.doc_id}#f{n}", doc.doc_id, " ".join(window), metadata={"strategy": "fixed"}))
        if start + size >= len(words):
            break
    return chunks


def _split_units(text: str, limit: int) -> list[str]:
    """Break ``text`` into pieces of at most ``limit`` tokens, preferring natural boundaries."""
    if count_tokens(text) <= limit:
        return [text.strip()] if text.strip() else []
    for splitter in (lambda t: re.split(r"\n\s*\n", t), lambda t: _SENTENCE.split(t)):
        parts = [p.strip() for p in splitter(text) if p.strip()]
        if len(parts) > 1:
            out: list[str] = []
            for p in parts:
                out.extend(_split_units(p, limit))
            return out
    words = text.split()
    return [" ".join(words[i : i + limit]) for i in range(0, len(words), limit)]


def chunk_recursive(doc: Document, size: int = 80) -> list[Chunk]:
    """Split on paragraphs -> sentences -> words, then merge neighbours up to ``size`` tokens."""
    units = _split_units(doc.text, size)
    merged: list[str] = []
    current = ""
    for u in units:
        candidate = f"{current}\n{u}".strip() if current else u
        if count_tokens(candidate) <= size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = u
    if current:
        merged.append(current)
    return [
        Chunk(f"{doc.doc_id}#r{n}", doc.doc_id, text, metadata={"strategy": "recursive"})
        for n, text in enumerate(merged)
    ]


def chunk_heading(doc: Document) -> list[Chunk]:
    """One chunk per markdown section; title and heading are preserved as metadata and context."""
    title = _title(doc.text, doc.doc_id)
    matches = list(_HEADING.finditer(doc.text))
    chunks: list[Chunk] = []
    for i, m in enumerate(matches):
        level, heading = len(m.group(1)), m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc.text)
        body = doc.text[m.end() : end].strip()
        if level == 1 or not body:
            continue  # the document title carries no content of its own
        chunks.append(
            Chunk(
                f"{doc.doc_id}#h{len(chunks)}",
                doc.doc_id,
                f"{title} - {heading}\n{body}",
                section=heading,
                metadata={"strategy": "heading", "title": title},
            )
        )
    return chunks or [Chunk(f"{doc.doc_id}#h0", doc.doc_id, doc.text.strip(), metadata={"strategy": "heading"})]


STRATEGIES: dict[str, Callable[[Document], list[Chunk]]] = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "heading": chunk_heading,
}


def chunk_documents(docs: list[Document], strategy: str) -> list[Chunk]:
    try:
        fn = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(f"unknown chunking strategy {strategy!r}; choose from {sorted(STRATEGIES)}") from None
    return [c for d in docs for c in fn(d)]
