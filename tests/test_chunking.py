import pytest

from rag_eval.chunking import (
    Document,
    chunk_documents,
    chunk_fixed,
    chunk_heading,
    chunk_recursive,
    count_tokens,
)

TEXT = "# Title\n\n## One\nAlpha beta gamma. Delta epsilon zeta.\n\n## Two\nEta theta iota kappa lambda mu."


def test_fixed_window_size_and_overlap():
    doc = Document("d", " ".join(f"w{i}" for i in range(100)))
    chunks = chunk_fixed(doc, size=30, overlap=10)
    assert all(count_tokens(c.text) <= 30 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-10:] == second[:10]
    assert {w for c in chunks for w in c.text.split()} == {f"w{i}" for i in range(100)}


def test_fixed_rejects_bad_parameters():
    with pytest.raises(ValueError):
        chunk_fixed(Document("d", "a b c"), size=10, overlap=10)


def test_heading_keeps_section_metadata_and_title():
    chunks = chunk_heading(Document("d", TEXT))
    assert [c.section for c in chunks] == ["One", "Two"]
    assert chunks[0].text.startswith("Title - One")
    assert chunks[0].metadata["title"] == "Title"


def test_recursive_respects_limit_and_loses_nothing():
    doc = Document("d", TEXT)
    chunks = chunk_recursive(doc, size=8)
    assert all(count_tokens(c.text) <= 8 for c in chunks)
    original = {w for w in doc.text.replace("#", " ").split()}
    assert original <= {w for c in chunks for w in c.text.split()}


def test_unknown_strategy():
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        chunk_documents([Document("d", "x")], "nope")


def test_chunk_ids_are_unique():
    chunks = chunk_documents([Document("a", TEXT), Document("b", TEXT)], "heading")
    assert len({c.chunk_id for c in chunks}) == len(chunks)
