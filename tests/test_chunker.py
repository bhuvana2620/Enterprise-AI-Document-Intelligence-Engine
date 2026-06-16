"""
Unit tests for src.ingestion.chunker

These run with no external dependencies (no API keys, no network,
no Pinecone) and are safe to run in CI on every push.
"""

import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingestion.chunker import chunk_text, normalize_text, split_with_overlap


def test_normalize_text_collapses_whitespace():
    raw = "Hello    world\n\nthis   is\ta test"
    assert normalize_text(raw) == "Hello world this is a test"


def test_normalize_text_handles_empty_string():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_split_with_overlap_respects_chunk_size():
    text = "a" * 1000
    chunks = split_with_overlap(text, chunk_size=200, chunk_overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1


def test_split_with_overlap_clamps_excessive_overlap():
    # chunker.py silently clamps overlap >= chunk_size down to chunk_size // 5
    # rather than raising, so chunking never gets stuck in an infinite loop.
    text = "a" * 1000
    chunks = split_with_overlap(text, chunk_size=100, chunk_overlap=150)
    assert len(chunks) > 0
    assert all(len(c) <= 100 for c in chunks)


def test_split_with_overlap_rejects_zero_chunk_size():
    try:
        split_with_overlap("some text", chunk_size=0, chunk_overlap=0)
        assert False, "Expected ValueError for chunk_size <= 0"
    except ValueError:
        pass


def test_chunk_text_returns_structured_dicts():
    text = "This is a sentence. " * 100
    chunks = chunk_text(text, source_name="test.txt", category="test")

    assert len(chunks) > 0

    for i, chunk in enumerate(chunks):
        assert "text" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["source"] == "test.txt"
        assert chunk["metadata"]["category"] == "test"
        assert chunk["metadata"]["chunk_index"] == i


def test_chunk_text_empty_input_returns_empty_list():
    assert chunk_text("", source_name="empty.txt", category="test") == []
    assert chunk_text("   ", source_name="empty.txt", category="test") == []


def test_chunk_text_no_empty_chunks_in_output():
    text = "word " * 500
    chunks = chunk_text(text, source_name="test.txt", category="test")
    assert all(chunk["text"].strip() != "" for chunk in chunks)


if __name__ == "__main__":
    import inspect

    current_module = sys.modules[__name__]
    test_functions = [
        obj for name, obj in inspect.getmembers(current_module)
        if name.startswith("test_") and inspect.isfunction(obj)
    ]

    passed = 0
    failed = 0

    for test_fn in test_functions:
        try:
            test_fn()
            print(f"PASS: {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test_fn.__name__} - {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
