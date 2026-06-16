"""
Unit tests for src.ingestion.text_cleaner

No external dependencies — safe for CI.
"""

import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingestion.text_cleaner import clean_text


def test_collapses_multiple_spaces():
    assert clean_text("Hello     world") == "Hello world"


def test_collapses_repeated_newlines():
    result = clean_text("line one\n\n\n\nline two")
    assert "\n\n" not in result


def test_replaces_tabs_with_space():
    result = clean_text("a\t\tb")
    assert "\t" not in result


def test_replaces_nbsp():
    result = clean_text("a\xa0b")
    assert "\xa0" not in result


def test_strips_leading_and_trailing_whitespace():
    assert clean_text("   padded text   ") == "padded text"


def test_handles_empty_string():
    assert clean_text("") == ""


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
