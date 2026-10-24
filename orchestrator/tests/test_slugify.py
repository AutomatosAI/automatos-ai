"""Unit tests for the shared ``slugify`` helper (#1074).

The two private copies this replaces were untested, so the edge cases here also
pin the behaviour the existing callers relied on.
"""

from core.utils.slug import slugify


def test_empty_title_uses_the_fallback():
    assert slugify("", fallback="post") == "post"
    assert slugify(None, fallback="post") == "post"


def test_punctuation_only_uses_the_fallback():
    assert slugify("!!!", fallback="card-7") == "card-7"
    assert slugify("---", fallback="report") == "report"


def test_separators_collapse_and_are_trimmed():
    assert slugify("Hello, World! 2026", fallback="post") == "hello-world-2026"
    assert slugify("--Hello--", fallback="post") == "hello"


def test_long_title_is_cut_and_never_ends_with_a_separator():
    assert slugify("a" * 200, max_len=60, fallback="post") == "a" * 60

    cut = slugify("word " * 40, max_len=10, fallback="post")
    assert len(cut) <= 10
    assert not cut.endswith("-")


def test_unicode_is_a_separator_like_it_was_before():
    assert slugify("Über café", fallback="post") == "ber-caf"
    assert slugify("中文标题", fallback="post") == "post"


def test_current_outputs_are_unchanged():
    # Filenames the two modules generate today must keep generating the same names.
    assert slugify("Q4 launch: what shipped?", fallback="post") == "q4-launch-what-shipped"
    assert slugify("Weekly digest", fallback="post") == "weekly-digest"
    assert slugify("Board card 42", fallback="card-42") == "board-card-42"
