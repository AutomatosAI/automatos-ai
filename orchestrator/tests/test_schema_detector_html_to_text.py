"""GenericDataExtractor._html_to_text's script/style stripping (CodeQL
py/bad-tag-filter #5).

The old patterns (``</style>``, ``</script>``) only matched a closing tag
with nothing else before '>'. A browser's tokenizer also closes on
``</script foo>`` — anything after the tag name that is not a letter ends
it. Under-recognising the close let the script/style body's own source text
survive into this plain-text email preview.
"""
from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from modules.tools.formatting.schema_detector import GenericDataExtractor  # noqa: E402


def test_a_script_closed_with_trailing_junk_is_stripped():
    """Regression: </script bar> (space + other text before '>') used to
    survive the strip entirely, leaving the script's source in the preview."""
    html = '<p>Hello</p><script>alert(document.cookie)</script bar><p>World</p>'
    text = GenericDataExtractor._html_to_text(html)
    assert "alert" not in text
    assert "Hello" in text and "World" in text


def test_a_style_closed_with_trailing_junk_is_stripped():
    html = '<style>body{color:red}</style foo><p>Visible</p>'
    text = GenericDataExtractor._html_to_text(html)
    assert "color:red" not in text
    assert "Visible" in text


def test_an_ordinary_script_and_style_are_still_stripped():
    html = '<style>body{color:red}</style><script>alert(1)</script><p>Visible</p>'
    text = GenericDataExtractor._html_to_text(html)
    assert "color:red" not in text
    assert "alert" not in text
    assert "Visible" in text
