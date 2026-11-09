"""fill_placeholders' script-boundary detection (CodeQL py/bad-tag-filter #247).

``_SCRIPT_BODY`` used to require pure whitespace between "script" and ">" in
the closing tag (``</script\\s*>``). A browser's tokenizer treats
``</script FOO>`` as the real close too — anything after "script" that is
not a letter ends the tag name, attributes or not. Under-recognising the
close meant the detected "in code" span could run past the real script and
swallow template text that should be filled freely; broadening it to
``</script\\b[^>]*>`` matches every real closing form.
"""
from __future__ import annotations

from media_render.composition import fill_placeholders


def test_a_closing_tag_with_a_space_ends_the_script():
    text = '<script>{{num}}</script ><p>{{word}}</p>'
    result = fill_placeholders(text, {"num": 3, "word": "hello"}, "t")
    assert result == '<script>3</script ><p>hello</p>'


def test_a_text_variable_inside_a_malformed_closing_tags_script_is_still_refused():
    """The exact CodeQL witness shape: a tab/newline then other text before '>'.

    Regression: the old pattern (``</script\\s*>``) never matched this close,
    so ``_code_spans`` found no span for this script at all — a text
    variable placed where ``{{word}}`` is below would have been treated as
    ordinary markup and HTML-escaped (safe for HTML, not for JS) instead of
    being refused as "text inside a script"."""
    import pytest

    from media_render.validate import BundleError

    text = '<script>{{word}}</script\t\n bar>'
    with pytest.raises(BundleError, match="inside a .script. or an event handler"):
        fill_placeholders(text, {"word": "not a number"}, "t")


def test_inside_a_script_only_a_number_or_boolean_is_filled():
    import pytest

    with pytest.raises(Exception):
        fill_placeholders('<script>{{word}}</script>', {"word": "not a number"}, "t")


def test_a_normal_closing_tag_still_works():
    text = '<script>{{num}}</script><p>{{word}}</p>'
    result = fill_placeholders(text, {"num": 3, "word": "hello"}, "t")
    assert result == '<script>3</script><p>hello</p>'


def test_a_false_closing_tag_inside_the_script_does_not_end_it():
    """CodeRabbit on #1448: with ``</script\\b``, "</script-x>" (which a browser does not
    read as a close: the tag name runs on through the hyphen) ended the protected span
    early, and a string value then filled a template literal later in the same script,
    where ``${...}`` runs. The span now reaches the real close, so the value is refused."""
    import pytest

    from media_render.validate import BundleError

    text = '<script>const s="</script-x>"; const x=`{{word}}`;</script>'
    with pytest.raises(BundleError, match="inside a .script. or an event handler"):
        fill_placeholders(text, {"word": "${alert(1)}"}, "t")
