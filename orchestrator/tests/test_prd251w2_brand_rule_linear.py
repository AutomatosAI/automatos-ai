"""PRD-251 — the brand rule reads hostile template CSS in linear time (#852, CodeQL).

The rule scans CSS a request can carry, so every scan must stay linear. Four once
had two ways to match the same text and took quadratic time on the cases below: a
font shorthand's number, the family list after its size, the spaces and bare
argument of a ``url()``, and a ``/* … */`` comment. Each case now finishes in
milliseconds; the bound is loose so a slow runner never trips it.

The other tests pin that the rewrites read CSS as before: an empty ``url()``, a
font shorthand's family, a closed comment and an unclosed one.
"""

from __future__ import annotations

import time

import pytest

from core.social_brand_rule import brand_literals

REPEAT = 20_000
BOUND_SECONDS = 2.0

HOSTILE = {
    "font-number": "a{font: 1" + "0" * REPEAT + "}",
    "font-family-list": "a{font: small " + "  " * REPEAT + "x\ny}",
    "url-spaces": "a{background: url(" + " " * REPEAT + "x y}",
    "url-repeated": "a{background: " + "url(" * REPEAT + "}",
    "unclosed-comments": "/*" + "a/*" * REPEAT,
}


@pytest.mark.parametrize("css", list(HOSTILE.values()), ids=list(HOSTILE))
def test_hostile_css_is_read_in_linear_time(css):
    started = time.perf_counter()
    brand_literals("", css)
    assert time.perf_counter() - started < BOUND_SECONDS


def test_an_empty_url_is_no_finding_and_no_error():
    assert brand_literals("", "a{background: url(); color: var(--brand-primary)}") == []


def test_a_font_shorthand_still_names_its_family():
    found = brand_literals("", "h1{font: 500 42px/1.2 Comic Sans MS, sans-serif}")
    assert any("Comic Sans MS" in finding for _, finding in found)


def test_a_closed_comment_hides_its_colour_and_an_unclosed_one_does_not_hide_the_rule_before_it():
    assert brand_literals("", "a{color: var(--brand-primary) /* #ff0000 */}") == []
    found = brand_literals("", "a{color: #ff0000} /* never closed")
    assert any("#ff0000" in finding for _, finding in found)
