"""F382 (night 11, 7 Oct), B6: a card is fitted to its words, not shrunk to half by its own glyphs.

The Carousel's cover and closing slides came out at about half scale, the Stats
card's 16:9 content in a thin band, the Definition card's body at ~14 px. The paper
cards set their display type at ``line-height: 0.95`` (the Carousel's close 0.98,
the Fact card's figure and the Carousel's numbers 0.9), tighter than their glyphs
reach (a face's ascent and descent: Geist 1.30em, Inter 1.21em). Chrome counts that
reach in a box's scroll height, so the fit (``scrollHeight > clientHeight + 1``)
read every card as overflowing and stepped it down to its 0.5 floor. The photo cards
got ``padding-bottom: 0.08em`` for it (cdf76eca5), which covers Liberation Serif but
not Geist or Inter. Pins, on every seeded image template:

* display type tighter than 1 carries the room its glyphs need below its last line,
  ``calc((var(--ink-reach) - <line-height>em) / 2)``, with ``--ink-reach`` at least
  Geist's reach;
* the fit allows the rest of a glyph's reach (lines of 1 and more) as ink, not
  overflow (``INK_SLACK`` of the largest type in the box);
* the Carousel lays a slide the timeline hides out unseen for its fit, so every slide
  fits when it is the one shot, not only on font events.
"""
from __future__ import annotations

import re

import pytest

from core.social_templates import SOCIAL_IMAGE
from modules.documents.social_starters import social_starters

# The tallest reach among the faces a card may print in (Geist's ascent + descent, 1300/1000).
TALLEST_REACH = 1.30
REACH = re.compile(r"--ink-reach: (\d+(?:\.\d+)?)em;")
SLACK = re.compile(r"const INK_SLACK = (\d+(?:\.\d+)?);")
# The quote mark is a pseudo-element drawn into a box of its own fixed height: no fit reads it.
EXEMPT = {".quote::before"}


def _images():
    return social_starters(SOCIAL_IMAGE)


def _css(html: str) -> str:
    css = "".join(re.findall(r"<style>(.*?)</style>", html, re.S))
    return re.sub(r"/\*.*?\*/", "", re.sub(r"\{\{[^{}]*\}\}", "0", css), flags=re.S)


def _tight_rules(html: str):
    """``(selector, line-height)`` of every rule setting a line height under 1."""
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _css(html)):
        for value in re.findall(r"line-height:\s*(0?\.\d+|0)\s*;", body):
            yield selector.strip(), value, body


@pytest.mark.parametrize("slug", [s["slug"] for s in social_starters(SOCIAL_IMAGE)])
def test_tight_display_type_has_room_for_its_glyphs(slug):
    html = next(s for s in _images() if s["slug"] == slug)["blocks"]["html"]
    reach = REACH.search(html)
    assert reach and float(reach.group(1)) >= TALLEST_REACH, slug
    for selector, value, body in _tight_rules(html):
        if selector in EXEMPT:
            continue
        room = f"padding-bottom: calc((var(--ink-reach) - {float(value):g}em) / 2);"
        assert room in body, (slug, selector, value)


@pytest.mark.parametrize("slug", [s["slug"] for s in social_starters(SOCIAL_IMAGE)])
def test_the_fit_reads_a_glyphs_reach_as_ink_not_overflow(slug):
    html = next(s for s in _images() if s["slug"] == slug)["blocks"]["html"]
    reach, slack = float(REACH.search(html).group(1)), SLACK.search(html)
    assert slack, slug
    # A line of 1 or more sets glyphs reaching (reach - 1) / 2 below it at most: inside the slack.
    assert (reach - 1) / 2 <= float(slack.group(1)), slug
    assert "el.scrollHeight > el.clientHeight + 1 + inkSlack(el)" in html, slug


def test_the_paper_cards_headlines_and_the_carousels_close_and_numbers_are_covered():
    by_slug = {s["slug"]: s["blocks"]["html"] for s in _images()}
    for slug in ("title-card", "quote-card", "stats-card", "definition-card", "announcement-card", "carousel", "fact-card"):
        assert "padding-bottom: calc((var(--ink-reach) - 0.95em) / 2);" in by_slug[slug], slug
    assert "padding-bottom: calc((var(--ink-reach) - 0.98em) / 2);" in by_slug["infographic"]
    carousel = dict((sel, body) for sel, _v, body in _tight_rules(by_slug["carousel"]))
    assert "0.98em" in carousel[".closing-title"] and "0.9em" in carousel[".point-no"]
    fact = dict((sel, body) for sel, _v, body in _tight_rules(by_slug["fact-card"]))
    assert "0.9em" in fact[".fact-value"]


def test_every_carousel_slide_is_fitted_when_it_is_shot():
    html = next(s for s in _images() if s["slug"] == "carousel")["blocks"]["html"]
    assert "shownFor(page, () => fitShown(page));" in html
    # A hidden slide is laid out unseen for its fit, then put back exactly as it was.
    assert 'slide.style.setProperty("display", "block", "important");' in html
    assert 'slide.style.setProperty("visibility", "hidden", "important");' in html
    assert 'slide.setAttribute("style", style);' in html and 'slide.removeAttribute("style");' in html
