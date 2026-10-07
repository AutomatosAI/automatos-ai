"""F382 (night 11, 7 Oct), B6: a card's body text never shrinks below a readable floor.

The Definition card's body came out at ~14 px (29 design px at the fit's 0.5 floor),
and smaller after "make it bigger"; at 16:9 the Fact card's statement and source and
the Stats card's labels were "microscopic" (``--u`` is the short side over 1080, so a
1600x900 card sets its type at 0.83 of a feed post's while it is shown as wide). Every
paper card now holds two floors: ``--body-floor`` (body text) and ``--label-floor``
(the small print), each at least a share of the feed post's size and of the card's own
width. The fit still shrinks the display type, so the headline gives way first. Pins:
the floors and their sizes at every size a card declares, and the elements that hold
them.
"""
from __future__ import annotations

import re

import pytest

from core.social_templates import parse_size
from modules.documents.social_starters import social_starters

FLOOR = re.compile(r"--(body|label)-floor: max\(calc\((\d+) \* var\(--u\)\), calc\(\{\{ size\.width \}\} \* (0\.\d+)px\)\);")
# The least each floor may be: in a feed post's design pixels, and as a share of the card's width.
LEAST = {"body": (24, 0.018), "label": (20, 0.015)}
FLOORED = {
    "title-card": {"body": [".subline"], "label": [".top", ".foot", ".eyebrow"]},
    "quote-card": {"body": [".subline"], "label": [".top", ".foot", ".attribution"]},
    "stats-card": {"body": [".subline", ".stat-label", ".note-title", ".note-body"], "label": [".top", ".foot"]},
    "definition-card": {"body": [".subline", ".card-title", ".card-body"], "label": [".top", ".foot"]},
    "announcement-card": {"body": [".subline", ".card-title", ".card-body"], "label": [".status", ".foot"]},
    "carousel": {"body": [".subline", ".point-body", ".cta-pill"], "label": [".top", ".foot"]},
    "infographic": {"body": [".subline", ".note"], "label": [".top", ".chip"]},
    "fact-card": {"body": [".fact-statement", ".fact-context"], "label": [".source", ".top"]},
}


def _html(slug):
    return next(s for s in social_starters() if s["slug"] == slug)["blocks"]["html"]


def _rule(html: str, selector: str) -> str:
    match = re.search(rf"\n\s*{re.escape(selector)} \{{([^{{}}]*)\}}", html)
    assert match, selector
    return match.group(1)


@pytest.mark.parametrize("slug", list(FLOORED))
def test_every_paper_card_holds_both_floors_at_every_size(slug):
    html = _html(slug)
    floors = {kind: (int(units), float(share)) for kind, units, share in FLOOR.findall(html)}
    assert set(floors) == {"body", "label"}, slug
    starter = next(s for s in social_starters() if s["slug"] == slug)
    for kind, (units, share) in floors.items():
        least_units, least_share = LEAST[kind]
        assert units >= least_units and share >= least_share, (slug, kind)
        for size in starter["blocks"]["sizes"]:
            width, height = parse_size(size)
            floor_px = max(units * min(width, height) / 1080, width * share)
            # Shown as wide as a feed post, the floor is never under 80% of a feed post's own floor.
            assert floor_px * 1080 / width >= 0.8 * least_units, (slug, kind, size)


@pytest.mark.parametrize("slug", list(FLOORED))
def test_the_body_text_and_small_print_hold_their_floor(slug):
    html = _html(slug)
    for kind, selectors in FLOORED[slug].items():
        for selector in selectors:
            assert f"font-size: max(var(--{kind}-floor), calc(" in _rule(html, selector), (slug, selector)


def test_the_display_type_is_what_the_fit_shrinks():
    for slug in FLOORED:
        rule = _rule(_html(slug), ".headline")
        assert "font-size: calc(" in rule and "floor" not in rule, slug
