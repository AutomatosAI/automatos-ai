"""F376 (night 11, 7 Oct): the full-width orange bar only under ``accent_use: bold``.

Every paper card drew a full-width bar of the raw primary along its top edge
(``#stripe``); the documents never do: they set one short accent rule. A token now
drives it (``core/social_kit_tokens``: ``accent-bar-span``, 1 under bold, 0 under
sparing), never the template reading the kit: under sparing the bar is a short rule
over the page's left margin, under bold the full width as before. The CI pixel probe
(Title card, Carousel) reads the bar where both are drawn, and the brand board's
social card miniature draws the bar the way a real card does.
"""
from __future__ import annotations

import re

import pytest

from core.media_render_bundle import brand_tokens
from core.social_kit_tokens import ACCENT_BAR_SPAN_TOKEN
from core.social_templates import parse_size
from modules.documents.blocks import brand_board as bb
from modules.documents.blocks import brand_board_html
from modules.documents.social_starters import social_starters

KIT = {"primary_color": "#c44a1a", "secondary_color": "#dcd2bd", "text_color": "#1a1814"}
BARRED = ("announcement-card", "brand-board", "carousel", "infographic", "definition-card", "fact-card",
          "title-card", "quote-card", "stats-card")
BAR_ALIAS = "--bar-span: var(--brand-accent-bar-span, 0);"
BAR_RULE = re.compile(r"--bar-rule: calc\((\d+) \* var\(--u\)\);")
BAR_INSET = re.compile(r"--bar-inset: calc\((\d+) \* var\(--u\)\);")
BAR_HEIGHT = 14  # design pixels
EDGE_PX = 2


def _starter(slug):
    return next(s for s in social_starters() if s["slug"] == slug)


def test_the_bar_spans_the_width_only_under_bold():
    assert brand_tokens(KIT)[ACCENT_BAR_SPAN_TOKEN] == "0"  # sparing, every kit's default
    assert brand_tokens({**KIT, "accent_use": "sparing"})[ACCENT_BAR_SPAN_TOKEN] == "0"
    assert brand_tokens({**KIT, "accent_use": "bold"})[ACCENT_BAR_SPAN_TOKEN] == "1"


@pytest.mark.parametrize("slug", BARRED)
def test_every_paper_card_draws_its_bar_from_the_token(slug):
    html = _starter(slug)["blocks"]["html"]
    assert BAR_ALIAS in html and BAR_RULE.search(html) and BAR_INSET.search(html)
    stripe = re.search(r"#stripe \{([^{}]*)\}", html).group(1)
    assert "var(--bar-span)" in stripe and "width: 100%" not in stripe


@pytest.mark.parametrize("slug", ["title-card", "carousel"])
def test_the_ci_probe_reads_the_bar_where_the_short_rule_is_drawn(slug):
    starter = _starter(slug)
    html, probe = starter["blocks"]["html"], starter["preview"]["probe"]
    rule, inset = int(BAR_RULE.search(html).group(1)), int(BAR_INSET.search(html).group(1))
    for size in starter["blocks"]["sizes"]:
        width, height = parse_size(size)
        u = min(width, height) / 1080
        assert inset * u + EDGE_PX < probe["x"] * width < (inset + rule) * u - EDGE_PX, size
        assert probe["y"] * height + 1 < BAR_HEIGHT * u, size


def test_the_boards_social_card_draws_the_bar_as_a_real_card_does():
    sparing = brand_board_html._social_card(KIT)
    assert f'class="board-social-stripe {brand_board_html.SOCIAL_RULE_CLASS}" style="background:#c44a1a"' in sparing
    bold = brand_board_html._social_card({**KIT, "accent_use": "bold"})
    assert 'class="board-social-stripe" style="background:#c44a1a"' in bold
    assert bb.social_bar({**KIT, "accent_use": "bold"}) == ("#c44a1a", True)
    paper, _ink, _accent = bb.social_colours(KIT)
    assert f'style="background:{paper}"' in sparing  # the documents' paper, as a real card's
