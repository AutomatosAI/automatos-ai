"""F382 (night 11, 7 Oct), B7/B21: an IG story's background runs to the bottom of the story.

A story's 9:16 render (PRD-251C US-C301) moved the template's ``.page`` box to 250 px
from the top and 340 px from the bottom, clear of Instagram's bars. The photo cards'
dark panel lives inside the page, so on the Offer and the Before/after stories it
ended at about 80% of the height with a hard edge. The safe zone is now two insets
(``--story-top``, ``--story-bottom``) every still template adds to its words' own
spacing: the page box and every background stay full-bleed and only the words keep
clear. Pins: the zone's css, and every still template reading both insets where its
words start and end, with its page box still the whole card.
"""
from __future__ import annotations

import re

import pytest

from core.media_render_bundle import STORY_BOTTOM_VAR, STORY_TOP_VAR, build_bundle, story_safe_css
from core.social_templates import SOCIAL_IMAGE, resolve_variables
from modules.documents.social_starters import SOCIAL_PHOTO_STARTER_SLUGS, social_starters

TOP, BOTTOM = "var(--story-top, 0px)", "var(--story-bottom, 0px)"
# Where each photo card's words end: the block whose own spacing takes the bottom inset.
PHOTO_BOTTOMS = {"photo-headline": ".panel", "photo-offer": ".panel", "photo-review": ".card", "photo-only": ".foot",
                 "photo-highlights": ".side", "before-after": ".panel"}


def _starter(slug):
    return next(s for s in social_starters() if s["slug"] == slug)


def _first_rule(html: str, selector: str) -> str:
    match = re.search(rf"\n\s*{re.escape(selector)} \{{([^{{}}]*)\}}", html)
    assert match, selector
    return match.group(1)


def test_the_safe_zone_is_two_insets_not_a_moved_page():
    css = story_safe_css(1080, 1920)
    assert f"{STORY_TOP_VAR}: 250px" in css and f"{STORY_BOTTOM_VAR}: 340px" in css
    assert ".page" not in css and "!important" not in css


@pytest.mark.parametrize("slug", [s["slug"] for s in social_starters(SOCIAL_IMAGE)])
def test_every_still_template_keeps_its_words_inside_the_insets_and_its_page_full_bleed(slug):
    html = _starter(slug)["blocks"]["html"]
    assert TOP in html and BOTTOM in html, slug
    page = _first_rule(html, ".page")
    assert "inset: 0" in page and "--story" not in page.split("padding")[0], slug


@pytest.mark.parametrize("slug", list(SOCIAL_PHOTO_STARTER_SLUGS))
def test_a_photo_cards_panel_runs_to_the_bottom_and_its_words_keep_clear(slug):
    rule = _first_rule(_starter(slug)["blocks"]["html"], PHOTO_BOTTOMS[slug])
    assert f"calc(60 * var(--u) + {BOTTOM})" in rule, slug


def test_the_offer_story_bundle_carries_the_insets_and_the_authored_page():
    starter = _starter("photo-offer")
    values = resolve_variables(starter["blocks"]["variables_schema"], starter["sample_data"]).values
    story = build_bundle(workspace_id="ws", reference="r", blocks=starter["blocks"], values=values, brand_kit={},
                         size="1080x1920", fmt=SOCIAL_IMAGE, story_safe=True)
    assert story["composition"]["css"] == (starter["blocks"].get("css") or "") + story_safe_css(1080, 1920)
    assert "position: absolute; inset: 0;" in _first_rule(story["composition"]["html"], ".page")
