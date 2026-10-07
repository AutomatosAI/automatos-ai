"""F376 (night 11, 7 Oct), #1812: a font stack a save sets keeps its fallbacks.

The Brand Designer's approved proposal saved ``font_family: "Geist"`` and
``heading_font: "Newsreader"`` over full stacks. A bare name is a valid
``--brand-heading-font``, so the template's ``var()`` fallback never applied and
Chrome fell to Times. Pins:

* at write time (``brand_kit.validate_brand_kit``, through
  ``font_fallbacks.with_font_fallbacks``), a stack without a generic family keeps the
  stack it replaces after it, each family once; failing that, its generic family;
* a stack with a generic family, an empty heading font and an unsafe stack are as
  before (the heading font's validator still refuses what is not one CSS value);
* at bundle time (``brand_tokens``), a stored bare stack gets its generic family.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.font_stacks import FAMILY_GENERICS, GENERIC_FAMILIES, has_generic, with_generic
from core.media_render_bundle import brand_tokens
from modules.documents import bundled_fonts
from modules.documents.brand_kit import DEFAULT_FONT, validate_brand_kit
from modules.documents.font_fallbacks import kept_fallbacks

NIGHT_BODY = "Geist, Inter, 'Segoe UI', system-ui, sans-serif"
NIGHT_HEADING = "Newsreader, Georgia, serif"
STORED = {"font_family": NIGHT_BODY, "heading_font": NIGHT_HEADING}


def test_the_designers_bare_names_keep_the_stacks_they_replace():
    kit = validate_brand_kit({"font_family": "Geist", "heading_font": "Newsreader"}, STORED)
    assert kit["font_family"] == NIGHT_BODY
    assert kit["heading_font"] == NIGHT_HEADING


def test_a_new_first_family_goes_in_front_of_the_stack_it_replaces():
    kit = validate_brand_kit({"font_family": "'Brand Sans'", "heading_font": "Playfair Display"}, STORED)
    assert kit["font_family"] == f"'Brand Sans', {NIGHT_BODY}"
    assert kit["heading_font"] == f"Playfair Display, {NIGHT_HEADING}"


def test_with_nothing_to_keep_the_generic_family_of_the_first_one_is_added():
    kit = validate_brand_kit({"font_family": "Geist", "heading_font": "Newsreader"})
    assert kit["font_family"] == f"Geist, {DEFAULT_FONT}"  # the default body stack it replaced
    assert kit["heading_font"] == "Newsreader, serif"  # no heading font before: its own generic
    assert kept_fallbacks("Playfair Display", "") == "Playfair Display, sans-serif"
    assert kept_fallbacks("Newsreader", "Georgia") == "Newsreader, Georgia, serif"


def test_a_stack_with_a_generic_family_and_an_empty_one_are_kept_as_sent():
    kit = validate_brand_kit({"heading_font": "'Brand Serif', serif", "font_family": "system-ui"}, STORED)
    assert (kit["heading_font"], kit["font_family"]) == ("'Brand Serif', serif", "system-ui")
    assert validate_brand_kit({"heading_font": ""}, STORED)["heading_font"] == ""
    assert not has_generic("'serif'")  # a quoted name is a font, not the generic


@pytest.mark.parametrize("stack", ["Inter; } body { color: red", "x" * 201, "Inter\nsans"])
def test_a_heading_font_that_is_not_one_css_value_is_still_refused(stack):
    with pytest.raises(ValidationError, match="heading_font"):
        validate_brand_kit({"heading_font": stack}, STORED)


def test_a_stored_bare_stack_reaches_the_render_with_its_generic_family():
    tokens = brand_tokens({"font_family": "Geist", "heading_font": "Newsreader"})
    assert tokens["body-font"] == tokens["mono-font"] == "Geist, sans-serif"
    assert tokens["heading-font"] == "Newsreader, serif"
    assert brand_tokens({"font_family": NIGHT_BODY})["body-font"] == NIGHT_BODY  # a full stack is as it is
    assert with_generic("") == ""


def test_a_long_bare_stack_gives_way_at_its_end_never_its_generic_family():
    from core.social_kit_tokens import MAX_TOKEN_CHARS, font_tokens

    families = ["Geist"] + [f"Family {n:02d}" for n in range(17)]  # 192 characters, no generic family
    stack = ", ".join(families)
    assert len(stack) <= MAX_TOKEN_CHARS < len(with_generic(stack))
    kept = font_tokens({"body-font": stack})["body-font"]
    assert kept.startswith("Geist, Family 00") and kept.endswith(", sans-serif") and len(kept) <= MAX_TOKEN_CHARS


def test_the_shipped_families_and_the_generic_names_agree_with_the_bundled_fonts():
    assert set(FAMILY_GENERICS) == {family.casefold() for family in bundled_fonts.BUNDLED_FAMILIES}
    assert set(bundled_fonts.GENERIC_FAMILIES) <= GENERIC_FAMILIES
