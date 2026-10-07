"""The kit has no accent colour of its own: the accent is the palette's (Gerard, 7 Oct).

F361 relabelled the kit's ``accent_color`` "the third colour (social videos)" and
contrast-checked it, but it still disagreed with the palette's accent, the one every
document highlights with. It is retired: the field, its patch, the agent tool's
schema, ``{{brand.accent_color}}`` and the Brand kit page's field are gone. What read
it reads the palette (``core.brand_palette.social_accent``): the social videos'
``accent-on-ink``, the social images' ``accent-on-paper`` and the ``accent`` token the
social brand board shows are the palette's ``accent_2`` when it has one, else its
``accent``. Pins:

* a stored kit that still carries ``accent_color`` loads without it, and the next save
  leaves it out (the settings JSONB: no migration);
* a PUT that sends it changes nothing, and the agent tool refuses it;
* no derivation looks at it: a pale one no longer becomes the page;
* the social accent is the palette's, with the primary as the last fallback.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from core.brand_palette import derive_palette, paper_palette, social_accent, stage_palette, to_hex
from core.media_render_bundle import brand_tokens
from modules.documents.brand_kit import BrandKit, BrandKitPatch, get_brand_kit
from modules.documents.social_starters import social_starters
from modules.documents.variables.catalog import CATALOG_BY_PATH
from modules.tools.discovery.action_registry import get_action_registry
from modules.tools.discovery.handlers_documents import update_brand_kit_tool
from tests import test_prd255w1_palette_roles as roles_tests

api = roles_tests.api  # the documents router over one workspace holding a v1 kit (white page)
KIT_ROUTE = roles_tests.KIT_ROUTE
V1_KIT = roles_tests.V1_KIT  # orange primary, navy secondary: a second hue
RETIRED = "accent_color"
PALE_YELLOW = "#f5e9a0"  # F361's slip: light enough to be a page
TEAL, PLUM = "#0f5c5c", "#6b2d5c"
GREY_SECONDARY = {"primary_color": "#c44a1a", "secondary_color": "#777777", "text_color": "#1a1a2e"}


def test_a_stored_kit_that_still_carries_it_loads_without_it_and_the_next_save_drops_it(api):
    api.workspace.settings = {**api.workspace.settings, "brand_kit": {**V1_KIT, RETIRED: PALE_YELLOW}}
    loaded = api.client.get(KIT_ROUTE)
    assert loaded.status_code == 200, loaded.text
    assert RETIRED not in loaded.json() and loaded.json()["primary_color"] == V1_KIT["primary_color"]
    saved = api.client.put(KIT_ROUTE, json={"name": "Harbourline"})
    assert saved.status_code == 200, saved.text
    assert RETIRED not in api.workspace.settings["brand_kit"]
    assert api.workspace.settings["brand_kit"]["name"] == "Harbourline"


def test_a_put_that_sends_it_changes_nothing(api):
    before = json.loads(json.dumps(api.workspace.settings))
    sent = api.client.put(KIT_ROUTE, json={RETIRED: PALE_YELLOW})
    assert sent.status_code == 200, sent.text
    assert RETIRED not in sent.json() and api.workspace.settings == before


def test_the_agent_tool_refuses_it_and_nothing_names_it():
    refused = asyncio.run(update_brand_kit_tool(MagicMock(), roles_tests.WS, {RETIRED: TEAL}))
    assert refused["success"] is False and f"cannot set {RETIRED}" in refused["error"]
    assert RETIRED not in get_action_registry().get("platform_update_brand_kit").parameters["properties"]
    assert RETIRED not in BrandKit.model_fields and RETIRED not in BrandKitPatch.model_fields
    assert f"brand.{RETIRED}" not in CATALOG_BY_PATH
    assert RETIRED not in get_brand_kit({"brand_kit": {RETIRED: TEAL}})


def test_no_derivation_looks_at_it():
    carrying = {**V1_KIT, RETIRED: PALE_YELLOW}
    assert derive_palette(carrying) == derive_palette(V1_KIT)  # a pale one no longer becomes the page
    assert paper_palette(carrying) == paper_palette(V1_KIT)
    assert stage_palette(carrying) == stage_palette(V1_KIT)
    assert brand_tokens({**V1_KIT, RETIRED: PLUM}) == brand_tokens(V1_KIT)


def test_the_social_accent_is_the_palettes_second_accent_else_its_accent():
    # A stored accent_2 wins; else the secondary when it is a hue of its own (the derived accent_2).
    assert to_hex(social_accent({**V1_KIT, "palette": {"accent_2": TEAL, "accent": PLUM}})) == TEAL
    assert to_hex(social_accent({**V1_KIT, "palette": {"accent": PLUM}})) == V1_KIT["secondary_color"]
    # A grey secondary is no second accent: the stored accent, else the primary.
    assert to_hex(social_accent({**GREY_SECONDARY, "palette": {"accent": PLUM}})) == PLUM
    assert to_hex(social_accent(GREY_SECONDARY)) == GREY_SECONDARY["primary_color"]
    assert social_accent({"text_color": "#1a1a2e"}) is None


def test_the_social_tokens_and_the_brand_board_read_it():
    kit = {**GREY_SECONDARY, "palette": {"accent": PLUM}}
    tokens = brand_tokens(kit)
    assert tokens["accent"] == PLUM  # the social brand board's swatch
    assert tokens["accent-on-paper"] == PLUM  # reads on the paper as it is
    assert tokens["accent-on-ink"] == stage_palette(kit)["accent-on-ink"] != tokens["primary-on-ink"]
    (board,) = [starter for starter in social_starters() if starter["slug"] == "brand-board"]
    label = board["blocks"]["variables_schema"]["accent_label"]
    assert label["label"] == label["default"] == "Accent"
