"""F361 (night 10c) — one accent: the kit's ``accent_color`` is the third colour, and it is contrast-checked.

The night's kit held two "accents" that disagreed: ``accent_color`` (navy, labelled
"Accent" on the Brand kit page) and ``palette.accent`` (orange, the highlight). The
legacy one had no contrast check: #f5e9a0 was accepted, and, being light enough to be
a page, it became every document's page until it was reverted. It is not retired:
social videos tint and mark with it (``--brand-accent-on-ink``), the social brand
board shows it, and ``{{brand.accent_color}}`` prints it. So it is relabelled "the
third colour" wherever an owner or an agent meets it, and a save that changes it
measures it as the accents are (3:1 on the page and on table header fills). Pins:

* a pale third colour is refused, naming the field, what it sits on and, when it would
  be the page itself, that; nothing is saved;
* a readable one saves; a stored one that fails blocks no other save, nor the GET body
  sent back;
* the agent tool, the template variable and the social brand board call it the third
  colour, and the tool points "the accent" at ``palette.accent``.
"""
from __future__ import annotations

import json

from modules.documents.social_starters import social_starters
from modules.documents.variables.catalog import CATALOG_BY_PATH
from modules.tools.discovery.action_registry import get_action_registry
from tests import test_prd255w1_palette_roles as roles_tests

api = roles_tests.api  # the documents router over one workspace holding a v1 kit (white page)
KIT_ROUTE = roles_tests.KIT_ROUTE
PALE_YELLOW = "#f5e9a0"  # the night's slip: light enough to be a page
GOLD = "#e0a458"  # 2.1:1 on white
NAVY = "#1d3658"  # 12:1 on white
THIRD_COLOUR = "accent_color (the third colour)"


def _refused(api, body):
    before = json.loads(json.dumps(api.workspace.settings))
    refused = api.client.put(KIT_ROUTE, json=body)
    assert refused.status_code == 422, refused.text
    assert api.workspace.settings == before  # nothing saved
    (error,) = refused.json()["detail"]["errors"]
    assert error["loc"] == ["accent_color"]
    return error["msg"]


def test_a_third_colour_that_does_not_read_on_the_page_is_refused(api):
    msg = _refused(api, {"accent_color": GOLD})
    assert msg.startswith(f"{THIRD_COLOUR} is 2.1:1 on the page (paper, white) and ")
    assert msg.endswith("text needs 4.5:1 (large text 3:1)")


def test_a_third_colour_light_enough_to_be_the_page_says_so(api):
    msg = _refused(api, {"accent_color": PALE_YELLOW})
    assert msg.startswith(f"{THIRD_COLOUR} is 1.0:1 on the page (paper, {PALE_YELLOW})")
    assert msg.endswith("a colour this light becomes the page itself: choose a darker one")


def test_a_readable_third_colour_saves(api):
    saved = api.client.put(KIT_ROUTE, json={"accent_color": NAVY})
    assert saved.status_code == 200, saved.text
    assert saved.json()["accent_color"] == NAVY


def test_a_stored_third_colour_that_fails_blocks_no_other_save(api):
    api.workspace.settings = {**api.workspace.settings, "brand_kit": {**roles_tests.V1_KIT, "accent_color": GOLD}}
    renamed = api.client.put(KIT_ROUTE, json={"name": "Harbourline"})
    assert renamed.status_code == 200, renamed.text
    resent = api.client.put(KIT_ROUTE, json=renamed.json())
    assert resent.status_code == 200, resent.text
    assert resent.json()["accent_color"] == GOLD


def test_the_agent_tool_calls_it_the_third_colour_and_points_the_accent_at_the_palette():
    described = get_action_registry().get("platform_update_brand_kit").parameters["properties"]["accent_color"]
    text = described["description"]
    assert "third colour" in text and "palette.accent" in text and "3:1" in text


def test_the_template_variable_and_the_social_brand_board_call_it_the_third_colour():
    assert CATALOG_BY_PATH["brand.accent_color"]["label"] == "Third colour"
    (board,) = [starter for starter in social_starters() if starter["slug"] == "brand-board"]
    label = board["blocks"]["variables_schema"]["accent_label"]
    assert label["default"] == "Third colour" and "accent" not in label["label"].lower()
