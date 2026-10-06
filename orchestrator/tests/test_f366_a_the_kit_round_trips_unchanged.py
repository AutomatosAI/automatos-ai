"""F366 (night 10c) — GET → PUT of the brand kit changes nothing, and resetting a role is documented.

The night's owner read the kit (every role ``derived``) and PUT it back with one
colour changed: all eight roles came back ``set``, pinned at that day's colours,
and ``palette_source`` in the body was ignored. Getting "derived" back took
``null`` per role, found by guessing. A pale accent's 422 said "on surface_2",
not where the colour sits. Pins (``modules/documents/brand_palette_source.py``,
``brand_system.contrast_message``):

* the GET body PUT back unchanged, with or without ``palette_source``, changes nothing;
* one colour changed in a GET body is the only role that becomes ``set``;
* a role marked ``derived`` follows a primary the same save changes; a role sent at
  today's colour with no ``palette_source`` is kept at it;
* a role is reset by ``palette_source`` ``derived`` (or ``null``), every role by
  ``palette_source: "derived"``; ``set`` pins a role at its colour; an unknown role is refused;
* the contrast 422 says what the colour sits on in plain words beside the role key.
"""
from __future__ import annotations

from tests import test_prd255w1_palette_roles as roles_tests

api = roles_tests.api  # the documents router over one workspace holding a v1 kit
KIT_ROUTE = roles_tests.KIT_ROUTE
TEAL = "#0f5c5c"
PALE_YELLOW = "#f5e9a0"  # 1.2:1 on white: the night's slip
NEW_PRIMARY = "#0055aa"


def _get(api):
    answer = api.client.get(KIT_ROUTE)
    assert answer.status_code == 200, answer.text
    return answer.json()


def _put(api, body):
    answer = api.client.put(KIT_ROUTE, json=body)
    assert answer.status_code == 200, answer.text
    return answer.json()


def _set_roles(kit):
    return {role for role, source in kit["palette_source"].items() if source == "set"}


def _stored_palette(api):
    return api.workspace.settings["brand_kit"]["palette"]


def test_the_get_body_put_back_unchanged_changes_nothing(api):
    before = _get(api)
    assert _set_roles(before) == set()
    assert _put(api, before) == before
    assert _get(api) == before and _stored_palette(api) == {}


def test_the_get_body_without_palette_source_changes_nothing_either(api):
    before = _get(api)
    body = {key: value for key, value in before.items() if key != "palette_source"}
    assert _put(api, body) == before and _stored_palette(api) == {}


def test_a_set_role_stays_set_through_the_round_trip(api):
    _put(api, {"palette": {"accent": TEAL}})
    before = _get(api)
    assert _set_roles(before) == {"accent"}
    assert _put(api, before) == before
    assert _stored_palette(api) == {"accent": TEAL}


def test_one_colour_changed_in_a_get_body_is_the_only_role_set(api):
    body = _get(api)
    saved = _put(api, {**body, "palette": {**body["palette"], "accent": TEAL}})
    assert saved["palette"]["accent"] == TEAL
    assert _set_roles(saved) == {"accent"} and _stored_palette(api) == {"accent": TEAL}


def test_a_role_marked_derived_follows_a_primary_the_same_save_changes(api):
    body = _get(api)
    saved = _put(api, {**body, "primary_color": NEW_PRIMARY})
    assert _set_roles(saved) == set()
    assert saved["palette"]["accent"] != body["palette"]["accent"]  # the accent derives from the new primary


def test_a_role_sent_at_todays_colour_without_a_source_is_kept_when_the_primary_moves(api):
    accent = _get(api)["palette"]["accent"]
    saved = _put(api, {"primary_color": NEW_PRIMARY, "palette": {"accent": accent}})
    assert saved["palette"]["accent"] == accent and _set_roles(saved) == {"accent"}


def test_palette_source_derived_resets_one_role(api):
    _put(api, {"palette": {"accent": TEAL, "ink": "#111111"}})
    saved = _put(api, {"palette_source": {"accent": "derived"}})
    assert _set_roles(saved) == {"ink"} and _stored_palette(api) == {"ink": "#111111"}


def test_marking_a_role_derived_in_the_get_body_resets_it(api):
    _put(api, {"palette": {"accent": TEAL}})
    body = _get(api)
    saved = _put(api, {**body, "palette_source": {**body["palette_source"], "accent": "derived"}})
    assert _set_roles(saved) == set() and saved["palette"]["accent"] != TEAL


def test_null_still_resets_a_role(api):
    _put(api, {"palette": {"accent": TEAL}})
    assert _set_roles(_put(api, {"palette": {"accent": None}})) == set()


def test_palette_source_derived_alone_resets_every_role(api):
    _put(api, {"palette": {"accent": TEAL, "ink": "#111111", "rule": "#cccccc"}})
    saved = _put(api, {"palette_source": "derived"})
    assert _set_roles(saved) == set() and _stored_palette(api) == {}


def test_palette_source_set_pins_a_role_at_its_colour(api):
    rule = _get(api)["palette"]["rule"]
    saved = _put(api, {"palette_source": {"rule": "set"}})
    assert _set_roles(saved) == {"rule"} and _stored_palette(api) == {"rule": rule}


def test_palette_source_naming_no_role_is_refused(api):
    refused = api.client.put(KIT_ROUTE, json={"palette_source": {"brand": "derived"}})
    assert refused.status_code == 422
    assert "palette_source has no role brand" in refused.text


def test_the_contrast_refusal_says_what_the_colour_sits_on(api):
    refused = api.client.put(KIT_ROUTE, json={"palette": {"accent": PALE_YELLOW}})
    assert refused.status_code == 422
    (error,) = refused.json()["detail"]["errors"]
    assert error["loc"] == ["palette", "accent"]
    msg = error["msg"]
    assert msg.startswith("accent is 1.2:1 on the page (paper, white) and ")
    assert "on table header fills (surface_2, #" in msg and msg.endswith("text needs 4.5:1 (large text 3:1)")


def test_the_update_tool_takes_palette_source():
    from modules.tools.discovery.action_registry import get_action_registry

    tool = get_action_registry().get("platform_update_brand_kit")
    schema = tool.parameters["properties"]["palette_source"]
    assert set(schema["properties"]) == set(roles_tests.PALETTE_ROLES)
    assert all(role["enum"] == ["set", "derived"] for role in schema["properties"].values())
    assert "palette_source" in tool.description
