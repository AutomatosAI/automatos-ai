"""PRD-255 Wave 1, US-001 — colour roles in the kit.

Pins (``modules/documents/brand_system.py``, wired through ``brand_kit.py``, the
GET/PUT routes and ``platform_update_brand_kit``):

* **The roles.** ``BrandKit.palette`` has the nine roles, each optional; a set
  role is stored as a 6-digit hex and only set roles are stored. Roles merge key
  by key, and an empty role goes back to derived. An unknown role or a non-hex
  colour is refused.
* **accent_use** is ``sparing`` by default for every kit, old ones included
  (Decision Q1), or ``bold``; anything else is refused.
* **Contrast on save (FR-5).** ink, heading and muted read at 4.5:1 on paper and
  surface_2; the accents at 3:1 (large text). Below that the PUT is a 422 whose
  error names the role (``loc ['palette', role]``) and the failing ratio, and
  nothing is saved. Measured on the EFFECTIVE palette: a stored dark paper fails
  the derived ink.
* **GET** answers the effective roles (stored, else derived) and, per role,
  ``palette_source`` ``set`` or ``derived``; the PUT answers the same shape.
* **The agent tool** takes ``palette`` and ``accent_use`` (the parity test pins
  the top level; this pins the roles and the enum).
"""
from __future__ import annotations

import json
import math
import os
import uuid
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from unittest.mock import MagicMock  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

import api.document_brand_kit as brand_kit_routes  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.brand_palette import PALETTE_ROLES, contrast, derive_palette, parse_hex  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.documents.brand_kit import (  # noqa: E402
    BrandKit,
    brand_kit_errors,
    get_brand_kit,
    validate_brand_kit,
)
from modules.documents.brand_system import (  # noqa: E402
    ACCENT_USES,
    BrandPalette,
    SAVE_ROLE_MIN_CONTRAST,
    palette_contrast_errors,
)
from modules.tools.discovery.action_registry import get_action_registry  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000002f5")
KIT_ROUTE = "/api/documents/brand-kit"
WHITE, BLACK = "#ffffff", "#000000"
# A page and table fill both white: a ratio is then the same on either ground.
WHITE_GROUNDS = {"paper": WHITE, "surface_2": WHITE}
LIGHT_GREY = "#bbbbbb"  # 1.9:1 on white: no text reads in it
MID_GREY = "#888888"  # 3.5:1 on white: large text only
PALE_GREY = "#aaaaaa"  # 2.3:1 on white: not even large text
DEEP_RED = "#aa3333"  # 6.5:1 on white
DARK_GREY = "#555555"  # 7.4:1 on white
V1_KIT = {"primary_color": "#c44a1a", "secondary_color": "#1d3658", "text_color": "#1a1a2e"}


def _ratio_text(colour: str, ground: str) -> str:
    """The ratio as the error prints it: one decimal, rounded down."""
    return f"{math.floor(contrast(parse_hex(colour), parse_hex(ground)) * 10) / 10:.1f}"


def _refusal(patch, existing=None):
    with pytest.raises(ValidationError) as caught:
        validate_brand_kit(patch, existing)
    return brand_kit_errors(caught.value)


def _error_at(errors, *loc):
    found = [error for error in errors if tuple(error["loc"]) == loc]
    assert found, errors
    return found[0]["msg"]


# ---------------------------------------------------------------------------
# The roles and accent_use
# ---------------------------------------------------------------------------


def test_the_palette_has_exactly_the_nine_roles_each_optional():
    assert set(BrandPalette.model_fields) == set(PALETTE_ROLES)
    assert BrandPalette().model_dump() == {}
    assert BrandKit().model_dump()["palette"] == {}


def test_a_set_role_is_stored_as_a_six_digit_hex_and_only_set_roles_are_stored():
    kit = validate_brand_kit({"palette": {"accent": "#A33", "ink": "", "heading": None}})
    assert kit["palette"] == {"accent": DEEP_RED}


def test_an_unknown_role_or_a_colour_that_is_not_hex_is_refused():
    assert _error_at(_refusal({"palette": {"brand": BLACK}}), "palette", "brand")
    assert "hex colour" in _error_at(_refusal({"palette": {"ink": "navy"}}), "palette", "ink")


def test_roles_merge_key_by_key_and_an_empty_role_goes_back_to_derived():
    first = validate_brand_kit({"palette": {"accent": DEEP_RED}})
    both = validate_brand_kit({"palette": {"ink": "#111111"}}, first)
    assert both["palette"] == {"accent": DEEP_RED, "ink": "#111111"}
    cleared = validate_brand_kit({"palette": {"accent": ""}}, both)
    assert cleared["palette"] == {"ink": "#111111"}


def test_accent_use_is_sparing_by_default_for_every_kit_old_ones_included():
    assert get_brand_kit(None)["accent_use"] == "sparing"
    assert get_brand_kit({"brand_kit": V1_KIT})["accent_use"] == "sparing"
    assert ACCENT_USES == ("sparing", "bold")


def test_accent_use_takes_bold_and_refuses_anything_else():
    assert validate_brand_kit({"accent_use": "bold"})["accent_use"] == "bold"
    assert _error_at(_refusal({"accent_use": "loud"}), "accent_use")


def test_a_stored_role_that_is_not_hex_costs_the_read_only_the_palette():
    kit = get_brand_kit({"brand_kit": {**V1_KIT, "palette": {"ink": "navy"}}})
    assert kit["palette"] == {} and kit["primary_color"] == V1_KIT["primary_color"]


# ---------------------------------------------------------------------------
# Contrast on save
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["ink", "heading", "muted"])
def test_a_text_role_under_4_5_to_1_on_the_paper_is_refused_naming_the_role_and_ratio(role):
    msg = _error_at(_refusal({"palette": {**WHITE_GROUNDS, role: LIGHT_GREY}}), "palette", role)
    assert msg == (f"{role} is {_ratio_text(LIGHT_GREY, WHITE)}:1 on the page and table header fills "
                   "(paper and surface_2, white); text needs 4.5:1")


@pytest.mark.parametrize("role", ["accent", "accent_2"])
def test_an_accent_over_3_to_1_is_kept_for_large_text(role):
    kit = validate_brand_kit({"palette": {**WHITE_GROUNDS, role: MID_GREY}})
    assert kit["palette"][role] == MID_GREY
    assert SAVE_ROLE_MIN_CONTRAST[role] == 3.0


@pytest.mark.parametrize("role", ["accent", "accent_2"])
def test_an_accent_under_3_to_1_is_refused_naming_the_ratio(role):
    msg = _error_at(_refusal({"palette": {**WHITE_GROUNDS, role: PALE_GREY}}), "palette", role)
    assert msg == (f"{role} is {_ratio_text(PALE_GREY, WHITE)}:1 on the page and table header fills "
                   "(paper and surface_2, white); text needs 4.5:1 (large text 3:1)")


def test_text_is_measured_on_the_table_fill_too():
    errors = _refusal({"palette": {"paper": WHITE, "surface_2": DARK_GREY, "heading": "#222222"}})
    msg = _error_at(errors, "palette", "heading")
    assert msg.startswith("heading is ") and f"on table header fills (surface_2, {DARK_GREY})" in msg
    assert "on the page" not in msg  # it reads on the white page


def test_a_dark_paper_fails_the_derived_ink_and_says_how_to_fix_it():
    msg = _error_at(_refusal({"palette": {"paper": BLACK}}), "palette", "ink")
    assert msg.startswith("ink is ") and "on the page (paper, black)" in msg
    assert msg.endswith("derived from the kit's colours: set it, or choose a lighter page")


def test_every_derived_palette_of_a_v1_kit_saves():
    for kit in (V1_KIT, {}, {"primary_color": "#ffe9a8", "text_color": "#fafafa"}):
        assert palette_contrast_errors(get_brand_kit({"brand_kit": kit})) == []


# ---------------------------------------------------------------------------
# GET and PUT
# ---------------------------------------------------------------------------


def _ctx():
    return RequestContext(
        workspace_id=WS,
        user=UserContext(id="owner-1", clerk_user_id="clerk-owner-1", system_role="user"),
        auth_type="clerk",
    )


@pytest.fixture
def api(monkeypatch):
    import api.document_generation as documents_module

    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(brand_kit_routes, "resolve_user_pk", lambda db, ctx: None)
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    workspace = SimpleNamespace(id=WS, name="Acme", settings={"brand_kit": dict(V1_KIT), "socials": {"enabled": True}})
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = workspace
    app = FastAPI()
    app.include_router(documents_module.router)
    app.dependency_overrides[get_request_context_hybrid] = _ctx
    app.dependency_overrides[get_db] = lambda: db
    return SimpleNamespace(client=TestClient(app), workspace=workspace)


def test_get_answers_a_v1_kit_with_every_role_derived(api):
    kit = api.client.get(KIT_ROUTE).json()
    assert kit["palette"] == derive_palette(get_brand_kit({"brand_kit": V1_KIT}))
    assert kit["palette_source"] == {role: "derived" for role in kit["palette"]}
    assert kit["accent_use"] == "sparing"


def test_a_put_stores_only_the_set_roles_and_get_reports_them_as_set(api):
    saved = api.client.put(KIT_ROUTE, json={"palette": {"accent": DEEP_RED}, "accent_use": "bold"})
    assert saved.status_code == 200, saved.text
    assert api.workspace.settings["brand_kit"]["palette"] == {"accent": DEEP_RED}
    kit = api.client.get(KIT_ROUTE).json()
    assert kit == saved.json()
    assert kit["palette"]["accent"] == DEEP_RED and kit["palette_source"]["accent"] == "set"
    assert {kit["palette_source"][role] for role in kit["palette"] if role != "accent"} == {"derived"}
    assert kit["accent_use"] == "bold"


def test_a_put_that_fails_contrast_is_a_422_naming_the_role_and_saves_nothing(api):
    before = json.loads(json.dumps(api.workspace.settings))
    # muted is set too: derived, it is the ink lightened, and would fail with it.
    palette = {**WHITE_GROUNDS, "ink": LIGHT_GREY, "muted": DARK_GREY}
    refused = api.client.put(KIT_ROUTE, json={"name": "Acme Studio", "palette": palette})
    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert detail["message"] == "Invalid brand kit"
    assert [(error["loc"], error["msg"]) for error in detail["errors"]] == [
        (["palette", "ink"], f"ink is {_ratio_text(LIGHT_GREY, WHITE)}:1 on the page and table header fills "
                             "(paper and surface_2, white); text needs 4.5:1"),
    ]
    assert api.workspace.settings == before


# ---------------------------------------------------------------------------
# The agent tool
# ---------------------------------------------------------------------------


def test_the_update_tool_takes_every_role_and_the_accent_use():
    properties = get_action_registry().get("platform_update_brand_kit").parameters["properties"]
    assert set(properties["palette"]["properties"]) == set(PALETTE_ROLES)
    assert tuple(properties["accent_use"]["enum"]) == ACCENT_USES
