"""PRD-255 Wave 1, US-002 — type scale, spacing, logo rules, logo variants, locale and tone meanings.

Pins (``modules/documents/brand_system.py``, wired through ``brand_kit.py``, the
brand kit routes and ``platform_update_brand_kit``):

* **The type scale.** Seven steps, each ``{size_pt, line_pt, weight}``, with ONE
  default professional document scale (Decision Q5: no presets). A step merges
  field by field; sizes, line heights and weights are bounded.
* **Spacing and margins.** ``spacing_unit_pt`` (4) and ``page_margin_mm`` (18), bounded.
* **Logo rules.** ``letterhead_mm`` (16), ``clear_space`` (0.5 logo heights),
  ``min_mm`` (8); merged key by key; the letterhead logo is never under its least size.
* **Logo variants.** ``logo_dark_path`` and ``logo_mono_path`` are uploads only
  (``/brand-kit/logo-dark``, ``/brand-kit/logo-mono``: upload, stream, remove), held
  to the logo's rules, written by a workspace manager, never by a PUT, and empty
  (never invented) until uploaded.
* **Locale.** ``currency`` is empty (no currency printed, FR-7) or an ISO 4217
  code; ``date_style`` is ``d MMMM yyyy`` (default) or ``MMMM d, yyyy``.
* **Tone meanings.** ``voice.tone`` is ``[{word, meaning}]``; a plain string stays
  valid and reads as a word with no meaning; the agents' rules block prints the meaning.
* **The agent tool** takes every new writable field (the parity test pins the top
  level; this pins the steps, the enum and the tone items).
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

import api.document_brand_kit as brand_kit_routes  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.documents.brand_logo as bl  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.documents.brand_kit import (  # noqa: E402
    PATCH_FIELDS,
    SERVER_MANAGED_FIELDS,
    BrandKit,
    brand_kit_errors,
    get_brand_kit,
    validate_brand_kit,
)
from modules.documents.brand_system import (  # noqa: E402
    DATE_STYLES,
    TYPE_STEPS,
    TypeScale,
    tone_words,
)
from modules.tools.discovery.action_registry import get_action_registry  # noqa: E402
from modules.tools.discovery.platform_executor import PlatformActionExecutor  # noqa: E402
from services.brand_rules import rules_for_kit  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000002f6")
KIT_ROUTE = "/api/documents/brand-kit"
DARK_ROUTE = "/api/documents/brand-kit/logo-dark"
MONO_ROUTE = "/api/documents/brand-kit/logo-mono"
MANIFEST = Path(__file__).resolve().parents[1] / "reports" / "route-manifest.json"
# The PRD's professional document scale (size_pt, line_pt, weight); display is the loop's choice.
PRD_SCALE = {
    "display": (32, 38, 600),
    "h1": (22, 28, 600),
    "h2": (15, 20, 600),
    "h3": (12, 16, 600),
    "body": (10, 15, 400),
    "small": (8.5, 12, 400),
    "caption": (7.5, 10, 400),
}


def png_bytes(width: int, height: int) -> bytes:
    """A PNG signature + IHDR chunk (enough header for sniffing + dimensions)."""
    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + ihdr + b"\x00" * 16


def _refusal(patch_, existing=None):
    with pytest.raises(ValidationError) as caught:
        validate_brand_kit(patch_, existing)
    return brand_kit_errors(caught.value)


def _msg_at(errors, *loc):
    found = [error for error in errors if tuple(error["loc"])[: len(loc)] == loc]
    assert found, errors
    return found[0]["msg"]


def _step(kit, step):
    value = kit["type_scale"][step]
    return (value["size_pt"], value["line_pt"], value["weight"])


# ---------------------------------------------------------------------------
# The type scale
# ---------------------------------------------------------------------------


def test_every_kit_old_ones_included_reads_the_one_default_scale():
    assert TYPE_STEPS == ("display", "h1", "h2", "h3", "body", "small", "caption")
    assert set(TypeScale.model_fields) == set(TYPE_STEPS)  # no preset field: one scale (Decision Q5)
    for kit in (get_brand_kit(None), get_brand_kit({"brand_kit": {"name": "Acme", "primary_color": "#0055aa"}})):
        assert {step: _step(kit, step) for step in TYPE_STEPS} == PRD_SCALE


def test_a_type_step_merges_field_by_field_and_the_other_steps_keep_their_values():
    first = validate_brand_kit({"type_scale": {"h1": {"size_pt": 24, "line_pt": 30, "weight": 700}}})
    assert _step(first, "h1") == (24, 30, 700)
    second = validate_brand_kit({"type_scale": {"h1": {"size_pt": 26}, "body": {"weight": 300}}}, first)
    assert _step(second, "h1") == (26, 30, 700)  # the stored line and weight kept
    assert _step(second, "body") == (10, 15, 300)
    assert _step(second, "h2") == PRD_SCALE["h2"]


@pytest.mark.parametrize(
    "step, value, field, reason",
    [
        ("h1", {"size_pt": 4}, "size_pt", "5 to 96 pt"),
        ("display", {"size_pt": 120, "line_pt": 130}, "size_pt", "5 to 96 pt"),
        ("body", {"line_pt": 9}, None, "line_pt must be from size_pt"),
        ("body", {"line_pt": 31}, None, "line_pt must be from size_pt"),
        ("h2", {"weight": 650}, "weight", "in hundreds"),
        ("h2", {"weight": 1000}, "weight", "in hundreds"),
    ],
)
def test_a_type_step_out_of_bounds_is_refused_naming_the_step(step, value, field, reason):
    loc = ("type_scale", step, field) if field else ("type_scale", step)
    assert reason in _msg_at(_refusal({"type_scale": {step: value}}), *loc)


def test_an_unknown_step_or_a_preset_is_refused():
    assert _msg_at(_refusal({"type_scale": {"h4": {"size_pt": 10}}}), "type_scale", "h4")
    assert _msg_at(_refusal({"type_scale": {"preset": "editorial"}}), "type_scale", "preset")


# ---------------------------------------------------------------------------
# Spacing, margins and logo rules
# ---------------------------------------------------------------------------


def test_spacing_and_margin_default_to_4_pt_and_18_mm_and_are_bounded():
    kit = get_brand_kit(None)
    assert (kit["spacing_unit_pt"], kit["page_margin_mm"]) == (4, 18)
    saved = validate_brand_kit({"spacing_unit_pt": 5, "page_margin_mm": 22.5})
    assert (saved["spacing_unit_pt"], saved["page_margin_mm"]) == (5, 22.5)
    assert "2 to 12 pt" in _msg_at(_refusal({"spacing_unit_pt": 1}), "spacing_unit_pt")
    assert "6 to 50 mm" in _msg_at(_refusal({"page_margin_mm": 60}), "page_margin_mm")


def test_logo_rules_default_merge_key_by_key_and_are_bounded():
    assert get_brand_kit(None)["logo_rules"] == {"letterhead_mm": 16, "clear_space": 0.5, "min_mm": 8}
    first = validate_brand_kit({"logo_rules": {"letterhead_mm": 20}})
    second = validate_brand_kit({"logo_rules": {"clear_space": 1}}, first)
    assert second["logo_rules"] == {"letterhead_mm": 20, "clear_space": 1, "min_mm": 8}
    assert "6 to 60 mm" in _msg_at(_refusal({"logo_rules": {"letterhead_mm": 80}}), "logo_rules", "letterhead_mm")
    assert "0 to 2 logo heights" in _msg_at(_refusal({"logo_rules": {"clear_space": 3}}), "logo_rules", "clear_space")
    assert "4 to 40 mm" in _msg_at(_refusal({"logo_rules": {"min_mm": 2}}), "logo_rules", "min_mm")
    assert "at least min_mm" in _msg_at(_refusal({"logo_rules": {"letterhead_mm": 10, "min_mm": 12}}), "logo_rules")


# ---------------------------------------------------------------------------
# Locale
# ---------------------------------------------------------------------------


def test_currency_is_empty_by_default_and_otherwise_an_iso_4217_code():
    assert get_brand_kit(None)["currency"] == ""  # FR-7: no currency the kit doesn't have
    assert validate_brand_kit({"currency": " gbp "})["currency"] == "GBP"
    assert validate_brand_kit({"currency": ""}, {"currency": "EUR"})["currency"] == ""
    for bad in ("GB", "POUND", "£", "G1P"):
        assert "ISO 4217" in _msg_at(_refusal({"currency": bad}), "currency"), bad


def test_date_style_is_day_first_by_default_or_month_first():
    assert DATE_STYLES == ("d MMMM yyyy", "MMMM d, yyyy")
    assert get_brand_kit(None)["date_style"] == "d MMMM yyyy"
    assert validate_brand_kit({"date_style": "MMMM d, yyyy"})["date_style"] == "MMMM d, yyyy"
    assert _msg_at(_refusal({"date_style": "dd/MM/yyyy"}), "date_style")


# ---------------------------------------------------------------------------
# Tone words with meanings
# ---------------------------------------------------------------------------


def test_a_tone_word_carries_a_meaning_and_a_plain_string_stays_valid():
    kit = validate_brand_kit({"voice": {"tone": [
        {"word": "Warm", "meaning": " friendly, never gushing "}, "plain", {"word": "local"}, "warm",
    ]}})
    assert kit["voice"]["tone"] == [
        {"word": "Warm", "meaning": "friendly, never gushing"},
        {"word": "plain", "meaning": ""},
        {"word": "local", "meaning": ""},
    ]
    # A v1 kit's plain words read the same way.
    old = get_brand_kit({"brand_kit": {"voice": {"tone": ["warm", "plain", "local"]}}})
    assert [t["meaning"] for t in old["voice"]["tone"]] == ["", "", ""]


def test_a_meaning_is_one_short_line_and_a_word_still_needs_a_letter():
    words = ["plain", "local"]
    for bad, reason in (
        ({"word": "warm", "meaning": "x" * 121}, "at most 120"),
        ({"word": "warm", "meaning": "kind\nand open"}, "one line"),
        ({"word": "123", "meaning": "numbers"}, "a letter"),
        ({"word": "warm", "colour": "red"}, "Extra inputs"),
    ):
        with pytest.raises(ValidationError, match=reason):
            validate_brand_kit({"voice": {"tone": [bad, *words]}})


def test_tone_words_reads_every_stored_form_and_the_rules_block_prints_the_meaning():
    kit = {"name": "Harbourline", "voice": {"tone": ["warm", {"word": "plain", "meaning": "short words"}, 7, {}]}}
    assert tone_words(kit) == [{"word": "warm", "meaning": ""}, {"word": "plain", "meaning": "short words"}]
    assert tone_words({"voice": "warm"}) == [] and tone_words(None) == []
    assert "- Tone: warm, plain (short words)." in rules_for_kit(kit)


# ---------------------------------------------------------------------------
# Logo variants
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(bl.config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(bl, "is_storage_configured", lambda: False)
    return tmp_path


def _ctx():
    return RequestContext(
        workspace_id=WS,
        user=UserContext(id="owner-1", clerk_user_id="clerk-owner-1", system_role="user"),
        auth_type="clerk",
    )


@pytest.fixture
def api(storage, monkeypatch):
    """The documents router (the brand kit routes ride it) over one workspace held in memory."""
    import api.document_generation as documents_module

    role = {"value": "owner"}
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: role["value"])
    monkeypatch.setattr(brand_kit_routes, "resolve_user_pk", lambda db, ctx: None)
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    stored = {"name": "Acme", "logo_url": "https://cdn.acme.com/logo.png"}
    workspace = SimpleNamespace(id=WS, name="Acme", settings={"brand_kit": stored, "socials": {"enabled": True}})
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = workspace
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
    app = FastAPI()
    app.include_router(documents_module.router)
    app.dependency_overrides[get_request_context_hybrid] = _ctx
    app.dependency_overrides[get_db] = lambda: db
    return SimpleNamespace(client=TestClient(app), db=db, workspace=workspace, role=role, storage=storage)


def test_a_variant_is_never_invented_and_a_put_cannot_point_at_one():
    kit = get_brand_kit({"brand_kit": {"logo_path": f"{WS}/brand/logo.png"}})
    assert (kit["logo_dark_path"], kit["logo_mono_path"]) == ("", "")
    assert {"logo_dark_path", "logo_mono_path"} <= SERVER_MANAGED_FIELDS
    assert not {"logo_dark_path", "logo_mono_path"} & set(PATCH_FIELDS)
    merged = validate_brand_kit({"logo_dark_path": "../../etc/passwd", "logo_mono_path": "x"}, {"logo_dark_path": "keep"})
    assert (merged["logo_dark_path"], merged["logo_mono_path"]) == ("keep", "")


def test_each_variant_is_stored_beside_the_logo_under_the_logos_rules(storage):
    dark = bl.save_brand_logo(WS, png_bytes(400, 100), bl.LOGO_DARK_STEM)
    mono = bl.save_brand_logo(WS, png_bytes(400, 100), bl.LOGO_MONO_STEM)
    assert (dark, mono) == (f"{WS}/brand/logo-dark.png", f"{WS}/brand/logo-mono.png")
    assert (storage / dark).exists() and (storage / mono).exists()
    with pytest.raises(bl.BrandLogoError, match="The logo for dark backgrounds must be a PNG or JPEG"):
        bl.save_brand_logo(WS, b"GIF89a" + b"\x00" * 10, bl.LOGO_DARK_STEM)
    with pytest.raises(bl.BrandLogoError, match="The one-colour logo must be at most"):
        bl.save_brand_logo(WS, png_bytes(5000, 100), bl.LOGO_MONO_STEM)


@pytest.mark.parametrize("route, field, stem", [(DARK_ROUTE, "logo_dark_path", "logo-dark"), (MONO_ROUTE, "logo_mono_path", "logo-mono")])
def test_the_variant_routes_upload_serve_and_remove_it(api, route, field, stem):
    assert api.client.get(route).status_code == 404
    uploaded = api.client.post(route, files={"file": ("logo.png", png_bytes(600, 150), "image/png")})
    assert uploaded.status_code == 200, uploaded.text
    body = uploaded.json()
    assert body[field] == f"{WS}/brand/{stem}.png" and body[f"{field[:-5]}_route"] == route
    stored = api.workspace.settings["brand_kit"]
    assert stored[field] == body[field]
    assert stored["logo_url"] == "https://cdn.acme.com/logo.png"  # a variant supersedes no URL
    served = api.client.get(route)
    assert served.status_code == 200 and served.content == png_bytes(600, 150)
    assert served.headers["content-type"] == "image/png"
    refused = api.client.post(route, files={"file": ("logo.gif", b"GIF89a" + b"\x00" * 10, "image/gif")})
    assert refused.status_code == 422 and "PNG or JPEG" in refused.json()["detail"]
    removed = api.client.delete(route)
    assert removed.status_code == 200 and removed.json()[field] == ""
    assert not (api.storage / f"{WS}/brand/{stem}.png").exists()
    assert api.client.get(route).status_code == 404


@pytest.mark.parametrize("route", [DARK_ROUTE, MONO_ROUTE])
def test_only_a_workspace_manager_uploads_or_removes_a_variant(api, route):
    api.role["value"] = "editor"
    assert api.client.post(route, files={"file": ("logo.png", png_bytes(600, 150), "image/png")}).status_code == 403
    assert api.client.delete(route).status_code == 403
    assert not (api.storage / str(WS)).exists()
    api.db.commit.assert_not_called()
    assert api.client.get(route).status_code == 404  # reading is any member's


def test_the_six_variant_routes_are_in_the_committed_route_manifest():
    manifest = json.loads(MANIFEST.read_text())
    pairs = {(r["method"], r["path"]) for r in manifest["routes"]}
    for route in (DARK_ROUTE, MONO_ROUTE):
        assert {("GET", route), ("POST", route), ("DELETE", route)} <= pairs
    assert manifest["route_count"] == len(manifest["routes"])


# ---------------------------------------------------------------------------
# The PUT and the agent tool
# ---------------------------------------------------------------------------


def test_the_put_saves_every_new_field_and_get_answers_it(api):
    saved = api.client.put(KIT_ROUTE, json={
        "type_scale": {"h1": {"size_pt": 24, "line_pt": 30}},
        "spacing_unit_pt": 8,
        "page_margin_mm": 20,
        "logo_rules": {"letterhead_mm": 18},
        "currency": "eur",
        "date_style": "MMMM d, yyyy",
        "voice": {"tone": [{"word": "warm", "meaning": "friendly"}, "plain", "local"]},
    })
    assert saved.status_code == 200, saved.text
    kit = api.client.get(KIT_ROUTE).json()
    assert _step(kit, "h1") == (24, 30, 600) and _step(kit, "body") == PRD_SCALE["body"]
    assert (kit["spacing_unit_pt"], kit["page_margin_mm"], kit["logo_rules"]["letterhead_mm"]) == (8, 20, 18)
    assert (kit["currency"], kit["date_style"]) == ("EUR", "MMMM d, yyyy")
    assert kit["voice"]["tone"][0] == {"word": "warm", "meaning": "friendly"}
    refused = api.client.put(KIT_ROUTE, json={"type_scale": {"body": {"size_pt": 2}}})
    assert refused.status_code == 422
    (error,) = refused.json()["detail"]["errors"]
    assert error["loc"] == ["type_scale", "body", "size_pt"] and "5 to 96 pt" in error["msg"]


def test_the_tool_schema_names_the_steps_the_date_styles_and_the_tone_items():
    properties = get_action_registry().get("platform_update_brand_kit").parameters["properties"]
    new = {"type_scale", "spacing_unit_pt", "page_margin_mm", "logo_rules", "currency", "date_style"}
    assert new <= set(properties) and new <= set(PATCH_FIELDS)
    assert set(properties["type_scale"]["properties"]) == set(TYPE_STEPS)
    assert set(properties["type_scale"]["properties"]["h1"]["properties"]) == {"size_pt", "line_pt", "weight"}
    assert set(properties["logo_rules"]["properties"]) == set(BrandKit.model_fields["logo_rules"].annotation.model_fields)
    assert properties["date_style"]["enum"] == list(DATE_STYLES)
    tone = properties["voice"]["properties"]["tone"]["items"]
    assert set(tone["properties"]) == {"word", "meaning"} and tone["required"] == ["word"]


def test_an_agent_sets_the_scale_and_the_currency_through_the_tool(api):
    executor = PlatformActionExecutor(api.db, WS)
    with patch.object(PlatformActionExecutor, "_full_autonomy", return_value=False), patch.object(
        PlatformActionExecutor, "_caller_is_admin", return_value=True
    ), patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        result = asyncio.run(executor.execute("platform_update_brand_kit", {
            "type_scale": {"h2": {"size_pt": 16, "line_pt": 21}}, "currency": "GBP",
        }))
    assert result["success"] is True, result
    assert result["changed"] == ["currency", "type_scale"]
    kit = api.workspace.settings["brand_kit"]
    assert _step(kit, "h2") == (16, 21, 600) and kit["currency"] == "GBP"
