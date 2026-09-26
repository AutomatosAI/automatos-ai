"""PRD-251 Wave 1, US-115 — agents read and change the brand kit.

Agents could not touch the brand kit: only the REST routes could
(``api/document_brand_kit.py``). ``platform_get_brand_kit`` and
``platform_update_brand_kit`` are thin wrappers over the functions those routes
call (``modules/documents/brand_kit.py``). Pins:

* **Read**: the tool's kit is what ``GET /api/documents/brand-kit`` returns, the
  US-108 fields included, and its suggestions are the ones the Brand Kit form
  prefills from, less the signed-in user an agent is not.
* **Refused means nothing saved**: an invalid hex colour comes back with the
  validator's message, the same errors the PUT's 422 carries, and nothing is
  written. A stored file (the logo, the mark, the fonts), a misspelt field or a
  non-http(s) logo URL is refused too, never dropped.
* **One writer**: an update through the tool is what the GET then returns; the
  PUT and the tool both go through ``update_brand_kit`` and ``save_brand_kit``,
  and no route assigns the workspace settings itself.
* **Registered and routed**: both tools are in the registry, mapped in
  ``PlatformActionExecutor._handlers`` and dispatched through ``execute`` here;
  the write tool takes the permission level of the comparable workspace-setting
  writers, and its schema is exactly the kit's patch fields.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import uuid
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

import api.document_brand_kit as brand_kit_routes  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.documents.brand_kit as brand_kit  # noqa: E402
import modules.tools.discovery.handlers_documents as handlers_documents  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.tools.discovery.action_registry import get_action_registry  # noqa: E402
from modules.tools.discovery.platform_executor import PlatformActionExecutor  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000000e5")
KIT_ROUTE = "/api/documents/brand-kit"
HEX_RULE = "must be a hex color such as #1a1a2e or #abc"

STORED_KIT = {
    "name": "Acme",
    "tagline": "Made well",
    "primary_color": "#0055aa",
    "font_family": "Inter, sans-serif",
    "logo_path": f"{WS}/brand/logo.png",
    "company": {"name": "Acme Ltd", "email": "hello@acme.com"},
    # US-108's fields
    "heading_font": '"Brand Display", serif',
    "logo_mark_url": "https://cdn.acme.com/mark.png",
    "social_handles": {"twitter": "acme", "linkedin": "acme-inc"},
    "voice": {"tone": ["warm", "plain", "confident"], "banned_phrases": ["game-changer"]},
}


def _ctx():
    return RequestContext(
        workspace_id=WS,
        user=UserContext(id="owner-1", clerk_user_id="clerk-owner-1", system_role="user"),
        auth_type="clerk",
    )


@pytest.fixture
def api(monkeypatch):
    """The documents router (the brand kit routes ride it) and the platform tools
    over one workspace held in memory."""
    import api.document_generation as documents_module

    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(brand_kit_routes, "resolve_user_pk", lambda db, ctx: None)
    workspace = SimpleNamespace(id=WS, name="Acme Workspace", settings={"brand_kit": json.loads(json.dumps(STORED_KIT))})
    db = MagicMock()
    # Workspace lookups: query(Workspace).filter(...).first()
    db.query.return_value.filter.return_value.first.return_value = workspace
    # The latest business profile: query(BusinessProfile).filter(...).order_by(...).first()
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
    app = FastAPI()
    app.include_router(documents_module.router)
    app.dependency_overrides[get_request_context_hybrid] = _ctx
    app.dependency_overrides[get_db] = lambda: db
    return SimpleNamespace(client=TestClient(app), db=db, workspace=workspace)


def _dispatch(db, action, params, *, for_an_admin=True):
    """Run a platform action the way an agent's call runs: through the executor.
    platform_update_brand_kit is an owner's or admin's (PUT /brand-kit is
    workspace:manage, F151), so the call is made for an admin unless a test says not."""
    executor = PlatformActionExecutor(db, WS)
    with patch.object(PlatformActionExecutor, "_full_autonomy", return_value=False), patch.object(
        PlatformActionExecutor, "_caller_is_admin", return_value=for_an_admin
    ), patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute(action, params))


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def test_the_get_tool_returns_what_the_rest_get_returns(api):
    rest = api.client.get(KIT_ROUTE)
    assert rest.status_code == 200, rest.text
    result = _dispatch(api.db, "platform_get_brand_kit", {})
    assert result["success"] is True, result
    assert result["brand_kit"] == rest.json()
    # The US-108 fields reach the agent as the UI sees them.
    kit = result["brand_kit"]
    assert kit["heading_font"] == STORED_KIT["heading_font"]
    assert kit["logo_mark_url"] == STORED_KIT["logo_mark_url"]
    assert kit["social_handles"] == STORED_KIT["social_handles"]
    assert kit["voice"] == STORED_KIT["voice"]
    assert kit["font_files"] == [] and kit["logo_path"] == STORED_KIT["logo_path"]


def test_the_get_tool_carries_the_suggestions_the_brand_kit_form_prefills_from(api):
    profile = SimpleNamespace(
        company_name="Acme Ltd", domain="acme.com",
        brands=[{"logo_url": "https://acme.com/logo.png"}], voice_notes="Build better.\nMore",
    )
    api.db.query.return_value.filter.return_value.order_by.return_value.first.return_value = profile
    suggestions = _dispatch(api.db, "platform_get_brand_kit", {})["suggestions"]
    assert suggestions["name"] == {"value": "Acme Ltd", "source": "business_profile"}
    assert suggestions["website"] == {"value": "https://acme.com", "source": "business_profile"}
    assert suggestions["logo_url"] == {"value": "https://acme.com/logo.png", "source": "business_profile"}
    # Without a signed-in user the form's candidates are the tool's.
    rest = api.client.get(f"{KIT_ROUTE}/suggestions")
    assert rest.status_code == 200, rest.text
    assert rest.json() == {"suggestions": suggestions}


# ---------------------------------------------------------------------------
# Refused means nothing saved
# ---------------------------------------------------------------------------


def test_an_invalid_hex_colour_is_refused_with_the_validators_message_and_nothing_is_saved(api):
    before = json.loads(json.dumps(api.workspace.settings))
    result = _dispatch(api.db, "platform_update_brand_kit", {"primary_color": "blue", "name": "Acme Studio"})
    assert result["success"] is False, result
    assert HEX_RULE in result["error"] and "primary_color" in result["error"]
    assert "nothing saved" in result["error"]
    assert [tuple(error["loc"]) for error in result["errors"]] == [("primary_color",)]
    api.db.commit.assert_not_called()
    assert api.workspace.settings == before
    # The PUT refuses the same change with the same errors.
    rest = api.client.put(KIT_ROUTE, json={"primary_color": "blue", "name": "Acme Studio"})
    assert rest.status_code == 422, rest.text
    assert rest.json()["detail"]["errors"] == json.loads(json.dumps(result["errors"]))
    api.db.commit.assert_not_called()
    assert api.workspace.settings == before


@pytest.mark.parametrize(
    "params, reason",
    [
        ({"logo_path": "elsewhere/brand/logo.png"}, "cannot set logo_path"),
        ({"font_files": [], "name": "Acme Studio"}, "cannot set font_files"),
        ({"logo_mark_path": "../../etc/passwd"}, "cannot set logo_mark_path"),
        ({"primary_colour": "#0055aa"}, "cannot set primary_colour"),
        ({"logo_mark_url": "javascript:alert(1)"}, "logo_mark_url must be an http(s) URL"),
        ({"logo_url": "file:///etc/passwd"}, "logo_url must be an http(s) URL"),
        ({"social_handles": {"twitter": "acme-hq"}}, "the twitter handle"),
        ({"voice": {"tone": ["warm", "plain"]}}, "give 3 to 5 tone words"),
        ({}, "Nothing to change"),
        ({"_agent_id": 7, "_turn_id": "t-1", "name": None}, "Nothing to change"),
    ],
)
def test_a_change_the_kit_does_not_take_is_refused_and_nothing_is_saved(api, params, reason):
    before = json.loads(json.dumps(api.workspace.settings))
    result = _dispatch(api.db, "platform_update_brand_kit", params)
    assert result["success"] is False, result
    assert reason in result["error"], result["error"]
    api.db.commit.assert_not_called()
    assert api.workspace.settings == before


# ---------------------------------------------------------------------------
# One writer
# ---------------------------------------------------------------------------


def test_an_update_through_the_tool_is_what_get_returns_and_the_put_shares_its_writer(api, monkeypatch):
    calls = []

    def spy(name, real):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return real(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(brand_kit, "update_brand_kit", spy("update", brand_kit.update_brand_kit))
    monkeypatch.setattr(brand_kit, "save_brand_kit", spy("save", brand_kit.save_brand_kit))

    result = _dispatch(api.db, "platform_update_brand_kit", {
        "_agent_id": 7,  # server-injected keys are not kit fields
        "accent_color": "#e4572e",
        "social_handles": {"twitter": "@acme_hq", "instagram": "acme.studio"},
        "voice": {"banned_phrases": ["synergy"]},
        "company": {"phone": "+44 20 7946 0000"},
    })
    assert result["success"] is True, result
    assert result["changed"] == ["accent_color", "company", "social_handles", "voice"]
    assert calls == ["update", "save"]
    api.db.commit.assert_called_once()

    rest = api.client.get(KIT_ROUTE).json()
    assert rest == result["brand_kit"]
    assert rest["accent_color"] == "#e4572e"
    # A partial merge: company and voice merge key by key, the handles map is replaced,
    # and every field the change left out keeps its value.
    assert rest["company"] == {**brand_kit.CompanyContact().model_dump(), **STORED_KIT["company"], "phone": "+44 20 7946 0000"}
    assert rest["voice"] == {"tone": STORED_KIT["voice"]["tone"], "banned_phrases": ["synergy"]}
    assert rest["social_handles"] == {"twitter": "acme_hq", "instagram": "acme.studio"}
    for field in ("name", "tagline", "primary_color", "font_family", "heading_font", "logo_mark_url", "logo_path"):
        assert rest[field] == STORED_KIT[field], field

    # The PUT goes through the same two functions.
    put = api.client.put(KIT_ROUTE, json={"tagline": "Made better"})
    assert put.status_code == 200, put.text
    assert calls == ["update", "save", "update", "save"]
    assert api.client.get(KIT_ROUTE).json()["tagline"] == "Made better"


def test_a_call_made_for_no_admin_cannot_change_the_kit(api):
    result = _dispatch(api.db, "platform_update_brand_kit", {"primary_color": "#112233"}, for_an_admin=False)
    assert result["success"] is False and result.get("permission_denied") is True


def test_only_brand_kit_py_assigns_the_workspace_settings():
    """The routes (the PUT, the logo, mark and font uploads and deletes) and the tool
    save through save_brand_kit; neither assigns the settings itself."""
    assert ".settings =" not in inspect.getsource(brand_kit_routes)
    for handler in (handlers_documents.get_brand_kit_tool, handlers_documents.update_brand_kit_tool):
        assert ".settings =" not in inspect.getsource(handler)
    assert "workspace.settings = " in inspect.getsource(brand_kit.save_brand_kit)
    assert "_persist_kit" not in inspect.getsource(brand_kit_routes)


# ---------------------------------------------------------------------------
# Registered and routed
# ---------------------------------------------------------------------------


def test_both_tools_are_registered_and_routed_to_their_handlers():
    registry = get_action_registry()
    read, write = registry.get("platform_get_brand_kit"), registry.get("platform_update_brand_kit")
    assert read is not None and write is not None
    assert (read.category, read.permission_level, read.requires_confirmation) == ("documents", "read", False)
    assert (write.category, write.permission_level) == ("documents", "write")
    # The write tool is gated like the workspace-settings writer: an owner's or
    # admin's, as PUT /brand-kit is (F147/F151).
    for comparable in ("platform_update_workspace_settings",):
        other = registry.get(comparable)
        assert (write.permission_level, write.requires_confirmation, write.admin_only, write.super_admin_only) == (
            other.permission_level, other.requires_confirmation, other.admin_only, other.super_admin_only
        ), comparable
    handlers = PlatformActionExecutor(None, None)._handlers
    for action, name in (("platform_get_brand_kit", "get_brand_kit_tool"), ("platform_update_brand_kit", "update_brand_kit_tool")):
        handler = handlers[action]
        assert (handler.__module__, handler.__name__) == ("modules.tools.discovery.handlers_documents", name), action


def test_the_update_schema_is_the_kits_patch_fields_and_never_a_stored_file():
    properties = get_action_registry().get("platform_update_brand_kit").parameters["properties"]
    assert set(properties) == set(brand_kit.PATCH_FIELDS)
    assert {"heading_font", "logo_mark_url", "social_handles", "voice"} <= set(properties)  # US-108
    assert not set(properties) & brand_kit.SERVER_MANAGED_FIELDS
    assert set(properties["voice"]["properties"]) == set(brand_kit.BrandVoice.model_fields)
    assert set(properties["company"]["properties"]) == set(brand_kit.CompanyContact.model_fields)
    assert get_action_registry().get("platform_update_brand_kit").parameters["required"] == []
