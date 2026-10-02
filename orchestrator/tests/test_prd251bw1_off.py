"""PRD-251B Wave 1, US-B106 — off means invisible (B3).

While Socials is off for a workspace — the platform master switch, or the workspace's
own — the workspace is shown nothing of it. Pinned, with the master off and then with
the workspace off (the two switches, in that order):

* ``GET /api/marketplace/packages`` leaves out the package whose slug is ``socials`` and
  lists the others; its deep link is not found; both are back on the next request once
  Socials is on (the seeder is untouched: no boot);
* the registry's ``build_prompt_summary``, ``build_filtered_prompt_summary`` and
  ``to_dispatcher_schema``, the pure ``lexical_rank`` and the index's eligible set leave
  out every action in the ``socials`` category, and no fallback re-admits one;
* ``GET /api/documents/brand-kit`` answers without ``social_handles``; a ``PUT`` while
  Socials is off leaves the stored handles exactly as they were, whether the payload
  lacks them or carries them;
* the check is fail-closed: a malformed workspace setting, a missing workspace or a
  master switch that cannot be read hides the actions and the package.

The marketplace and brand-kit routers run on a mini app over SQLite copies of
``workspaces`` and ``marketplace_packages``; the master switch's system setting and the
caller's role are stubbed, as in the S0.3b harness.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.document_brand_kit as brand_kit_api  # noqa: E402
import api.marketplace as marketplace_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.marketplace_packages import MarketplacePackage  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from core.seeds.seed_socials_package import SOCIALS_PACKAGE  # noqa: E402
from modules.documents.brand_kit import BRAND_KIT_SETTINGS_KEY  # noqa: E402
from modules.socials.settings import (  # noqa: E402
    SOCIALS_ACTION_CATEGORY,
    SOCIALS_PACKAGE_SLUG,
    socials_actions_hidden,
)
from modules.tools.discovery.action_registry import ActionDefinition, ActionRegistry, action_is_available  # noqa: E402
from modules.tools.discovery.action_semantic_index import ActionSemanticIndex  # noqa: E402
from modules.tools.discovery.hidden_categories import (  # noqa: E402
    exclude_kwargs,
    hidden_categories_for,
    hidden_categories_for_workspace,
    without_hidden,
)
from modules.tools.discovery.lexical_rank import lexical_rank  # noqa: E402
from tests.test_prd251_api import _ctx, _sqlite_copy  # noqa: E402

WS_ON = uuid.uuid4()
WS_OFF = uuid.uuid4()
WS_BAD = uuid.uuid4()
NOW = datetime(2026, 10, 2, 12, 0)
HANDLES = {"linkedin": "acme-inc", "twitter": "acme"}
OTHER_SLUG = "shopify-management"
PACKAGES = "/api/marketplace/packages"
BRAND_KIT = "/api/documents/brand-kit"
HIDDEN = (SOCIALS_ACTION_CATEGORY,)


def _workspace(ws_id, socials):
    return Workspace(
        id=ws_id, name=f"ws-{ws_id.hex[:6]}", plan="basic", plan_limits={},
        settings={"socials": socials, BRAND_KIT_SETTINGS_KEY: {"name": "Acme", "social_handles": dict(HANDLES)}},
        onboarding={}, created_at=NOW, updated_at=NOW,
    )


def _package(slug, name, showcase):
    return MarketplacePackage(
        id=uuid.uuid4(), slug=slug, name=name, description=f"{name} package", vertical_tags=[slug],
        matching={}, members=[], setup_manifest={}, showcase=showcase, created_at=NOW, updated_at=NOW,
    )


@pytest.fixture
def app(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    copies = sa.MetaData()
    _sqlite_copy(Workspace.__table__, copies)
    _sqlite_copy(MarketplacePackage.__table__, copies)
    copies.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all([
        _workspace(WS_ON, {"enabled": True}),
        _workspace(WS_OFF, {"enabled": False}),
        _workspace(WS_BAD, "yes"),  # malformed: not an object
        _package(SOCIALS_PACKAGE_SLUG, "Socials", True),
        _package(OTHER_SLUG, "Shopify Management", False),
    ])
    session.commit()

    state = SimpleNamespace(master="true", role="owner", ctx=_ctx(WS_ON), session=session)
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: state.master)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)

    fastapi_app = FastAPI()
    fastapi_app.include_router(marketplace_api.router)
    fastapi_app.include_router(brand_kit_api.router, prefix="/api/documents")
    fastapi_app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    fastapi_app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(fastapi_app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _slugs(app):
    resp = app.client.get(PACKAGES)
    assert resp.status_code == 200, resp.text
    return [package["slug"] for package in resp.json()]


def _stored_handles(app, ws_id=WS_ON):
    app.session.expire_all()
    return app.session.get(Workspace, ws_id).settings[BRAND_KIT_SETTINGS_KEY].get("social_handles")


def _off_states(app):
    """The two switches, in that order: master off (workspace on), then master on and the
    workspace off; each restored afterwards."""
    app.master, app.ctx = "false", _ctx(WS_ON)
    yield "master off"
    app.master, app.ctx = "true", _ctx(WS_OFF)
    yield "workspace off"
    app.master, app.ctx = "true", _ctx(WS_ON)


# ---------------------------------------------------------------------------
# The marketplace package
# ---------------------------------------------------------------------------


def test_the_socials_package_is_the_seeded_slug():
    assert SOCIALS_PACKAGE["slug"] == SOCIALS_PACKAGE_SLUG


def test_the_socials_package_is_listed_only_while_socials_is_on(app):
    assert _slugs(app) == [SOCIALS_PACKAGE_SLUG, OTHER_SLUG]  # showcased first
    assert app.client.get(f"{PACKAGES}/{SOCIALS_PACKAGE_SLUG}").status_code == 200

    for state in _off_states(app):
        assert _slugs(app) == [OTHER_SLUG], state
        assert app.client.get(f"{PACKAGES}/{SOCIALS_PACKAGE_SLUG}").status_code == 404, state
        assert app.client.get(f"{PACKAGES}/{OTHER_SLUG}").status_code == 200, state
        assert app.client.get(f"{PACKAGES}?showcase=true").json() == [] if state == "master off" else True

    # Back on: the next request lists it, with no boot and no reseed.
    assert _slugs(app) == [SOCIALS_PACKAGE_SLUG, OTHER_SLUG]
    assert [p["slug"] for p in app.client.get(f"{PACKAGES}?showcase=true").json()] == [SOCIALS_PACKAGE_SLUG]


def test_a_malformed_workspace_setting_hides_the_package(app):
    app.ctx = _ctx(WS_BAD)
    assert _slugs(app) == [OTHER_SLUG]
    assert app.client.get(f"{PACKAGES}/{SOCIALS_PACKAGE_SLUG}").status_code == 404


# ---------------------------------------------------------------------------
# The brand kit's handles
# ---------------------------------------------------------------------------


def test_the_brand_kit_keeps_its_handles_to_itself_while_socials_is_off(app):
    assert app.client.get(BRAND_KIT).json()["social_handles"] == HANDLES

    for state in _off_states(app):
        ws_id = WS_ON if state == "master off" else WS_OFF
        shown = app.client.get(BRAND_KIT).json()
        assert "social_handles" not in shown and shown["name"] == "Acme", state

        # A PUT without the handles keeps them; a PUT carrying them changes nothing of them.
        saved = app.client.put(BRAND_KIT, json={"name": "Acme Ltd"})
        assert saved.status_code == 200, saved.text
        assert "social_handles" not in saved.json() and saved.json()["name"] == "Acme Ltd"
        assert _stored_handles(app, ws_id) == HANDLES, state
        meddled = app.client.put(BRAND_KIT, json={"tagline": "Hello", "social_handles": {"twitter": "someone_else"}})
        assert meddled.status_code == 200, meddled.text
        assert "social_handles" not in meddled.json()
        assert _stored_handles(app, ws_id) == HANDLES, state

    # On again: the handles are shown and editable, as before.
    assert app.client.get(BRAND_KIT).json()["social_handles"] == HANDLES
    changed = app.client.put(BRAND_KIT, json={"social_handles": {"linkedin": "acme-inc", "twitter": "acme_hq"}})
    assert changed.status_code == 200, changed.text
    assert changed.json()["social_handles"] == {"linkedin": "acme-inc", "twitter": "acme_hq"}
    assert _stored_handles(app) == {"linkedin": "acme-inc", "twitter": "acme_hq"}


# ---------------------------------------------------------------------------
# The check, and the categories it hides
# ---------------------------------------------------------------------------


def test_socials_actions_hidden_reads_both_switches_and_fails_closed(app, monkeypatch):
    on, off, bad = (app.session.get(Workspace, ws) for ws in (WS_ON, WS_OFF, WS_BAD))
    assert socials_actions_hidden(on) is False
    assert socials_actions_hidden(off) is True and socials_actions_hidden(bad) is True
    assert socials_actions_hidden(None) is True
    app.master = "false"
    assert socials_actions_hidden(on) is True
    app.master = "true"
    assert socials_actions_hidden(on) is False  # no restart

    def cannot_read(category, key):
        raise RuntimeError("pool exhausted")

    monkeypatch.setattr(socials_settings, "read_system_setting", cannot_read)
    assert socials_actions_hidden(on) is True


def test_hidden_categories_resolve_from_the_workspace_or_its_id(app):
    on = app.session.get(Workspace, WS_ON)
    assert hidden_categories_for(on) == ()
    assert hidden_categories_for(None) == HIDDEN
    assert hidden_categories_for_workspace(WS_ON, app.session) == ()
    assert hidden_categories_for_workspace(str(WS_ON), app.session) == ()
    assert hidden_categories_for_workspace(WS_OFF, app.session) == HIDDEN
    assert hidden_categories_for_workspace(WS_BAD, app.session) == HIDDEN
    assert hidden_categories_for_workspace(uuid.uuid4(), app.session) == HIDDEN  # no such workspace
    assert hidden_categories_for_workspace(None) == HIDDEN
    assert hidden_categories_for_workspace("", app.session) == HIDDEN
    assert hidden_categories_for_workspace("not-a-uuid", app.session) == HIDDEN

    def broken_get(model, key):
        raise RuntimeError("connection dropped")

    assert hidden_categories_for_workspace(WS_ON, SimpleNamespace(get=broken_get)) == HIDDEN  # a failed read hides
    assert exclude_kwargs(()) == {} and exclude_kwargs(None) == {}
    assert exclude_kwargs(HIDDEN) == {"exclude_categories": HIDDEN}


# ---------------------------------------------------------------------------
# The catalogs, the enum and the shortlists
# ---------------------------------------------------------------------------


def _action(name, category, promoted=False):
    return ActionDefinition(
        name=name, description=f"{name.replace('_', ' ')} for the workspace", category=category,
        parameters={"type": "object", "properties": {}, "required": []}, promoted=promoted,
        tags=[category], examples=[f"please {name.replace('_', ' ')}"],
    )


@pytest.fixture
def registry():
    reg = ActionRegistry()
    reg._initialized = True  # no live registrar
    for action in (
        _action("platform_create_social_post", "socials"),
        _action("platform_submit_social_post", "socials", promoted=True),
        _action("platform_list_agents", "agents"),
        _action("platform_list_documents", "documents", promoted=True),
    ):
        reg.register(action)
    return reg


def _names(summary):
    return [line.split("`")[1] for line in summary.splitlines() if line.startswith("- `")]


def _enum(schema):
    return schema["function"]["parameters"]["properties"]["action"].get("enum", [])


def test_the_summaries_leave_a_hidden_category_out(registry):
    plain = registry.build_prompt_summary(exclude_admin=True)
    assert "platform_create_social_post" in plain and "### Socials" in plain
    hidden = registry.build_prompt_summary(exclude_admin=True, exclude_categories=HIDDEN)
    assert _names(hidden) == ["platform_list_documents", "platform_list_agents"]
    assert "Socials" not in hidden and "social" not in hidden

    everything = [a.name for a in registry.get_all()]
    filtered = registry.build_filtered_prompt_summary(everything, exclude_admin=True, exclude_categories=HIDDEN)
    assert _names(filtered) == ["platform_list_documents", "platform_list_agents"]
    only_hidden = registry.build_filtered_prompt_summary(
        ["platform_create_social_post", "platform_submit_social_post"], exclude_categories=HIDDEN
    )
    assert _names(only_hidden) == []
    # Nothing hidden: byte-identical to before.
    assert registry.build_prompt_summary(exclude_admin=True, exclude_categories=()) == plain


def test_the_dispatcher_enum_never_readmits_a_hidden_category(registry, monkeypatch):
    import modules.tools.turn_narrowing as turn_narrowing

    # The narrowed enum itself (F025's cache-stable mode would ship the whole eligible set).
    monkeypatch.setattr(turn_narrowing, "enum_is_cache_stable", lambda: False)
    monkeypatch.setattr(turn_narrowing, "publish_narrowed_actions", lambda names: None)
    assert _enum(registry.to_dispatcher_schema(exclude_promoted=False)) == [
        "platform_create_social_post", "platform_list_agents", "platform_list_documents", "platform_submit_social_post",
    ]
    hidden = registry.to_dispatcher_schema(exclude_promoted=False, exclude_categories=HIDDEN)
    assert _enum(hidden) == ["platform_list_agents", "platform_list_documents"]
    # An allow-list of hidden names empties the intersection; the fallback is the visible set.
    fallen_back = registry.to_dispatcher_schema(
        exclude_promoted=False, allowed_names=["platform_create_social_post"], exclude_categories=HIDDEN
    )
    assert _enum(fallen_back) == ["platform_list_agents", "platform_list_documents"]
    pinned = registry.to_dispatcher_schema(
        exclude_promoted=True, allowed_names=["platform_submit_social_post", "platform_list_documents"],
        allow_promoted_in_allowlist=True, exclude_categories=HIDDEN,
    )
    assert _enum(pinned) == ["platform_list_documents"]


def test_the_shortlists_leave_a_hidden_category_out(registry):
    actions = registry.get_all()
    query = "create a social post for the workspace"
    assert [n for n, _ in lexical_rank(query, actions)][:1] == ["platform_create_social_post"]
    ranked = [n for n, _ in lexical_rank(query, actions, exclude_categories=HIDDEN)]
    assert ranked and not any("social" in n for n in ranked)
    assert [a.name for a in without_hidden(actions, HIDDEN)] == ["platform_list_agents", "platform_list_documents"]
    assert without_hidden(actions, ()) == actions

    index = ActionSemanticIndex.__new__(ActionSemanticIndex)  # no embedding manager: the eligible set only
    index._registry = registry
    assert {a.name for a in index._eligible_actions(False, False, exclude_categories=HIDDEN)} == {
        "platform_list_agents", "platform_list_documents",
    }
    shortlist = index.lexical_rank(query, exclude_admin=False, exclude_promoted=False, exclude_categories=HIDDEN)
    assert shortlist and not any("social" in n for n in shortlist)


def test_the_real_registry_hides_every_socials_action():
    from modules.tools.discovery.action_registry import get_action_registry

    registry = get_action_registry()
    socials = [a.name for a in registry.get_by_category(SOCIALS_ACTION_CATEGORY)]
    assert len(socials) >= 5, socials  # US-116's draft tools
    hidden = registry.build_prompt_summary(exclude_admin=True, exclude_promoted=False, exclude_categories=HIDDEN)
    assert not any(name in hidden for name in socials)
    assert not set(socials) & set(_enum(registry.to_dispatcher_schema(exclude_promoted=False, exclude_categories=HIDDEN)))
    shown = [a.name for a in registry.get_by_category(SOCIALS_ACTION_CATEGORY) if action_is_available(a)]
    if shown:  # where the Socials tools can run, the plain catalog still describes them
        assert shown[0] in registry.build_prompt_summary(exclude_admin=True, exclude_promoted=False)
