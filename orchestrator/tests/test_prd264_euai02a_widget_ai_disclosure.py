"""PRD-264 EUAI-02a — the widget config carries an AI notice no client can drop.

Article 50(1): a visitor is told they are dealing with an AI before the first
answer. The server builds the notice (``modules/policy/ai_disclosure.py``):
a fixed badge from D17's per-locale table, and a first-turn line a deployer
may reword only if it still contains the locale's term. Every public widget
config carries it, including a workspace with no widget settings at all, and
the Sites settings save refuses a line that doesn't say AI with HTTP 422.

Nothing here is gated by ``AUTH_EDITION``; the ``edition`` fixture runs each
case in both editions to prove it.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from config import config as app_config  # noqa: E402

SHOP = "Harbour Coffee"
NO_TERM = "Hi there, how can I help you today?"
PT_LINE = "Olá! Sou o assistente de IA da Harbour Coffee."


@pytest.fixture(params=("local", "saas"))
def edition(request, monkeypatch):
    monkeypatch.setattr(app_config, "AUTH_EDITION", request.param)
    return request.param


def _notice(line: str, badge: str = "AI", locale: str = "en") -> dict:
    return {"badge": badge, "line": line, "locale": locale}


# ── the badge table and the line rule (D17) ────────────────────────────────

@pytest.mark.parametrize("locale, badge", [
    ("en", "AI"), ("en-GB", "AI"),
    ("pt", "IA"), ("pt-BR", "IA"), ("es", "IA"), ("fr-FR", "IA"), ("it", "IA"),
    ("de", "KI"), ("de_AT", "KI"),
    ("ja", "AI"), ("zz-ZZ", "AI"), ("", "AI"), (None, "AI"), (7, "AI"),
])
def test_the_badge_comes_from_one_table(locale, badge):
    from modules.policy.ai_disclosure import badge_for

    assert badge_for(locale) == badge


def test_the_default_line_names_the_site_or_says_ai_without_one():
    from modules.policy.ai_disclosure import default_line

    assert default_line(SHOP) == f"You are chatting with an AI assistant for {SHOP}."
    assert default_line("  ") == "You are chatting with an AI assistant."
    assert default_line(None) == "You are chatting with an AI assistant."


def test_a_line_without_the_term_is_refused_with_the_prds_message():
    from modules.policy.ai_disclosure import AIDisclosureRejected, check_line

    with pytest.raises(AIDisclosureRejected) as refused:
        check_line(NO_TERM, "en")
    assert str(refused.value) == "The AI notice must say it is an AI (for this language: 'AI')."
    with pytest.raises(AIDisclosureRejected) as refused_de:
        check_line("You are chatting with an AI assistant.", "de-DE")
    assert str(refused_de.value) == "The AI notice must say it is an AI (for this language: 'KI')."


@pytest.mark.parametrize("line, locale", [
    (PT_LINE, "pt-PT"),
    ("Hola, soy un asistente de IA.", "es"),
    ("Bonjour ! Je suis l'IA de la boutique.", "fr"),
    ("Hallo, hier antwortet ein KI-Assistent.", "de"),
    ("AI-powered help for Harbour Coffee.", "en"),
])
def test_a_line_with_the_locales_term_as_a_whole_word_is_accepted(line, locale):
    from modules.policy.ai_disclosure import check_line

    assert check_line(f"  {line}  ", locale) == line


@pytest.mark.parametrize("line", [
    "you are chatting with an ai assistant",  # the term as written: AI
    "MAIL us any time",
    "Ask our AIs anything",
])
def test_the_term_must_stand_as_a_whole_word_in_its_case(line):
    from modules.policy.ai_disclosure import AIDisclosureRejected, check_line

    with pytest.raises(AIDisclosureRejected):
        check_line(line, "en")


def test_a_line_that_is_not_text_or_too_long_is_refused():
    from modules.policy.ai_disclosure import MAX_LINE_CHARS, AIDisclosureRejected, check_line

    with pytest.raises(AIDisclosureRejected):
        check_line(["AI"], "en")
    with pytest.raises(AIDisclosureRejected):
        check_line("AI " * MAX_LINE_CHARS, "en")
    assert check_line(None, "en") is None
    assert check_line("   ", "en") is None


def test_the_badge_is_not_an_editable_field():
    from modules.policy.ai_disclosure import AIDisclosureRejected, validate_settings

    with pytest.raises(AIDisclosureRejected):
        validate_settings({"ai_disclosure": {"badge": "Bot", "line": "An AI helps here."}})
    with pytest.raises(AIDisclosureRejected):
        validate_settings({"ai_disclosure": "An AI helps here."})
    validate_settings({"ai_disclosure": {"line": "An AI helps here."}})
    validate_settings({"callback": {"enabled": True}})


# ── the public widget config (GET /api/widgets/config and the session) ─────

def _db(site=None, workspace=None):
    """A db whose Site query returns ``site`` and Workspace query ``workspace``."""
    db = MagicMock()

    def query(model):
        chain = MagicMock()
        if "Site" in repr(model):
            chain.filter.return_value.order_by.return_value.first.return_value = site
        else:
            chain.filter.return_value.first.return_value = workspace
        return chain

    db.query.side_effect = query
    return db


def _site(settings=None, display_name=SHOP, workspace_id=None):
    return NS(
        id=uuid4(), workspace_id=workspace_id or uuid4(), type="custom", external_id=None,
        display_name=display_name, status="active", settings=settings if settings is not None else {},
        capabilities={}, secrets=None,
        created_at=datetime(2026, 10, 19, tzinfo=timezone.utc), updated_at=datetime(2026, 10, 19, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("settings", [None, {}])
def test_the_config_carries_the_notice_when_the_workspace_has_no_widget_settings(edition, settings):
    """With no Site, the notice is the unnamed line: the workspace's name is
    never a public name (FR-15)."""
    from api.widgets.config import resolve_widget_config

    workspace = NS(id=uuid4(), name="Harbour Coffee Ltd", settings=settings)
    config = resolve_widget_config(_db(site=None, workspace=workspace), workspace.id)
    assert config == {"ai_disclosure": _notice("You are chatting with an AI assistant.")}
    assert "Harbour Coffee Ltd" not in str(config)


def test_the_config_endpoint_always_returns_the_notice(edition):
    from api.widgets.config import get_widget_config

    workspace_id = uuid4()
    response = get_widget_config(auth=NS(workspace_id=workspace_id), db=_db(site=None, workspace=None))
    assert response.config == {"ai_disclosure": _notice("You are chatting with an AI assistant.")}


def test_the_site_name_wins_over_the_workspace_name(edition):
    from api.widgets.config import resolve_widget_config

    workspace = NS(id=uuid4(), name="Harbour Holdings", settings={"cart_idle": {"enabled": True}})
    config = resolve_widget_config(_db(site=_site(settings={}), workspace=workspace), workspace.id)
    assert config == {
        "cart_idle": {"enabled": True},
        "ai_disclosure": _notice(f"You are chatting with an AI assistant for {SHOP}."),
    }


# ── FR-15: the workspace name never reaches public widget output ──────────

PERSONAL_WORKSPACE_NAME = "jane@example.com's Workspace"
_LEAKS = (PERSONAL_WORKSPACE_NAME, "jane", "example.com", "@")


def _assert_no_leak(output) -> None:
    text = str(output)
    for leaked in _LEAKS:
        assert leaked not in text, leaked


@pytest.mark.parametrize("settings", [None, {}, {"cart_idle": {"enabled": True}}])
def test_a_workspace_named_after_its_owner_never_reaches_the_public_config(edition, settings):
    """An auto-created workspace is "{name or email}'s Workspace"; with no Site
    the resolved config, ``GET /api/widgets/config`` and the PRD-007 builder
    all carry the unnamed line and neither the email nor the name. (The
    session token and the platform tool have their own tests.)"""
    from api.widgets.config import build_widget_config, get_widget_config, resolve_widget_config

    workspace = NS(id=uuid4(), name=PERSONAL_WORKSPACE_NAME, settings=settings)
    unnamed = _notice("You are chatting with an AI assistant.")

    config = resolve_widget_config(_db(site=None, workspace=workspace), workspace.id)
    assert config["ai_disclosure"] == unnamed
    _assert_no_leak(config)

    response = get_widget_config(auth=NS(workspace_id=workspace.id), db=_db(site=None, workspace=workspace))
    assert response.config["ai_disclosure"] == unnamed
    _assert_no_leak(response.config)

    assert build_widget_config(workspace)["ai_disclosure"] == unnamed
    _assert_no_leak(build_widget_config(workspace))


def test_the_session_token_config_never_carries_the_workspace_name(edition, monkeypatch):
    """The session token exchange (``api/widgets/session.py``) hands the
    browser ``widget_config``: the unnamed notice, with neither the owner's
    email nor the workspace name."""
    import asyncio
    import secrets

    import api.widgets.session as session_mod

    workspace = NS(id=uuid4(), name=PERSONAL_WORKSPACE_NAME, settings={})
    key_record = NS(
        id=uuid4(), workspace_id=workspace.id, key_type="server", permissions=[],
        default_agent_id=None, team=None,
    )
    monkeypatch.setattr(session_mod, "WIDGET_TOKEN_SECRET", secrets.token_hex(32))  # throwaway, per run
    monkeypatch.setattr(session_mod, "ApiKeyService", NS(validate_api_key=lambda db, raw_key: key_record))

    response = asyncio.run(session_mod.exchange_session_token(
        request=session_mod.SessionTokenRequest(api_key="throwaway-test-key"),
        db=_db(site=None, workspace=workspace),
    ))
    assert response.widget_config == {"ai_disclosure": _notice("You are chatting with an AI assistant.")}
    _assert_no_leak(response.widget_config)


@pytest.mark.parametrize("display_name", [
    PERSONAL_WORKSPACE_NAME,       # a Site backfilled from an auto-created workspace
    "Jane Doe's Workspace",
    "jane@example.com",
])
def test_a_site_named_after_the_owner_is_not_shown(edition, display_name):
    """Sites created from a workspace copy its name (the PRD-008-A backfill,
    ``ensure_shopify_site``), so the Site name passes the same FR-15 test."""
    from api.widgets.config import resolve_widget_config

    site = _site(settings={"cart_idle": {"enabled": True}}, display_name=display_name)
    config = resolve_widget_config(_db(site=site), site.workspace_id)
    assert config["ai_disclosure"] == _notice("You are chatting with an AI assistant.")
    assert display_name not in str(config)
    assert "Jane" not in str(config) and "jane" not in str(config)


def test_a_site_with_settings_but_no_name_does_not_fall_back_to_the_workspace(edition):
    from api.widgets.config import resolve_widget_config

    workspace = NS(id=uuid4(), name="Harbour Holdings", settings={})
    site = _site(settings={"cart_idle": {"enabled": True}}, display_name="", workspace_id=workspace.id)
    config = resolve_widget_config(_db(site=site, workspace=workspace), workspace.id)
    assert config["ai_disclosure"] == _notice("You are chatting with an AI assistant.")
    assert "Harbour Holdings" not in str(config)


def test_a_badge_or_unknown_field_in_settings_is_never_copied(edition):
    from api.widgets.config import resolve_widget_config

    site = _site(settings={"locale": "pt-PT", "ai_disclosure": {"badge": "Bot", "line": "Olá!", "hidden": True}})
    config = resolve_widget_config(_db(site=site), site.workspace_id)
    assert config["ai_disclosure"] == _notice(f"You are chatting with an AI assistant for {SHOP}.", "IA", "pt")


def test_a_stored_line_that_lost_its_term_is_not_served(edition):
    """A line saved before the rule, or under an older locale, falls back."""
    from api.widgets.config import resolve_widget_config

    site = _site(settings={"locale": "de", "ai_disclosure": {"line": "An AI assistant answers here."}})
    config = resolve_widget_config(_db(site=site), site.workspace_id)
    assert config["ai_disclosure"] == _notice(f"You are chatting with an AI assistant for {SHOP}.", "KI", "de")


def test_a_reworded_line_with_the_term_is_served(edition):
    from api.widgets.config import resolve_widget_config

    site = _site(settings={"locale": "pt-PT", "ai_disclosure": {"line": PT_LINE}})
    config = resolve_widget_config(_db(site=site), site.workspace_id)
    assert config["ai_disclosure"] == _notice(PT_LINE, "IA", "pt")


def test_ai_disclosure_is_a_public_widget_config_key():
    from api.widgets.config import PUBLIC_WIDGET_CONFIG_KEYS

    assert "ai_disclosure" in PUBLIC_WIDGET_CONFIG_KEYS


# ── the Sites settings save refuses a line that doesn't say AI (HTTP 422) ──

def _sites_client(monkeypatch, site):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import api.sites as sites_mod
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(sites_mod.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=site.workspace_id,
        user=UserContext(id="test-user", email="owner@harbour.test"),
        auth_type="anonymous",
    )

    def _override_db():
        yield MagicMock()

    app.dependency_overrides[get_db] = _override_db
    saved: list = []

    def _get_site(db, workspace_id, site_id):
        return site if workspace_id == site.workspace_id and site_id == site.id else None

    def _update(db, *, workspace_id, site_id, settings_patch):
        saved.append(settings_patch)
        return NS(**{**vars(site), "settings": {**site.settings, **settings_patch}})

    def _create(db, *, workspace_id, type, display_name, external_id=None, settings=None):
        saved.append(settings)
        return site

    monkeypatch.setattr(sites_mod, "get_site", _get_site)
    monkeypatch.setattr(sites_mod, "update_site_settings", _update)
    monkeypatch.setattr(sites_mod, "create_site", _create)
    monkeypatch.setattr(sites_mod, "public_site_dict", lambda row: {"id": str(row.id), "settings": row.settings})
    return TestClient(app), saved


def test_a_save_without_the_term_is_refused_with_422(monkeypatch, edition):
    site = _site()
    client, saved = _sites_client(monkeypatch, site)
    resp = client.patch(f"/api/sites/{site.id}/settings", json={"settings": {"ai_disclosure": {"line": NO_TERM}}})
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"] == "The AI notice must say it is an AI (for this language: 'AI')."
    assert saved == []


def test_a_portuguese_line_with_ia_is_accepted(monkeypatch, edition):
    site = _site(settings={"locale": "pt-PT"})
    client, saved = _sites_client(monkeypatch, site)
    resp = client.patch(f"/api/sites/{site.id}/settings", json={"settings": {"ai_disclosure": {"line": PT_LINE}}})
    assert resp.status_code == 200, resp.text
    assert saved == [{"ai_disclosure": {"line": PT_LINE}}]


def test_a_locale_and_line_saved_together_are_checked_together(monkeypatch, edition):
    site = _site()
    client, saved = _sites_client(monkeypatch, site)
    patch = {"locale": "pt-BR", "ai_disclosure": {"line": PT_LINE}}
    resp = client.patch(f"/api/sites/{site.id}/settings", json={"settings": patch})
    assert resp.status_code == 200, resp.text
    assert saved == [patch]


def test_a_locale_change_that_strands_the_saved_line_is_refused(monkeypatch, edition):
    site = _site(settings={"ai_disclosure": {"line": "An AI assistant answers here."}})
    client, saved = _sites_client(monkeypatch, site)
    resp = client.patch(f"/api/sites/{site.id}/settings", json={"settings": {"locale": "de"}})
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"] == "The AI notice must say it is an AI (for this language: 'KI')."
    assert saved == []


def test_a_save_that_tries_to_set_the_badge_is_refused(monkeypatch, edition):
    site = _site()
    client, saved = _sites_client(monkeypatch, site)
    resp = client.patch(f"/api/sites/{site.id}/settings",
                        json={"settings": {"ai_disclosure": {"badge": "Bot", "line": "An AI helps."}}})
    assert resp.status_code == 422, resp.text
    assert saved == []


def test_a_site_created_with_a_line_without_the_term_is_refused(monkeypatch, edition):
    site = _site()
    client, saved = _sites_client(monkeypatch, site)
    resp = client.post("/api/sites", json={
        "type": "custom", "display_name": SHOP, "settings": {"ai_disclosure": {"line": NO_TERM}},
    })
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"] == "The AI notice must say it is an AI (for this language: 'AI')."
    assert saved == []


def test_a_save_of_other_settings_is_untouched(monkeypatch, edition):
    site = _site()
    client, saved = _sites_client(monkeypatch, site)
    resp = client.patch(f"/api/sites/{site.id}/settings", json={"settings": {"cart_idle": {"enabled": True}}})
    assert resp.status_code == 200, resp.text
    assert saved == [{"cart_idle": {"enabled": True}}]


def test_a_failed_settings_lookup_still_sends_the_default_notice(edition, monkeypatch):
    """A failed Site/settings read never breaks widget chat: the read runs in a
    savepoint and the notice falls back to the unnamed English line."""
    from sqlalchemy.exc import ProgrammingError

    import api.widgets.config as widget_config

    def _fail(*_args, **_kwargs):
        raise ProgrammingError("SELECT sites", {}, Exception('relation "sites" does not exist'))

    monkeypatch.setattr(widget_config, "resolve_widget_source", _fail)
    db = MagicMock()

    disclosure = widget_config.resolve_ai_disclosure(db, uuid4())

    assert (disclosure.badge, disclosure.line, disclosure.locale) == ("AI", "You are chatting with an AI assistant.", "en")
    db.begin_nested.assert_called_once()
