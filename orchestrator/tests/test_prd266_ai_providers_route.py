"""PRD-266 US-017c: ``GET /api/workspaces/current`` answers ``ai_providers``.

The route adds ``ai_providers`` (``services.ai_providers.workspace_ai_providers``)
in both editions. Saving the workspace's own key through ``POST /api/keys`` adds
its provider and ``DELETE /api/keys/{id}`` removes it; switching the workspace's
Auto Live on through ``PUT /api/workspaces/current/voice-live`` (Business plan,
platform switch on, Retell armed) adds Retell and switching it off removes it.
A read that fails answers 200 with ``ai_providers: null``, logged at ERROR, and
the rest of the answer as it was.

``api/workspaces.py`` may not grow, so the integration-token masking moved to
``services.workspace_view.masked_settings``: it behaves exactly as the inline
code did (pinned here against a copy of that code).

The route tests run on real Postgres (CI), inside the test's rolled-back
transaction; system settings are a table in each test. The masking tests are
DB-free.
"""
from __future__ import annotations

import copy
import logging
import os
import uuid
from datetime import datetime
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import api.user_api_keys as user_api_keys_api  # noqa: E402
import api.workspaces as workspaces_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.llm.manager as llm_manager_mod  # noqa: E402
import core.llm.rerank_manager as rerank_manager_mod  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.llm import providers as registry  # noqa: E402
from core.models.credentials import Credential  # noqa: E402
from modules.voice import live_settings  # noqa: E402
from services import ai_providers  # noqa: E402
from services.ai_providers import CREDENTIAL_NAME_FORMS  # noqa: E402
from services.workspace_view import masked_settings  # noqa: E402

CURRENT = "/api/workspaces/current"
VOICE_LIVE = "/api/workspaces/current/voice-live"
KEYS = "/api/keys"

# Stored or sent where a key would be. Low-entropy on purpose: no secret scanner reads it as a key.
KEY_SENTINEL = "deepseek-key-sentinel-never-read"
ROW_SENTINEL = "stored-key-sentinel-never-read"
ENV_SENTINEL = "env-key-sentinel-never-read"

DEEPSEEK = {"id": "deepseek", "name": "DeepSeek"}
RETELL = {"id": "retell", "name": "Retell AI"}


# ---------------------------------------------------------------------------
# Fixtures: a saas workspace whose every provider source is off
# ---------------------------------------------------------------------------


def _refuse(*_args, **_kwargs):
    raise AssertionError("no key is decrypted here")


async def _proven(*_args, **_kwargs):
    """The live key check, passed: what a working key's save sees."""
    return user_api_keys_api.ApiKeyValidation(valid=True, message=user_api_keys_api.KEY_VALID,
                                              tested_at=datetime.utcnow())


def _quiet_sources(db, monkeypatch) -> dict:
    """Every provider source off; system settings read from the returned table."""
    for spec in registry.REGISTRY.values():
        if spec.env_key:
            monkeypatch.setattr(config, spec.env_key, "", raising=False)
    monkeypatch.setattr(config, "PLATFORM_KEY_WORKSPACE_ID", "")
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", None)
    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", False)
    # The platform lane reads the credential store unscoped: a row another test
    # committed is removed inside this test's transaction (its rollback restores it).
    every_name = [form.format(slug=slug) for slug in registry.all_slugs() for form in CREDENTIAL_NAME_FORMS]
    db.query(Credential).filter(Credential.name.in_(every_name)).delete(synchronize_session=False)
    settings: dict = {}

    def _setting(category, key, default=None):
        return settings.get((category, key)) or default

    monkeypatch.setattr(llm_manager_mod, "get_system_setting", _setting)
    monkeypatch.setattr(live_settings, "get_system_setting", _setting)
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda _category, _key: None)
    monkeypatch.setattr(rerank_manager_mod, "get_rerank_manager", lambda: SimpleNamespace(is_available=lambda: False))
    return settings


def _client(db, ctx) -> TestClient:
    app = FastAPI()
    app.include_router(workspaces_api.router)
    app.include_router(user_api_keys_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


@pytest.fixture
def floor(db_session, seed_workspace, monkeypatch):
    """A saas workspace, its owner signed in, no provider source on; key saves pass their live check."""
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    settings = _quiet_sources(db_session, monkeypatch)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda _db, _ctx: "owner")
    monkeypatch.setattr(user_api_keys_api, "_validate_provider_key", _proven)
    monkeypatch.setattr(user_api_keys_api, "get_encryption_service",
                        lambda: SimpleNamespace(encrypt=lambda _raw: ROW_SENTINEL, decrypt=_refuse))
    ws = seed_workspace()
    ctx = RequestContext(
        workspace_id=uuid.UUID(ws),
        user=UserContext(id="owner-1", clerk_user_id="clerk_owner", system_role="user"),
        auth_type="clerk",
    )
    return SimpleNamespace(db=db_session, ws=ws, settings=settings, client=_client(db_session, ctx))


def _current(floor) -> dict:
    resp = floor.client.get(CURRENT)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _plan(floor, plan: str) -> None:
    floor.db.execute(text("UPDATE workspaces SET plan = :plan WHERE id = CAST(:id AS uuid)"),
                     {"plan": plan, "id": floor.ws})
    floor.db.expire_all()


def _platform_voice(floor) -> None:
    """The platform switch on and Retell armed (key, webhook secret and agent id present)."""
    floor.settings.update({
        (live_settings.VOICE_SETTINGS_CATEGORY, live_settings.KEY_LIVE_ENABLED): "true",
        (live_settings.VOICE_SETTINGS_CATEGORY, live_settings.KEY_RETELL_API_KEY): ROW_SENTINEL,
        (live_settings.VOICE_SETTINGS_CATEGORY, live_settings.KEY_RETELL_WEBHOOK_SECRET): ROW_SENTINEL,
        (live_settings.VOICE_SETTINGS_CATEGORY, live_settings.KEY_RETELL_AGENT_ID): "retell-agent-sentinel",
    })


# ---------------------------------------------------------------------------
# The field, in both editions
# ---------------------------------------------------------------------------


def test_the_route_answers_ai_providers_an_empty_list_when_no_source_is_on(floor):
    body = _current(floor)

    assert body["ai_providers"] == []
    assert body["id"] == floor.ws


def test_the_route_answers_the_platform_providers_on_saas(floor, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", ENV_SENTINEL)

    assert _current(floor)["ai_providers"] == [{"id": "openrouter", "name": "OpenRouter"}]


def test_the_local_edition_answers_ai_providers_through_the_same_route(db_session, seed_workspace, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    _quiet_sources(db_session, monkeypatch)
    monkeypatch.setattr(config, "NVIDIA_API_KEY", ENV_SENTINEL)  # the operator's own key: local only
    ws = seed_workspace()
    ctx = RequestContext(workspace_id=uuid.UUID(ws), user=UserContext(id="local", system_role="user"),
                         auth_type="anonymous")

    resp = _client(db_session, ctx).get(CURRENT)

    assert resp.status_code == 200, resp.text
    assert resp.json()["ai_providers"] == [{"id": "nvidia", "name": "NVIDIA"}]


# ---------------------------------------------------------------------------
# US-017's AC: a key added in Settings adds its provider; deleting it removes it
# ---------------------------------------------------------------------------


def test_saving_the_workspaces_own_key_adds_its_provider_and_deleting_it_removes_it(floor):
    assert DEEPSEEK not in _current(floor)["ai_providers"]

    saved = floor.client.post(KEYS, json={"provider": "deepseek", "api_key": KEY_SENTINEL})
    assert saved.status_code == 201, saved.text
    assert saved.json()["is_active"] is True

    assert _current(floor)["ai_providers"] == [DEEPSEEK]

    assert floor.client.delete(f"{KEYS}/{saved.json()['id']}").status_code == 204

    assert _current(floor)["ai_providers"] == []


# ---------------------------------------------------------------------------
# Retell, when the workspace can use Auto Live voice
# ---------------------------------------------------------------------------


def test_switching_auto_live_on_adds_retell_and_switching_it_off_removes_it(floor):
    _plan(floor, "business")
    _platform_voice(floor)
    assert RETELL not in _current(floor)["ai_providers"]

    assert floor.client.put(VOICE_LIVE, json={"voice_live": {"enabled": True}}).status_code == 200
    assert _current(floor)["ai_providers"] == [RETELL]

    assert floor.client.put(VOICE_LIVE, json={"voice_live": {"enabled": False}}).status_code == 200
    assert _current(floor)["ai_providers"] == []


def test_a_basic_workspace_with_auto_live_on_never_lists_retell(floor):
    _plan(floor, "basic")
    _platform_voice(floor)

    assert floor.client.put(VOICE_LIVE, json={"voice_live": {"enabled": True}}).status_code == 200
    assert _current(floor)["ai_providers"] == []


# ---------------------------------------------------------------------------
# A read failure: null, logged at ERROR, the rest of the answer unchanged
# ---------------------------------------------------------------------------


def test_a_failing_provider_read_answers_200_with_null_and_logs_at_error(floor, monkeypatch, caplog):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", ENV_SENTINEL)
    readable = _current(floor)
    assert readable["ai_providers"] == [{"id": "openrouter", "name": "OpenRouter"}]

    def _unreadable(*_args, **_kwargs):
        raise OperationalError("SELECT credentials.name FROM credentials", {}, Exception("store unreadable"))

    monkeypatch.setattr(ai_providers, "_credential_store_slugs", _unreadable)
    caplog.set_level(logging.ERROR, logger="services.workspace_view")

    failed = _current(floor)

    assert failed["ai_providers"] is None
    assert {k: v for k, v in failed.items() if k != "ai_providers"} == {
        k: v for k, v in readable.items() if k != "ai_providers"
    }
    errors = [r for r in caplog.records if r.name == "services.workspace_view" and r.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].exc_info is not None
    assert ENV_SENTINEL not in errors[0].getMessage()


# ---------------------------------------------------------------------------
# The moved masking helper behaves exactly as the inline code did (DB-free)
# ---------------------------------------------------------------------------


def _inline_masking(workspace_settings):
    """The masking block exactly as it stood inline in get_current_workspace (base 7ae677b52)."""
    settings = dict(workspace_settings or {})
    integrations = dict(settings.get("integrations", {}))
    masked = {}
    for k, v in integrations.items():
        if isinstance(v, str) and len(v) > 8 and "token" in k:
            masked[k] = v[:4] + "..." + v[-4:]
        else:
            masked[k] = v
    settings["integrations"] = masked
    return settings


# A made-up bot token of 9 characters: one over the 8 that are shown as they are,
# and too short for a secret scanner to read as a credential (its rule needs 10).
FAKE_BOT_TOKEN = "tg-fake-9"
FAKE_BOT_TOKEN_SHOWN = "tg-f...ke-9"

MASKING_CASES = [
    None,
    {},
    {"theme": "dark"},
    {"integrations": {}},
    {"integrations": {"telegram_bot_token": FAKE_BOT_TOKEN}},
    {"integrations": {"slack_bot_token": "12345678"}},  # exactly 8: shown as it is
    {"integrations": {"slack_bot_token": "123456789"}},  # 9: masked
    {"integrations": {"whatsapp_access_token": "whatsapp-access-value-xyz",
                      "telegram_default_chat_id": "-100123456789"}},
    {"integrations": {"telegram_bot_token": 1234567890, "slack_trigger_mode": "allow_all"}},
    {"integrations": {"slack_bot_token": None, "telegram_bot_token": {"nested": "value-longer-than-8"}}},
    {"integrations": {"slack_default_channel": "#a-long-channel-name"}, "byok_overrides": {"deepseek": True}},
]


@pytest.mark.parametrize("workspace_settings", MASKING_CASES)
def test_masked_settings_answers_exactly_what_the_inline_code_did(workspace_settings):
    before = copy.deepcopy(workspace_settings)

    assert masked_settings(workspace_settings) == _inline_masking(workspace_settings)
    assert workspace_settings == before  # the stored settings are never changed


def test_a_token_over_8_characters_shows_its_first_4_and_last_4_around_dots():
    shown = masked_settings({"integrations": {"telegram_bot_token": FAKE_BOT_TOKEN, "slack_bot_token": "short"}})

    assert shown["integrations"] == {"telegram_bot_token": FAKE_BOT_TOKEN_SHOWN, "slack_bot_token": "short"}


def test_masked_settings_answers_a_new_dict_not_the_stored_one():
    stored = {"integrations": {"telegram_bot_token": FAKE_BOT_TOKEN}, "theme": "dark"}

    shown = masked_settings(stored)

    assert shown is not stored and shown["integrations"] is not stored["integrations"]
    assert stored["integrations"]["telegram_bot_token"] == FAKE_BOT_TOKEN
