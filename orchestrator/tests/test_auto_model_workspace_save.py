"""A workspace's save sets only its own Auto; the platform default is an admin's (19 Oct 2026).

* Settings > Orchestrator writes that workspace's Auto row and nothing else: no
  system_settings write (one tenant's save had become every workspace's default).
  Saving the platform default's own pair, or leaving a following row's model as it
  was, keeps the row following the platform default; any other pair marks it
  ``workspace_choice``, which no restamp moves.
* Every admin path that changes the platform default moves the Auto rows on it:
  POST /api/system-settings/reset-to-defaults and the platform_update_system_setting
  tool, beside Part 1's PUT and bulk-update; a failed move puts the setting back.
* GET /api/models/platform-default gives any signed-in user ``default_route()``, so
  the frontend keeps no model id.
PURE: rows, settings and sessions are faked.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from core.llm import auto_model_sync as sync
from core.llm import new_agent_model

HAIKU = ("anthropic", "claude-haiku-5-5")
SONNET = ("anthropic", "claude-sonnet-5-5")
GEMINI_SEED = {"provider": "openrouter", "model_id": "google/gemini-2.5-flash", "temperature": 0.7, "max_tokens": 4000}
ADMIN = NS(auth_type="clerk", user=NS(system_role="admin"))


@pytest.fixture
def platform_default(monkeypatch):
    def install(route):
        monkeypatch.setattr(new_agent_model, "default_route", lambda: route)
    install(HAIKU)
    return install


@pytest.fixture
def restamps(monkeypatch):
    calls = []
    monkeypatch.setattr(sync, "restamp_inheriting_auto_agents",
                        lambda db: calls.append(db) or {"moved": 2, "by_route": {}})
    return calls


def _boom(db):
    raise RuntimeError("pool gone")


# ── the workspace save marks the row ────────────────────────────────────────

HAIKU_UPDATE = {"provider": HAIKU[0], "model_id": HAIKU[1]}
SONNET_UPDATE = {"provider": SONNET[0], "model_id": SONNET[1]}
CHOSEN = {**GEMINI_SEED, "model_id": "openai/gpt-5.5"}


def test_saving_the_platform_default_pair_keeps_the_row_inheriting():
    for stored in (dict(GEMINI_SEED), dict(CHOSEN), {**CHOSEN, "source": sync.WORKSPACE_CHOICE_SOURCE}):
        saved = sync.workspace_saved_model_config(stored, HAIKU_UPDATE, HAIKU)
        assert sync.is_inheriting(saved) and saved["platform_route"] == HAIKU_UPDATE
        assert saved["max_tokens"] == 4000


def test_saving_another_pair_is_the_workspaces_choice_and_a_new_dict():
    stored = sync.inheriting_model_config({**GEMINI_SEED, **HAIKU_UPDATE})
    saved = sync.workspace_saved_model_config(stored, SONNET_UPDATE, HAIKU)
    assert saved["source"] == sync.WORKSPACE_CHOICE_SOURCE and "platform_route" not in saved
    assert (saved["provider"], saved["model_id"]) == SONNET and not sync.is_inheriting(saved)
    assert stored["source"] == sync.PLATFORM_DEFAULT_SOURCE and stored["model_id"] == HAIKU[1]   # not mutated


def test_a_save_that_leaves_a_following_rows_model_keeps_it_following():
    """A Soul or sampling save carries the LLM block as loaded: the row's model is no
    choice, even when the platform default has moved and this row is still to follow."""
    marked_sonnet = sync.inheriting_model_config({**GEMINI_SEED, **SONNET_UPDATE})
    for stored in (dict(GEMINI_SEED), marked_sonnet):
        saved = sync.workspace_saved_model_config(stored, {"temperature": 0.3}, HAIKU)
        assert sync.is_inheriting(saved) and saved["temperature"] == 0.3
        assert saved["model_id"] == stored["model_id"]


def test_a_save_that_leaves_a_chosen_rows_model_marks_the_choice():
    saved = sync.workspace_saved_model_config(dict(CHOSEN), {"temperature": 0.3}, HAIKU)
    assert saved["source"] == sync.WORKSPACE_CHOICE_SOURCE and saved["model_id"] == "openai/gpt-5.5"


@pytest.mark.parametrize("route", [("openrouter", "google/gemini-2.5-flash"), ("openrouter", "openai/gpt-4o"), HAIKU])
def test_a_workspace_choice_is_never_read_as_inheriting_even_on_a_seed_pair(route):
    chosen = {**GEMINI_SEED, "provider": route[0], "model_id": route[1], "source": sync.WORKSPACE_CHOICE_SOURCE}
    assert not sync.is_inheriting(chosen)
    assert sync.plan_restamp([NS(model_config=chosen)], SONNET).chosen == 1


@pytest.fixture
def flagged(monkeypatch):
    calls = []
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda obj, key: calls.append(key))
    return calls


def _save(model_config, llm):
    import services.orchestrator_auto_agent as auto

    agent = NS(model_config=model_config)
    db = NS()  # any attribute access would raise: the save never touches the session
    auto.sync_auto_llm(db, agent, llm)
    return agent.model_config


def test_a_save_of_the_platform_default_follows_the_next_admin_change(platform_default, flagged):
    saved = _save(dict(CHOSEN), {**HAIKU_UPDATE, "temperature": 0.3})
    assert sync.is_inheriting(saved) and saved["temperature"] == 0.3
    plan = sync.plan_restamp([NS(model_config=saved)], SONNET)
    assert len(plan.moves) == 1 and flagged == ["model_config"]


def test_a_save_of_another_model_stays_through_an_admin_change(platform_default, flagged):
    saved = _save(dict(GEMINI_SEED), {"provider": "openrouter", "model_id": "openai/gpt-5.5"})
    assert saved["source"] == sync.WORKSPACE_CHOICE_SOURCE
    assert sync.plan_restamp([NS(model_config=saved)], SONNET).chosen == 1


@pytest.mark.parametrize("stored", [dict(CHOSEN), sync.inheriting_model_config(dict(GEMINI_SEED))])
def test_a_save_with_a_blank_model_keeps_the_stored_one(platform_default, flagged, stored):
    """The Orchestrator tab's fallback config (its GET failed) carries a blank provider
    and model: a save of it writes the sampling keys and never blanks the row's model."""
    saved = _save(dict(stored), {"provider": "", "model_id": "", "temperature": 0.3})
    assert (saved["provider"], saved["model_id"]) == (stored["provider"], stored["model_id"])
    assert saved["temperature"] == 0.3
    assert sync.is_inheriting(saved) == sync.is_inheriting(stored)


class _OneAgentDb:
    """A session holding one agent; a write would fail the test."""

    def __init__(self, agent):
        self.agent = agent

    def query(self, *entities):
        return self

    def filter(self, *criteria):
        return self

    def first(self):
        return self.agent

    def commit(self):
        raise AssertionError("a model_config naming no model must not be written")

    def rollback(self):
        raise AssertionError("nothing was written to roll back")


@pytest.mark.parametrize("body", [{"provider": "", "model_id": "", "temperature": 0.7}, {"temperature": 0.7}])
def test_the_agent_model_config_save_refuses_a_body_naming_no_model(body):
    """PUT /api/agents/{id}/model-config replaces the whole model_config: one with no
    model_id would blank the agent's model past every catalog and policy check."""
    from fastapi import HTTPException

    from api.agent_endpoints import update_agent_model_config

    agent = NS(id=7, name="Auto", is_system_agent=True, model_config=dict(CHOSEN))
    with pytest.raises(HTTPException) as raised:
        asyncio.run(update_agent_model_config(7, body, ctx=NS(workspace_id="ws"), db=_OneAgentDb(agent)))
    assert raised.value.status_code == 422 and agent.model_config == CHOSEN


def test_the_save_no_longer_writes_system_settings():
    import services.orchestrator_auto_agent as auto

    assert not hasattr(auto, "_sync_orchestrator_llm_settings")
    assert "SystemSetting" not in Path(auto.__file__).read_text(encoding="utf-8")


# ── reset-to-defaults restamps ──────────────────────────────────────────────

class _ResetDb:
    def __init__(self, settings):
        self.settings = settings
        self.committed = self.rolled_back = 0

    def query(self, *entities):
        return self

    def filter(self, *criteria):
        return self

    def all(self):
        return self.settings

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1


def _setting(category, key, value, default_value):
    return NS(category=category, key=key, value=value, default_value=default_value, is_sensitive=False)


def _reset(settings, category=None):
    from api import system_settings as settings_api

    db = _ResetDb(settings)
    return settings_api.reset_settings_to_defaults(category=category, db=db, ctx=ADMIN), db


def test_reset_to_defaults_is_a_plain_def():
    from api import system_settings as settings_api

    assert not inspect.iscoroutinefunction(settings_api.reset_settings_to_defaults)


def test_a_reset_that_moves_the_platform_default_restamps(restamps):
    model = _setting("orchestrator_llm", "model", "openai/gpt-5.5", "claude-haiku-5-5")
    reply, db = _reset([model, _setting("orchestrator_llm", "temperature", "0.2", "0.7")], "orchestrator_llm")
    assert model.value == "claude-haiku-5-5" and restamps == [db]
    assert reply["auto_models"] == {"moved": 2, "by_route": {}} and db.committed == 2


def test_a_reset_that_leaves_the_route_does_not_restamp(restamps):
    reply, _ = _reset([_setting("orchestrator_llm", "model", "claude-haiku-5-5", "claude-haiku-5-5"),
                       _setting("system_llm", "model", "openai/gpt-5.5", "claude-haiku-5-5")])
    assert restamps == [] and "auto_models" not in reply


def test_a_failed_restamp_after_a_reset_puts_every_setting_back(monkeypatch):
    from api import system_settings as settings_api

    monkeypatch.setattr(sync, "restamp_inheriting_auto_agents", _boom)
    model = _setting("orchestrator_llm", "model", "openai/gpt-5.5", "claude-haiku-5-5")
    temperature = _setting("orchestrator_llm", "temperature", "0.2", "0.7")
    with pytest.raises(settings_api.HTTPException) as raised:
        _reset([model, temperature])
    assert raised.value.status_code == 500 and raised.value.detail == sync.RESTAMP_UNDONE
    assert (model.value, temperature.value) == ("openai/gpt-5.5", "0.2")


def test_the_failure_message_hands_the_admin_no_command():
    assert "scripts/" not in sync.RESTAMP_FAILED and "--apply" not in sync.RESTAMP_FAILED


# ── the platform_update_system_setting tool restamps ────────────────────────

class _ToolDb(_ResetDb):
    def first(self):
        return self.settings[0]


def _tool(setting, value):
    from modules.tools.discovery.handlers_workspace import update_system_setting

    db = _ToolDb([setting])
    params = {"category": setting.category, "key": setting.key, "value": value}
    return asyncio.run(update_system_setting(db, None, params)), db


def _tool_setting(category, key, value):
    return NS(id=3, category=category, key=key, value=value, is_sensitive=False, is_required=True, description=None)


def test_the_tool_restamps_on_a_platform_default_change_and_reports_it(restamps):
    result, db = _tool(_tool_setting("orchestrator_llm", "provider", "openrouter"), "anthropic")
    assert result["success"] is True and result["auto_models"] == {"moved": 2, "by_route": {}}
    assert restamps == [db] and db.committed == 2


@pytest.mark.parametrize("category, key, old, new", [
    ("orchestrator_llm", "temperature", "0.7", "0.2"),
    ("orchestrator_llm", "model", "claude-haiku-5-5", "claude-haiku-5-5"),
    ("system_llm", "model", "google/gemini-2.5-flash", "claude-haiku-5-5"),
])
def test_the_tool_leaves_the_auto_rows_for_any_other_write(restamps, category, key, old, new):
    result, _ = _tool(_tool_setting(category, key, old), new)
    assert result["success"] is True and "auto_models" not in result and restamps == []


def test_a_failed_restamp_through_the_tool_puts_the_setting_back(monkeypatch):
    monkeypatch.setattr(sync, "restamp_inheriting_auto_agents", _boom)
    setting = _tool_setting("orchestrator_llm", "model", "openai/gpt-5.5")
    result, _ = _tool(setting, "claude-haiku-5-5")
    assert result == {"success": False, "error": sync.RESTAMP_UNDONE}
    assert setting.value == "openai/gpt-5.5"


# ── GET /api/models/platform-default ────────────────────────────────────────

def _route():
    from api.models_endpoints import router

    paths = [route.path for route in router.routes]
    return next(route for route in router.routes if route.path == "/api/models/platform-default"), paths


def test_the_endpoint_returns_default_route(platform_default):
    from api.models_endpoints import get_platform_default_route

    platform_default(SONNET)
    assert get_platform_default_route(ctx=ADMIN) == {"provider": SONNET[0], "model_id": SONNET[1]}
    assert not inspect.iscoroutinefunction(get_platform_default_route)


def test_the_endpoint_requires_a_signed_in_caller_and_wins_over_model_id():
    from core.auth.hybrid import get_request_context_hybrid

    route, paths = _route()
    assert route.methods == {"GET"}
    assert get_request_context_hybrid in [dep.call for dep in route.dependant.dependencies]
    assert paths.index("/api/models/platform-default") < paths.index("/api/models/{model_id}")


def test_the_endpoint_is_in_the_committed_route_manifest():
    manifest = Path(__file__).resolve().parents[1] / "reports" / "route-manifest.json"
    routes = json.loads(manifest.read_text(encoding="utf-8"))["routes"]
    assert {"method": "GET", "path": "/api/models/platform-default"} in routes
