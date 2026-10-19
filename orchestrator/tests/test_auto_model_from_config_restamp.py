"""An admin's change of the platform default moves the Auto rows that never chose one.

Gerard, 19 Oct 2026: the live default for Auto is system_settings
``orchestrator_llm``. Deploying moves nothing; when an admin changes the setting
through the System Settings API, ``restamp_inheriting_auto_agents`` moves the Auto
rows still on a shipped seed's pair or marked as inheriting, and leaves every row a
workspace chose in Settings > Orchestrator. PURE: rows, settings and the session are
faked (tests/test_auto_model_from_config_restamp_realdb.py holds the SQL filter).
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from api import system_settings as settings_api
from core.llm import auto_model_sync as sync
from core.llm import new_agent_model
from core.models.system_settings import SystemSettingUpdate

HAIKU = ("anthropic", "claude-haiku-5-5")
SONNET = ("anthropic", "claude-sonnet-5-5")
GEMINI_SEED = {"provider": "openrouter", "model_id": "google/gemini-2.5-flash", "temperature": 0.7, "max_tokens": 4000}
GPT4O_SEED = {"provider": "openrouter", "model_id": "openai/gpt-4o", "temperature": 0.7, "max_tokens": 4000}
CHOSEN = {"provider": "openrouter", "model_id": "openai/gpt-5.5", "temperature": 0.2, "max_tokens": 4000}


class _Db:
    def __init__(self):
        self.flushed = self.committed = self.rolled_back = 0

    def flush(self):
        self.flushed += 1

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1


def _auto(ws, model_config):
    return NS(slug=f"auto-{ws}", workspace_id=ws, model_config=model_config)


@pytest.fixture
def platform(monkeypatch):
    """``platform(rows, target)``: these Auto rows; ``target`` as the platform default."""
    def install(rows, target):
        def selected(db, *, lock=False):
            install.locked.append(lock)
            return rows

        monkeypatch.setattr(sync, "auto_agent_rows", selected)
        monkeypatch.setattr(new_agent_model, "default_route", lambda: target)
        return rows
    install.locked = []
    return install


# ── which rows inherit ──────────────────────────────────────────────────────

def test_the_shipped_seed_pairs_are_inheriting_and_a_choice_is_not():
    assert sync.is_inheriting(GEMINI_SEED) and sync.is_inheriting(GPT4O_SEED)
    assert not sync.is_inheriting(CHOSEN)
    assert not sync.is_inheriting({}) and not sync.is_inheriting(None)


def test_a_seed_pair_under_any_other_marker_is_not_read_as_a_seed():
    """Only a row with no marker at all can be a seed from before the marker."""
    assert not sync.is_inheriting({**GEMINI_SEED, "source": "workspace"})
    assert not sync.is_inheriting({**GPT4O_SEED, "source": None})


def test_a_marked_row_re_pointed_in_settings_is_a_choice():
    """Settings > Orchestrator copies the stored dict and writes only provider and
    model, so the marker survives a choice; the recorded route tells them apart."""
    marked = sync.inheriting_model_config({**GEMINI_SEED, "provider": HAIKU[0], "model_id": HAIKU[1]})
    assert sync.is_inheriting(marked)
    re_pointed = {**marked, "provider": "openrouter", "model_id": "openai/gpt-5.5"}
    assert not sync.is_inheriting(re_pointed)


def test_inheriting_model_config_builds_a_new_dict_and_keeps_other_keys():
    original = dict(GEMINI_SEED)
    stamped = sync.inheriting_model_config(original, HAIKU)
    assert original == GEMINI_SEED                                      # not mutated
    assert stamped["temperature"] == 0.7 and stamped["max_tokens"] == 4000
    assert (stamped["provider"], stamped["model_id"]) == HAIKU


# ── the restamp, across workspaces ──────────────────────────────────────────

def test_the_restamp_moves_seed_and_marked_rows_and_leaves_chosen_ones(platform):
    marked_sonnet = sync.inheriting_model_config({**GEMINI_SEED, "provider": SONNET[0], "model_id": SONNET[1]})
    rows = platform([
        _auto("ws-a", dict(GEMINI_SEED)), _auto("ws-b", dict(GEMINI_SEED)), _auto("ws-c", dict(GPT4O_SEED)),
        _auto("ws-d", marked_sonnet), _auto("ws-e", dict(CHOSEN)), _auto("ws-f", sync.inheriting_model_config(
            {**GEMINI_SEED, "provider": HAIKU[0], "model_id": HAIKU[1]})),
    ], HAIKU)
    db = _Db()
    counts = sync.restamp_inheriting_auto_agents(db)
    assert counts["target"] == "anthropic/claude-haiku-5-5"
    assert (counts["moved"], counts["current"], counts["chosen"]) == (4, 1, 1)
    assert counts["by_route"] == {
        "openrouter/google/gemini-2.5-flash -> anthropic/claude-haiku-5-5": 2,
        "openrouter/openai/gpt-4o -> anthropic/claude-haiku-5-5": 1,
        "anthropic/claude-sonnet-5-5 -> anthropic/claude-haiku-5-5": 1,
    }
    for moved in rows[:4]:
        assert (moved.model_config["provider"], moved.model_config["model_id"]) == HAIKU
        assert sync.is_inheriting(moved.model_config) and moved.model_config["max_tokens"] == 4000
    assert rows[4].model_config == CHOSEN
    assert db.flushed == 1 and db.committed == 0                       # the caller commits
    assert platform.locked == [True]                                   # FOR UPDATE until that commit


def test_a_second_restamp_to_the_same_default_moves_nothing(platform):
    rows = platform([_auto("ws-a", dict(GEMINI_SEED))], HAIKU)
    sync.restamp_inheriting_auto_agents(_Db())
    first = dict(rows[0].model_config)
    counts = sync.restamp_inheriting_auto_agents(_Db())
    assert (counts["moved"], counts["current"]) == (0, 1) and rows[0].model_config == first


def test_the_restamped_row_follows_the_next_change_too(platform):
    rows = platform([_auto("ws-a", dict(GEMINI_SEED))], HAIKU)
    sync.restamp_inheriting_auto_agents(_Db())
    platform(rows, SONNET)
    sync.restamp_inheriting_auto_agents(_Db())
    assert (rows[0].model_config["provider"], rows[0].model_config["model_id"]) == SONNET


# ── the admin endpoints trigger it only on a provider/model change ──────────

class _SettingsDb(_Db):
    def __init__(self, setting):
        super().__init__()
        self.setting = setting

    def query(self, *entities):
        return self

    def filter(self, *criteria):
        return self

    def first(self):
        return self.setting

    def refresh(self, obj):
        pass


# A platform admin signed in through Clerk: the caller that also passes the route's
# require_workspace_permission("workspace:manage"), which refuses an api_key caller.
ADMIN = NS(auth_type="clerk", user=NS(system_role="admin"))
MEMBER = NS(auth_type="clerk", user=NS(system_role="user"))


def _setting(category, key, value):
    return NS(id=7, category=category, key=key, value=value, description=None, is_sensitive=False)


@pytest.fixture
def restamps(monkeypatch):
    calls = []
    monkeypatch.setattr(sync, "restamp_inheriting_auto_agents",
                        lambda db: calls.append(db) or {"moved": 3, "by_route": {}})
    return calls


def _put(setting, value, ctx=ADMIN, db=None):
    db = db or _SettingsDb(setting)
    settings_api.update_system_setting(setting_id=setting.id, update_data=SystemSettingUpdate(value=value),
                                       db=db, ctx=ctx)
    return db


def _bulk(setting, value):
    db = _SettingsDb(setting)
    return settings_api.bulk_update_settings(
        request=None, updates=[settings_api.BulkUpdateItem(id=setting.id, value=value)], db=db, ctx=ADMIN)


def test_the_two_routes_are_plain_defs_run_in_the_threadpool():
    """AGENTS.md: a route over a synchronous session is never ``async def``."""
    import inspect

    assert not inspect.iscoroutinefunction(settings_api.update_system_setting)
    assert not inspect.iscoroutinefunction(settings_api.bulk_update_settings)


def test_a_member_who_is_not_an_admin_is_refused_before_any_move(restamps):
    with pytest.raises(settings_api.HTTPException) as raised:
        _put(_setting("orchestrator_llm", "model", "old"), "new", ctx=MEMBER)
    assert raised.value.status_code == 403 and restamps == []


@pytest.mark.parametrize("key", ["provider", "model", "llm_provider", "llm_model"])
def test_a_put_that_changes_the_orchestrator_route_restamps(restamps, key):
    db = _put(_setting("orchestrator_llm", key, "old"), "new")
    assert restamps == [db] and db.committed == 2                      # the setting, then the moves


@pytest.mark.parametrize("category, key, old, new", [
    ("orchestrator_llm", "model", "claude-haiku-5-5", "claude-haiku-5-5"),   # unchanged
    ("orchestrator_llm", "temperature", "0.7", "0.2"),                       # not the route
    ("system_llm", "model", "google/gemini-2.5-flash", "claude-haiku-5-5"),  # another tier
])
def test_a_put_that_leaves_the_orchestrator_route_does_not(restamps, category, key, old, new):
    _put(_setting(category, key, old), new)
    assert restamps == []


def test_a_bulk_update_that_changes_the_model_restamps_and_reports_it(restamps):
    reply = _bulk(_setting("orchestrator_llm", "model", "google/gemini-2.5-flash"), "claude-haiku-5-5")
    assert len(restamps) == 1 and reply["auto_models"] == {"moved": 3, "by_route": {}}
    assert reply["message"] == "Updated 1 settings successfully"


def test_a_bulk_update_elsewhere_leaves_the_reply_as_it_was(restamps):
    reply = _bulk(_setting("system_llm", "model", "google/gemini-2.5-flash"), "claude-haiku-5-5")
    assert restamps == [] and reply == {"message": "Updated 1 settings successfully"}


def _boom(db):
    raise RuntimeError("pool gone")


def test_a_failed_restamp_puts_the_setting_back_so_the_same_save_retries(monkeypatch):
    """The setting is committed before the move (default_route reads it on its own
    connection); a failed move undoes it, so the setting and the rows never part
    and saving the same value again changes it again, and so moves the rows."""
    monkeypatch.setattr(sync, "restamp_inheriting_auto_agents", _boom)
    setting = _setting("orchestrator_llm", "model", "old")
    db = _SettingsDb(setting)
    with pytest.raises(settings_api.HTTPException) as raised:
        _put(setting, "new", db=db)
    assert raised.value.status_code == 500 and raised.value.detail == settings_api.RESTAMP_UNDONE
    assert setting.value == "old" and db.rolled_back == 1 and db.committed == 2   # saved, then undone


class _UndoFailsDb(_SettingsDb):
    def commit(self):
        super().commit()
        if self.committed > 1:
            raise RuntimeError("pool gone")


def test_when_the_undo_fails_too_the_admin_is_told_to_run_the_script(monkeypatch):
    monkeypatch.setattr(sync, "restamp_inheriting_auto_agents", _boom)
    setting = _setting("orchestrator_llm", "model", "old")
    with pytest.raises(settings_api.HTTPException) as raised:
        _put(setting, "new", db=_UndoFailsDb(setting))
    assert raised.value.detail == settings_api.RESTAMP_FAILED


def test_a_failed_restamp_after_a_bulk_update_puts_every_saved_setting_back(monkeypatch):
    monkeypatch.setattr(sync, "restamp_inheriting_auto_agents", _boom)
    setting = _setting("orchestrator_llm", "model", "google/gemini-2.5-flash")
    with pytest.raises(settings_api.HTTPException) as raised:
        _bulk(setting, "claude-haiku-5-5")
    assert raised.value.detail == settings_api.RESTAMP_UNDONE and setting.value == "google/gemini-2.5-flash"
