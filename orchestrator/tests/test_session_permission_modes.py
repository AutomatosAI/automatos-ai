"""Session permission modes: Manual, Edit automatically, Plan and Auto.

Testers of the open-source edition answered a card for every ``mkdir``,
``npm install`` and ``python script.py`` their sessions ran, and the only way
out was a host flag nobody knew about. Sessions now take the four modes Claude
Code users already know: the workspace sets a default on Settings → Session
mode (Auto in the local edition), an agent may override it, and the claim
carries the result to the host as ``permission_mode``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")
_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from api.cli_hosts import SessionModeSettingsRequest  # noqa: E402
from core import session_permission_modes as modes  # noqa: E402
from core.cli_runtime import validate_runtime_configuration  # noqa: E402
from services import cli_host_service as svc  # noqa: E402

WS = uuid4()


class _Query:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._result


class _DB:
    def __init__(self, workspace=None, agent=None):
        self.workspace, self.agent, self.commits = workspace, agent, 0

    def query(self, model):
        return _Query(self.workspace if getattr(model, "__name__", "") == "Workspace" else self.agent)

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        pass


class _BrokenDB(_DB):
    def query(self, model):
        raise RuntimeError("db down")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda obj, key: None)
    monkeypatch.setattr(svc, "host_allow_dirs", lambda db, ws: [])


def _ws(settings=None):
    return NS(id=WS, settings=settings)


@pytest.mark.parametrize("edition, expected", [("local", "auto"), ("saas", "edits")])
def test_the_default_is_auto_in_the_open_source_edition_only(monkeypatch, edition, expected):
    monkeypatch.setattr(svc.config, "AUTH_EDITION", edition)
    assert svc.session_mode_settings(_DB(_ws(None)), WS)["permission_mode"] == expected
    assert svc.session_permission_mode(_DB(_ws(None)), WS) == expected


@pytest.mark.parametrize("mode", modes.PERMISSION_MODES)
def test_the_workspace_saves_any_of_the_four_and_keeps_its_other_choices(monkeypatch, mode):
    monkeypatch.setattr(svc.config, "AUTH_EDITION", "local")
    original = {"session_mode": {"default_folder": "projects"}}
    ws = _ws(original)
    db = _DB(ws)
    out = svc.save_session_mode_settings(db, WS, permission_mode=mode)
    assert out["permission_mode"] == mode and db.commits == 1
    assert ws.settings == {"session_mode": {"default_folder": "projects", "permission_mode": mode}}
    assert original == {"session_mode": {"default_folder": "projects"}}  # rebuilt, never mutated
    assert svc.session_permission_mode(db, WS) == mode


def test_only_the_four_modes_are_saved():
    with pytest.raises(ValueError):
        svc.save_session_mode_settings(_DB(_ws(None)), WS, permission_mode="bypassPermissions")
    with pytest.raises(ValueError):
        svc.save_session_mode_settings(_DB(_ws(None)), WS)
    with pytest.raises(ValueError):
        SessionModeSettingsRequest(permission_mode="bypassPermissions")
    assert SessionModeSettingsRequest(permission_mode="plan").default_folder is None


def test_an_agents_own_mode_wins_and_an_unknown_one_falls_back():
    assert modes.ticket_permission_mode({"permission_mode": "plan"}, "auto") == "plan"
    assert modes.ticket_permission_mode({"permission_mode": "yolo"}, "auto") == "auto"
    assert modes.ticket_permission_mode(None, "manual") == "manual"
    assert modes.workspace_permission_mode({"permission_mode": "yolo"}, "saas") == "edits"


def test_an_agent_configuration_accepts_only_the_four_modes():
    agent = {"runtime": "cli", "provider": "claude"}
    assert validate_runtime_configuration({**agent, "permission_mode": "manual"}, cli_enabled=True) == []
    assert validate_runtime_configuration(agent, cli_enabled=True) == []
    errors = validate_runtime_configuration({**agent, "permission_mode": "bypassPermissions"}, cli_enabled=True)
    assert errors and "permission_mode" in errors[0]


def test_unreadable_settings_never_block_a_claim_and_fall_back_to_edit_automatically():
    assert svc.session_permission_mode(_BrokenDB(), WS) == "edits"


def _task():
    return NS(id=7, workspace_id=WS, title="Build the site", status="assigned", assigned_agent_id=3,
              runtime_ref={}, blocked_reason=None, attempts=1, review_mode="auto", attachment_ids=[])


@pytest.mark.parametrize("stored, agent_mode, expected", [
    (None, None, "auto"),                                        # the local edition's default
    ({"session_mode": {"permission_mode": "manual"}}, None, "manual"),  # the workspace's choice
    ({"session_mode": {"permission_mode": "manual"}}, "plan", "plan"),  # the agent's override
])
def test_the_claim_carries_the_sessions_mode_to_the_host(monkeypatch, stored, agent_mode, expected):
    monkeypatch.setattr(svc.config, "AUTH_EDITION", "local")
    task = _task()
    agent = NS(id=3, name="Builder", configuration={"runtime": "cli", "provider": "claude",
                                                    **({"permission_mode": agent_mode} if agent_mode else {})})
    monkeypatch.setattr(svc, "claim_tasks", lambda db, **kw: [task])
    monkeypatch.setattr(svc, "_blocked_pending_approval", lambda db, t: False)
    monkeypatch.setattr(svc, "served_providers_of", lambda h: None)
    monkeypatch.setattr(svc, "default_session_folder", lambda db, ws: None)
    monkeypatch.setattr(svc, "explorer_root_for", lambda *a, **k: None)
    monkeypatch.setattr(svc, "_session_system_prompt", lambda agent: "")
    monkeypatch.setattr(svc, "_ticket_prompt", lambda t, memory="": "Build the site.")
    monkeypatch.setattr(svc, "_field_memory_block", lambda db, t: "")

    claimed = svc.claim_for_host(_DB(_ws(stored), agent), NS(id="h1", workspace_id=WS), limit=1)["tasks"][0]

    assert claimed["permission_mode"] == expected
    assert claimed["session_id"] == task.runtime_ref["session_id"] and claimed["resume_session_id"] is None
    assert claimed["session_token"] and claimed["prompt"] == "Build the site." and claimed["agent_name"] == "Builder"
