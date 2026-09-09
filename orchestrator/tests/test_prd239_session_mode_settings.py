"""PRD-239 S6c — Settings → Session mode: where a ticket runs when its agent
names no folder (the projects folder by default, else a fresh sessions folder),
stored on the workspace, used by the claim and the Runtime Canvas; the
projects folder itself stays a .env/Docker mount the tab only explains."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
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

from services import cli_host_service as svc  # noqa: E402
from services import cli_ticket_lane as lane  # noqa: E402

WS = uuid4()


class _Query:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def first(self):
        return self._result

    def all(self):
        return self._result if isinstance(self._result, list) else []


class _DB:
    def __init__(self, *, workspace=None, ticket=None, hosts=None):
        self.workspace, self.ticket, self.hosts = workspace, ticket, hosts or []
        self.added, self.commits = [], 0

    def query(self, model):
        name = getattr(model, "__name__", "")
        if name == "Workspace":
            return _Query(self.workspace)
        if name == "CliHost":
            return _Query(self.hosts)
        return _Query(self.ticket)

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = 601

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        pass


@pytest.fixture(autouse=True)
def _no_flag_modified(monkeypatch):
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda obj, key: None)


def _ws(settings=None):
    return SimpleNamespace(id=WS, settings=settings)


def test_the_default_is_the_projects_folder_when_one_is_configured(monkeypatch):
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_DIR", "/Users/me/Development", raising=False)
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_MOUNT", "rw", raising=False)
    monkeypatch.setattr(svc, "host_allow_dirs", lambda db, ws: ["/Users/me/Development"])
    out = svc.session_mode_settings(_DB(workspace=_ws(None)), WS)
    assert out == {
        "default_folder": "projects", "default_folder_explicit": False,
        "local_projects_dir": "/Users/me/Development", "projects_mount": "rw",
        "host_allowed_roots": ["/Users/me/Development"],
    }
    assert svc.default_session_folder(_DB(workspace=_ws(None)), WS) == "/Users/me/Development"


def test_without_a_projects_folder_tickets_get_their_own_sessions_folder(monkeypatch):
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_DIR", "", raising=False)
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_MOUNT", "", raising=False)
    monkeypatch.setattr(svc, "host_allow_dirs", lambda db, ws: [])
    out = svc.session_mode_settings(_DB(workspace=_ws({"session_mode": {"default_folder": "projects"}})), WS)
    assert out["default_folder"] == "projects" and out["local_projects_dir"] is None
    # the choice is recorded but cannot be honoured → None → the host's sessions/<ticket>
    assert svc.default_session_folder(_DB(workspace=_ws({"session_mode": {"default_folder": "projects"}})), WS) is None
    assert svc.session_mode_settings(_DB(workspace=_ws(None)), WS)["default_folder"] == "sessions"


def test_the_operators_explicit_choice_wins_and_is_saved_without_mutating(monkeypatch):
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_DIR", "/Users/me/Development", raising=False)
    monkeypatch.setattr(svc, "host_allow_dirs", lambda db, ws: [])
    original = {"integrations": {"x": 1}, "session_mode": {"other": True}}
    ws = _ws(original)
    db = _DB(workspace=ws)
    out = svc.save_session_mode_settings(db, WS, default_folder="sessions")
    assert out["default_folder"] == "sessions" and out["default_folder_explicit"] is True
    assert ws.settings == {"integrations": {"x": 1}, "session_mode": {"other": True, "default_folder": "sessions"}}
    assert ws.settings is not original and original == {"integrations": {"x": 1}, "session_mode": {"other": True}}
    assert db.commits == 1
    assert svc.default_session_folder(db, WS) is None
    with pytest.raises(ValueError):
        svc.save_session_mode_settings(db, WS, default_folder="/tmp")
    with pytest.raises(LookupError):
        svc.save_session_mode_settings(_DB(workspace=None), WS, default_folder="projects")


def test_a_settings_problem_never_blocks_a_claim(monkeypatch):
    monkeypatch.setattr(svc, "session_mode_settings", lambda db, ws: (_ for _ in ()).throw(RuntimeError("db down")))
    assert svc.default_session_folder(_DB(), WS) is None


def test_the_runtime_canvas_session_uses_the_workspace_default_when_the_agent_names_no_folder(monkeypatch):
    monkeypatch.setattr(svc, "default_session_folder", lambda db, ws: "/Users/me/Development")
    monkeypatch.setattr("services.board_consent.consent_for_created_ticket", lambda db, **kw: None)
    monkeypatch.setattr("services.board_events.notify_board_event", lambda db, **kw: None)
    monkeypatch.setattr(svc.config, "LOCAL_PROJECTS_DIR", "/Users/me/Development", raising=False)
    agent = SimpleNamespace(id=7, name="Bob", configuration={"runtime": "cli"})
    host = SimpleNamespace(id=uuid4(), workspace_id=WS)
    task, created = lane.open_session_ticket(_DB(ticket=None), workspace_id=WS, agent=agent, chat_id="c1", host=host, actor="user:1")
    assert created and task.runtime_ref["cwd"] == "/Users/me/Development" and task.runtime_ref["explorer_root"] == "projects"


def test_the_settings_routes_are_declared_in_the_mount_manifest():
    manifest = json.loads((_ORCH / "reports" / "route-manifest.json").read_text())
    for method in ("GET", "PUT"):
        assert {"path": "/api/v1/cli-hosts/settings", "method": method} in manifest["routes"]
