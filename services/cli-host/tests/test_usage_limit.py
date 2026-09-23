"""F083 — a CLI's usage limit is a pause, not a failure.

Gerard runs sprints until his Claude window closes and resumes after it, so the
limit WILL hit the runtime agents. Before this, a session that exited on the
limit became "<cli> exited (code N) before finishing the turn": every in-flight
ticket errored and every later claim errored on spawn until the window reopened.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from automatos_cli_host import usage_limit
from automatos_cli_host.config import HostConfig
from automatos_cli_host.hook_server import HookServer
from automatos_cli_host.host import Host
from automatos_cli_host.session import Session

from conftest import FAKE_CLAUDE

NOW = datetime(2026, 9, 23, 13, 5, tzinfo=timezone.utc)


# ── what the CLI says ───────────────────────────────────────────────────────

def test_the_persona_runners_limit_wordings_are_recognised():
    for said in ("You've hit your limit · resets 3pm (Europe/Dublin)",
                 "Claude usage limit reached. Your limit will reset at 15:00.",
                 "API Error: 429 {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\"}}",
                 "Error: 529 overloaded",
                 '{"type":"result","subtype":"error_usage_limit","is_error":true}'):
        assert usage_limit.is_usage_limit(said), said
    for said in ("Traceback (most recent call last): KeyError: 'x'", "zsh: command not found: claude", ""):
        assert not usage_limit.is_usage_limit(said), said


def test_the_reset_time_is_read_when_the_cli_gives_one():
    assert usage_limit.resets_at("try again in 2 hours", NOW) == NOW + timedelta(hours=2)
    assert usage_limit.resets_at("Please try again in 30 minutes.", NOW) == NOW + timedelta(minutes=30)
    dublin = usage_limit.resets_at("You've hit your limit · resets 3pm (Europe/Dublin)", NOW)
    assert (dublin.hour, dublin.minute, str(dublin.tzinfo)) == (15, 0, "Europe/Dublin")
    assert dublin > NOW and dublin - NOW < timedelta(hours=24)
    earlier_today = usage_limit.resets_at("limit will reset at 09:00", NOW)
    assert earlier_today.astimezone(timezone.utc).date() == (NOW + timedelta(days=1)).date()   # already past: tomorrow
    assert usage_limit.resets_at("Usage limit reached.", NOW) is None


def test_without_a_reset_time_the_host_checks_again_in_fifteen_minutes():
    until, known = usage_limit.pause("Usage limit reached.", NOW)
    assert (until, known) == (NOW + timedelta(minutes=15), False)
    assert "checking again at" in usage_limit.describe("claude", until, known)
    until, known = usage_limit.pause("try again in 1 hour", NOW)
    line = usage_limit.describe("claude", until, known)
    assert line.startswith("paused: claude usage limit, resumes ~") and "key" not in line


# ── the session: limit output mid-turn is a pause ───────────────────────────

def _cfg(short_tmp, **over) -> HostConfig:
    base = dict(state_dir=short_tmp / "state", cli_binaries={"claude": str(FAKE_CLAUDE)}, use_worktrees=False,
                session_timeout_seconds=120)
    base.update(over)
    return HostConfig(**base)


def _ticket(workdir: Path, **over) -> dict:
    t = {"task_id": 42, "attempt": 1, "session_id": str(uuid.uuid4()), "agent_name": "Dwight",
         "title": "Say hi", "prompt": "OBJECTIVE: write hello.txt\nOUTPUT: the file\nTOOLS: Write\nBOUNDARIES: this dir",
         "cwd": str(workdir), "model": "sonnet", "allowed_tools": [], "provider": "claude"}
    t.update(over)
    return t


def _run(short_tmp, ticket):
    cfg = _cfg(short_tmp)
    hooks = HookServer(cfg.socket_path)
    hooks.start()
    allow = [str(short_tmp / "ws")]
    s = Session(ticket, cfg, allow, cfg.socket_path, default_root=allow[0])
    hooks.register(str(ticket["task_id"]), s.handle_hook)
    try:
        return s, s.run()
    finally:
        hooks.unregister(str(ticket["task_id"]))
        hooks.stop()


def test_limit_output_mid_turn_is_a_usage_limit_not_an_error(short_tmp, fake_home, env_clean, monkeypatch):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "usage-limit")
    _, out = _run(short_tmp, _ticket(workdir))
    assert out.status == "usage_limit", out.error
    assert out.error.startswith("paused: claude usage limit, resumes ~")
    assert "exited (code" not in out.error and "key" not in out.error
    assert datetime.fromisoformat(out.resets_at).hour == 15
    payload = out.as_result_payload(1)
    assert payload["status"] == "usage_limit" and payload["resets_at"] == out.resets_at


def test_a_normal_non_zero_exit_still_errors(short_tmp, fake_home, env_clean, monkeypatch):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "exit-early")
    _, out = _run(short_tmp, _ticket(workdir))
    assert out.status == "error" and "code 3" in out.error and out.resets_at is None


# ── the host: no claims for a paused CLI until its window reopens ───────────

class _Api:
    def __init__(self, tasks):
        self.tasks = tasks

    def claim(self, host_id, limit):
        return {"tasks": list(self.tasks), "parked": []}


def _host(short_tmp, tasks=()):
    host = Host(_cfg(short_tmp, url="http://127.0.0.1:9", allow_dirs=[short_tmp / "ws"]))
    host.api = _Api(tasks)
    return host


def test_a_paused_cli_is_not_served_and_says_why(short_tmp, fake_home, env_clean):
    host = _host(short_tmp)
    assert "claude" in host.capabilities()["providers"]
    until = (datetime.now().astimezone() + timedelta(hours=1)).isoformat()
    host._last_heartbeat = time.time()
    host._pause_cli("claude", until, "paused: claude usage limit, resumes ~15:00")
    caps = host.capabilities()
    assert "claude" not in caps["providers"]                    # the backend's claim filter
    assert caps["clis"]["claude"]["served"] is False
    assert caps["clis"]["claude"]["reason"] == "paused: claude usage limit, resumes ~15:00"
    assert caps["paused"] == {"claude": "paused: claude usage limit, resumes ~15:00"}
    assert host._last_heartbeat == 0.0                          # the backend hears on the next tick


def test_the_window_reopening_restores_the_cli(short_tmp, fake_home, env_clean):
    host = _host(short_tmp)
    past = (datetime.now().astimezone() - timedelta(minutes=1)).isoformat()
    host._pause_cli("claude", past, "paused: claude usage limit, resumes ~now")
    caps = host.capabilities()
    assert "claude" in caps["providers"] and "paused" not in caps and host.limited == {}


def test_a_ticket_claimed_for_a_paused_cli_goes_straight_back(short_tmp, fake_home, env_clean):
    ticket = {"task_id": 77, "attempt": 2, "provider": "claude", "session_id": str(uuid.uuid4())}
    host = _host(short_tmp, tasks=[ticket])
    until = (datetime.now().astimezone() + timedelta(hours=1)).isoformat()
    host._pause_cli("claude", until, "paused: claude usage limit, resumes ~15:00")
    host._claim_and_start("h1")
    assert host.sessions == {}                                  # nothing spawned
    released = host.pending_results["77"]
    assert released["status"] == "usage_limit" and released["attempt"] == 2
    assert released["resets_at"] == until and released["error"].startswith("paused: claude usage limit")
