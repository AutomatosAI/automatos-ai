"""PRD-253 S1.7 — one ticket through a supervised ``copilot -p`` session, against the
fake ``copilot`` (``tests/fake_copilot.py``, the 1.0.91 spellings).

The turn the PRD describes: the agent's own home built (the operator's untouched),
Copilot spawned on the pty with no allow flag, its Claude-format hooks gating every
call (a write allowed, ``git push`` denied, its own re-ask for a fetch answered with
the gate's verdict), the process exit as the turn's end, usage and AI credits read
from the session record, and the same session id resumed. Every permission mode
runs on Copilot as on any CLI; a run whose hooks never loaded is refused.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from automatos_cli_host.config import HostConfig
from automatos_cli_host.hook_server import HookServer
from automatos_cli_host.permission_modes import PLAN_EDIT_REFUSED_TURN, PLAN_EVENT
from automatos_cli_host.session import Session

from conftest import FAKE_COPILOT

ALLOW_FLAGS = ("--allow-all-tools", "--allow-all", "--yolo", "--allow-tool", "--allow-all-paths", "--allow-all-urls")


def _cfg(short_tmp, **over) -> HostConfig:
    base = dict(state_dir=short_tmp / "state", cli_binaries={"copilot": str(FAKE_COPILOT)}, use_worktrees=False,
                session_timeout_seconds=120, ask_timeout=1.0)
    base.update(over)
    return HostConfig(**base)


def _ticket(workdir: Path, **over) -> dict:
    t = {"task_id": 91, "attempt": 1, "session_id": str(uuid.uuid4()), "agent_id": 58, "agent_name": "Coder",
         "title": "Say hi", "prompt": "OBJECTIVE: write hello.txt", "cwd": str(workdir), "model": "claude-sonnet-4.6",
         "allowed_tools": [], "provider": "copilot"}
    t.update(over)
    return t


def _run(short_tmp, ticket, cfg=None):
    cfg = cfg or _cfg(short_tmp)
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


def _workdir(short_tmp) -> Path:
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    return workdir


def _tree(root: Path):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_copilot_turn_end_to_end(short_tmp, fake_copilot_home, env_clean):
    workdir = _workdir(short_tmp)
    before = _tree(fake_copilot_home / ".copilot")
    ticket = _ticket(workdir)
    s, out = _run(short_tmp, ticket)
    assert out.status == "success" and out.exit_reason == "completed", out
    assert (workdir / "hello.txt").read_text() == "hi\n"
    assert "Push was denied by policy" in out.result_text and "Pushed!" not in out.result_text
    assert "Fetched the spec" in out.result_text            # its own re-ask got the gate's verdict
    assert "Contract seen" in out.result_text                # the ticket rode UserPromptSubmit
    assert any("git push" in json.dumps(d) for d in out.permission_denials)
    assert out.session_id == ticket["session_id"]
    assert out.usage["total_tokens"] == 140 and out.usage["ai_credits"] == 2.0 and out.usage["premium_requests"] == 1
    assert out.usage["model"] == "claude-sonnet-4.6" and "usd" not in json.dumps(out.usage)
    argv = list(s.proc.args)
    assert argv[argv.index("-p") + 1].startswith("Work the Automatos ticket")
    assert not set(ALLOW_FLAGS) & set(argv)
    for flag in ("--no-ask-user", "--disable-builtin-mcps", "--no-remote", "--no-auto-update"):
        assert flag in argv
    assert "--no-auto-login" not in argv                     # F233: it switches the session's sign-in off
    assert _tree(fake_copilot_home / ".copilot") == before   # the operator's home is untouched
    home = short_tmp / "state" / "agents" / "58" / ".copilot"
    assert json.loads((home / "config.json").read_text())["trustedFolders"] == []
    assert json.loads((home / "settings.json").read_text())["sandbox"]["enabled"] is True


def test_a_resumed_session_continues_the_same_session_id(short_tmp, fake_copilot_home, env_clean):
    workdir = _workdir(short_tmp)
    _, first = _run(short_tmp, _ticket(workdir))
    assert first.status == "success"
    second_s, second = _run(short_tmp, _ticket(workdir, session_id=str(uuid.uuid4()), resume_session_id=first.session_id))
    assert second.status == "success", second
    argv = list(second_s.proc.args)
    assert argv[argv.index("--resume") + 1] == first.session_id and "--session-id" not in argv
    assert second.usage["total_tokens"] == 140                # this turn only, not both
    assert second.usage["ai_credits"] == 2.0 and second.usage["premium_requests"] == 1
    _, unknown = _run(short_tmp, _ticket(workdir, resume_session_id="never-was"))
    assert unknown.status == "error"


def test_every_permission_mode_runs_on_copilot(short_tmp, fake_copilot_home, env_clean):
    workdir = _workdir(short_tmp)
    _, manual = _run(short_tmp, _ticket(workdir, permission_mode="manual"))      # the edit is a card nobody answers
    assert not (workdir / "hello.txt").exists() and "Push was denied" in manual.result_text
    _, auto = _run(short_tmp, _ticket(workdir, permission_mode="auto"))
    assert (workdir / "hello.txt").exists() and "Pushed!" not in auto.result_text  # never-allowed holds in Auto too


def test_plan_on_copilot_is_the_plan_turn(short_tmp, fake_copilot_home, env_clean):
    """Wave P: Copilot's own plan mode is not used — the gate holds it read-only, and
    the turn's final message is the plan, which goes to the backend as the Plan card."""
    workdir = _workdir(short_tmp)
    s, out = _run(short_tmp, _ticket(workdir, permission_mode="plan"))
    assert out.status == "success", out
    assert not (workdir / "hello.txt").exists()
    assert any(d["reason"] == PLAN_EDIT_REFUSED_TURN for d in out.permission_denials)
    assert s.plan == {"text": out.result_text, "approved_in_turn": False}
    events = []
    while not s.events.empty():
        events.append(s.events.get_nowait())
    assert [e["text"] for e in events if e.get("event") == PLAN_EVENT] == [out.result_text]
    approved_s, approved = _run(short_tmp, _ticket(workdir, permission_mode="edits", plan_approved=True,
                                                   resume_session_id=out.session_id))
    assert approved.status == "success" and (workdir / "hello.txt").exists() and approved_s.plan is None


def test_a_run_whose_hooks_never_loaded_reports_nothing(short_tmp, fake_copilot_home, env_clean, monkeypatch):
    """PRD-253 S0.2: no SessionStart, no gate — the run is refused, whatever it said."""
    monkeypatch.setenv("FAKE_COPILOT_SCENARIO", "no-hooks")
    workdir = _workdir(short_tmp)
    _, out = _run(short_tmp, _ticket(workdir))
    assert out.status == "error" and out.exit_reason == "ungated_exit" and out.result_text == ""


def test_running_out_of_ai_credits_is_a_pause(short_tmp, fake_copilot_home, env_clean, monkeypatch):
    monkeypatch.setenv("FAKE_COPILOT_SCENARIO", "ai-credits")
    _, out = _run(short_tmp, _ticket(_workdir(short_tmp)))
    assert out.status == "usage_limit" and out.error.startswith("paused: copilot usage limit")
