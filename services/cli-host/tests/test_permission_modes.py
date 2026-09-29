"""The four permission modes on the host's gate: Manual, Edit automatically, Plan, Auto.

A tester's sessions asked about every ``mkdir`` and ``npm install``: the gate
had one stance (edits run, anything off the Bash allowlist is a card) and a
host flag nobody knew about. The claim now carries the session's mode; the gate
applies it, and its hard lines hold in every mode.
"""
from __future__ import annotations

import threading
import time

import pytest

from automatos_cli_host import permission_modes as modes
from automatos_cli_host import policy
from automatos_cli_host.adapters import claude as claude_adapter
from automatos_cli_host.adapters.base import LaunchContext, Prepared, ToolClass
from automatos_cli_host.config import parse_args
from automatos_cli_host.presets import CLAUDE, CODEX
from automatos_cli_host.session import Session

_CLAUDE = claude_adapter.ClaudeAdapter(CLAUDE)
PLAN = "1. Add the page.\n2. Run the tests."


def _decide(tool_name, tool_input, ctx):
    return policy.decide(_CLAUDE.tool_intent(tool_name, tool_input), ctx)


def _ctx(tmp_path, mode):
    return policy.PolicyContext(cwd=tmp_path, permission_mode=mode)


@pytest.mark.parametrize("mode, edit, unlisted", [
    ("manual", "ask", "ask"),
    ("edits", "allow", "ask"),
    ("plan", "deny", "ask"),
    ("auto", "allow", "allow"),
])
def test_each_mode_on_an_edit_and_an_unlisted_command(tmp_path, mode, edit, unlisted):
    ctx = _ctx(tmp_path, mode)
    assert _decide("Write", {"file_path": str(tmp_path / "index.html")}, ctx).behavior == edit
    assert _decide("Bash", {"command": "mkdir -p site/css"}, ctx).behavior == unlisted
    # the hard lines hold in every mode
    assert _decide("Bash", {"command": "git push origin main"}, ctx).behavior == "deny"
    assert _decide("Write", {"file_path": "/etc/hosts"}, ctx).behavior == "deny"
    assert _decide("Read", {"file_path": str(tmp_path / "notes.md")}, ctx).behavior == "allow"


def test_a_plan_is_a_card_only_in_plan_mode(tmp_path):
    intent = _CLAUDE.tool_intent("ExitPlanMode", {"plan": PLAN})
    assert intent.cls is ToolClass.PLAN and intent.subject.startswith("1. Add the page.")
    assert policy.decide(intent, _ctx(tmp_path, "plan")).behavior == "ask"
    assert policy.decide(intent, _ctx(tmp_path, "auto")).behavior == "allow"


@pytest.mark.parametrize("host, ticket, resuming, can_plan, expected", [
    (None, "auto", False, True, "auto"),        # the claim's mode
    ("manual", "auto", False, True, "manual"),  # the host's override wins
    (None, None, False, True, "edits"),         # an older backend: today's behaviour
    (None, "bypassPermissions", False, True, "edits"),
    (None, "plan", False, True, "plan"),
    (None, "plan", True, True, "edits"),        # a resumed session presented its plan already
    (None, "plan", False, False, "edits"),      # a CLI with no plan mode could never present one
])
def test_which_mode_a_session_runs_in(host, ticket, resuming, can_plan, expected):
    assert modes.session_mode(host, ticket, resuming=resuming, can_plan=can_plan) == expected


def _launch(tmp_path, plan_first):
    ctx = LaunchContext(cwd=tmp_path, session_dir=tmp_path / "s", ticket_path=tmp_path / "t.md",
                        system_prompt_path=tmp_path / "sp.md", task_id="1", session_id="sid", plan_first=plan_first)
    return _CLAUDE.launch_args(ctx, Prepared())


def test_plan_mode_starts_claude_code_in_its_own_plan_mode(tmp_path):
    planning, working = _launch(tmp_path, True), _launch(tmp_path, False)
    assert planning[planning.index("--permission-mode") + 1] == "plan"
    assert working[working.index("--permission-mode") + 1] == "acceptEdits"
    assert CLAUDE.plan_stance and not CODEX.plan_stance


def test_the_host_flag_and_its_old_spelling(tmp_path):
    base = []
    assert parse_args(base).permission_mode is None
    assert parse_args(base + ["--permission-mode", "plan"]).permission_mode == "plan"
    # a service installed with --unlisted-bash still starts, in the matching mode
    assert parse_args(base + ["--unlisted-bash", "allow"]).permission_mode == "auto"
    assert parse_args(base + ["--unlisted-bash", "ask"]).permission_mode == "edits"
    with pytest.raises(SystemExit):
        parse_args(base + ["--permission-mode", "bypassPermissions"])


def _session(tmp_path, mode):
    cfg = type("Cfg", (), {"ask_timeout": 2.0, "sessions_dir": tmp_path, "socket_path": tmp_path / "s.sock"})()
    s = Session({"task_id": 5, "attempt": 1, "session_id": "sid"}, cfg, [str(tmp_path)], tmp_path / "s.sock",
                default_root=str(tmp_path))
    s._policy = _ctx(tmp_path, mode)
    s._plan_dir = tmp_path / "deliverables"
    return s


def _answer(s, approved):
    def run():
        for _ in range(200):
            pending = list(s._pending_asks)
            if pending:
                s.resolve_ask(pending[0], approved)
                return
            time.sleep(0.01)
    threading.Thread(target=run, daemon=True).start()


def _hook(s, tool, tool_input):
    out = s.handle_hook({"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input})
    return out["hookSpecificOutput"]["permissionDecision"]


def test_an_approved_plan_lets_the_session_work_as_edit_automatically(tmp_path):
    s = _session(tmp_path, "plan")
    page = {"file_path": str(tmp_path / "index.html")}
    assert _hook(s, "Write", page) == "deny"
    _answer(s, True)
    assert _hook(s, "ExitPlanMode", {"plan": PLAN}) == "allow"
    assert (tmp_path / "deliverables" / modes.PLAN_FILENAME).read_text(encoding="utf-8").strip() == PLAN
    assert s._policy.permission_mode == "edits"
    assert _hook(s, "Write", page) == "allow"


def test_a_declined_plan_keeps_the_session_planning(tmp_path):
    s = _session(tmp_path, "plan")
    _answer(s, False)
    assert _hook(s, "ExitPlanMode", {"plan": PLAN}) == "deny"
    assert s._policy.permission_mode == "plan"
    assert _hook(s, "Write", {"file_path": str(tmp_path / "index.html")}) == "deny"


def test_a_write_that_names_no_path_follows_the_mode_too(tmp_path):
    intent = policy.ToolIntent(tool="NotebookEdit", cls=ToolClass.FILE_WRITE)
    assert policy.decide(intent, _ctx(tmp_path, "manual")).behavior == "ask"
    assert policy.decide(intent, _ctx(tmp_path, "plan")).behavior == "deny"
    assert policy.decide(intent, _ctx(tmp_path, "edits")).behavior == "allow"


def test_the_plan_is_read_from_claude_codes_plan_file_when_only_its_path_arrives(tmp_path):
    plans = tmp_path / ".claude" / "plans"
    plans.mkdir(parents=True)
    (plans / "tidy-otter.md").write_text(PLAN, encoding="utf-8")
    (tmp_path / "secret.md").write_text("not a plan", encoding="utf-8")
    root = tmp_path / ".claude"
    assert modes.plan_text({"planFilePath": str(plans / "tidy-otter.md")}, root) == PLAN
    assert modes.plan_text({"plan": "inline", "planFilePath": str(plans / "tidy-otter.md")}, root) == "inline"
    # only a markdown file under Claude Code's own folder is read
    assert modes.plan_text({"planFilePath": str(tmp_path / "secret.md")}, root) == ""
    assert modes.plan_text({"planFilePath": str(plans / ".." / ".." / "secret.md")}, root) == ""
    assert modes.plan_text({"planFilePath": str(plans / "missing.md")}, root) == ""


# ── Codex: the same modes on Codex's own tool names ─────────────────────────

def _codex():
    from automatos_cli_host.adapters.codex import CodexAdapter
    return CodexAdapter(CODEX)


def _codex_patch(path):
    return {"input": f"*** Begin Patch\n*** Add File: {path}\n+x\n*** End Patch\n"}


@pytest.mark.parametrize("mode, edit, unlisted", [
    ("manual", "ask", "ask"),
    ("edits", "allow", "ask"),
    ("auto", "allow", "allow"),
])
def test_codex_sessions_take_the_same_modes(tmp_path, mode, edit, unlisted):
    codex, ctx = _codex(), _ctx(tmp_path, mode)
    verdict = lambda tool, tool_input: policy.decide(codex.tool_intent(tool, tool_input), ctx).behavior  # noqa: E731
    assert verdict("apply_patch", _codex_patch(tmp_path / "index.html")) == edit
    assert verdict("exec_command", {"cmd": "mkdir -p site/css"}) == unlisted
    assert verdict("shell", {"command": ["mkdir", "-p", "site"]}) == unlisted
    # the hard lines hold for Codex too
    assert verdict("exec_command", {"cmd": "git push origin main"}) == "deny"
    assert verdict("apply_patch", _codex_patch("/etc/hosts")) == "deny"


def test_codex_has_no_plan_mode_so_plan_runs_as_edit_automatically(tmp_path):
    """Plan needs a CLI that can present a plan; Codex cannot, so it never waits for one."""
    assert modes.session_mode(None, "plan", resuming=False, can_plan=bool(CODEX.plan_stance)) == "edits"
    ctx = LaunchContext(cwd=tmp_path, session_dir=tmp_path / "s", ticket_path=tmp_path / "t.md",
                        system_prompt_path=tmp_path / "sp.md", task_id="1", session_id="sid", plan_first=True)
    args = _codex().launch_args(ctx, Prepared())
    assert all(token in args for token in CODEX.ungated_stance)
