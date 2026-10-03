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


@pytest.mark.parametrize("host, ticket, approved, expected", [
    (None, "auto", False, "auto"),          # the claim's mode
    ("manual", "auto", False, "manual"),    # the host's override wins
    (None, None, False, "edits"),           # an older backend: today's behaviour
    (None, "bypassPermissions", False, "edits"),
    (None, "plan", False, "plan"),          # PRD-253 Wave P: on every CLI, resumed or not — the claim decides
    (None, "edits", True, "edits"),         # the backend sends Edit automatically once the plan is approved
    ("plan", "edits", True, "edits"),       # a host whose override is Plan carries on once the plan is approved
    ("plan", "edits", False, "plan"),
])
def test_which_mode_a_session_runs_in(host, ticket, approved, expected):
    assert modes.session_mode(host, ticket, plan_approved=approved) == expected


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


@pytest.mark.parametrize("mode", ["manual", "edits", "plan", "auto"])
def test_a_write_that_names_no_file_is_refused_in_every_mode(tmp_path, mode):
    """PRD-253 S0.1: the gate cannot place a write with no path, so no mode lets it
    run. (#845 had it follow the mode, which allowed it in Edit automatically and
    Auto — harmless for Claude Code, whose edits always carry ``file_path``, but
    not for a patch whose paths an adapter could not read.)"""
    intent = policy.ToolIntent(tool="NotebookEdit", cls=ToolClass.FILE_WRITE)
    decision = policy.decide(intent, _ctx(tmp_path, mode))
    assert decision.behavior == "deny"
    assert decision.reason == policy.WRITE_NAMES_NO_FILE


@pytest.mark.parametrize("mode", ["manual", "edits", "auto"])
def test_a_search_without_a_path_still_works_in_the_folder(tmp_path, mode):
    for tool in ("Grep", "Glob"):
        intent = _CLAUDE.tool_intent(tool, {"pattern": "TODO"})
        assert not intent.paths
        assert policy.decide(intent, _ctx(tmp_path, mode)).behavior == "allow"


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
    # PRD-253 S0.1: a patch whose files the adapter cannot read is a write to nowhere
    headless = {"input": "*** Begin Patch\n+x\n*** End Patch\n"}
    assert codex.tool_intent("apply_patch", headless).paths == ()
    assert verdict("apply_patch", headless) == "deny"


def test_codex_plans_too_its_plan_is_the_turns_final_message(tmp_path):
    """PRD-253 Wave P: Plan needs no plan mode of the CLI's own. Codex launches as it
    always does; the gate holds it read-only and its refusals say how a plan is
    presented without ExitPlanMode."""
    assert modes.session_mode(None, "plan") == "plan"
    ctx = LaunchContext(cwd=tmp_path, session_dir=tmp_path / "s", ticket_path=tmp_path / "t.md",
                        system_prompt_path=tmp_path / "sp.md", task_id="1", session_id="sid", plan_first=True)
    args = _codex().launch_args(ctx, Prepared())
    assert all(token in args for token in CODEX.ungated_stance)
    plan = policy.PolicyContext(cwd=tmp_path, permission_mode="plan", allowed_bash=policy.PLAN_BASH_ALLOW)
    refused = policy.decide(_codex().tool_intent("apply_patch", _codex_patch(tmp_path / "index.html")), plan)
    assert refused.behavior == "deny" and refused.reason == modes.PLAN_EDIT_REFUSED_TURN
    claude_plan = policy.PolicyContext(cwd=tmp_path, permission_mode="plan", plan_tool=True)
    assert _decide("Write", {"file_path": str(tmp_path / "x.md")}, claude_plan).reason == modes.PLAN_EDIT_REFUSED


@pytest.mark.parametrize("command, expected", [
    ("git log --oneline", "allow"), ("git diff", "allow"), ("cat a.txt", "allow"), ("rg TODO", "allow"),
    ("sort < data.txt", "allow"), ("ls > /dev/null", "allow"), ("grep -rn x . 2>&1", "allow"),
    ("git commit -m x", "ask"), ("git add a", "ask"), ("pytest", "ask"), ("npm run build", "ask"),
    ("echo x > notes.md", "ask"), ("cat a >> b.txt", "ask"), ("sed -i 's/a/b/' f", "ask"),
    ("sed --in-place 's/a/b/' f", "ask"), ("awk -i inplace '{print}' f", "ask"), ("sed -n '1p' f", "allow"),
])
def test_plan_is_read_only_in_the_shell_too(tmp_path, command, expected):
    """Plan is read-only on every CLI; a CLI with no plan mode has only the gate holding it there."""
    plan = policy.PolicyContext(cwd=tmp_path, permission_mode="plan", allowed_bash=policy.PLAN_BASH_ALLOW)
    assert policy.decide_bash(command, plan).behavior == expected, command
    assert set(policy.PLAN_BASH_ALLOW) < set(policy.DEFAULT_BASH_ALLOW)


def test_outside_plan_a_redirection_is_judged_as_before(tmp_path):
    edits = policy.PolicyContext(cwd=tmp_path, permission_mode="edits")
    assert policy.decide_bash("echo x > notes.md", edits).behavior == "allow"
    assert policy.decide_bash("sed -i 's/a/b/' f", edits).behavior == "allow"


def test_an_unanswered_plan_card_hands_the_plan_to_the_operator(tmp_path):
    """Claude Code's plan card nobody answered (PRD-253 Wave P): not a refusal that
    sends the ticket to review — the plan goes to the operator as the Plan card,
    the turn ends, and Approve resumes this same session. A second presentation
    is not a second card."""
    s = _session(tmp_path, "plan")
    s.cfg.ask_timeout = 0.2
    assert _hook(s, "ExitPlanMode", {"plan": PLAN}) == "deny"
    assert s.plan == {"text": PLAN, "approved_in_turn": False}
    assert s.denials == []                                        # handed over, not refused
    assert not s._pending_asks
    assert _hook(s, "ExitPlanMode", {"plan": PLAN}) == "deny"      # already with the operator: no new card
    assert s._policy.permission_mode == "plan"


def test_a_plan_approved_in_the_turn_is_reported_as_approved(tmp_path):
    s = _session(tmp_path, "plan")
    _answer(s, True)
    assert _hook(s, "ExitPlanMode", {"plan": PLAN}) == "allow"
    assert s.plan == {"text": PLAN, "approved_in_turn": True}


def test_a_plan_turn_is_named_in_the_ticket_file():
    from automatos_cli_host.session_prompt import PLAN_TURN_SECTION, build_ticket_file
    ticket = {"task_id": 4, "title": "t", "prompt": "do it"}
    assert PLAN_TURN_SECTION in build_ticket_file(ticket, None, True)
    assert "Plan mode" not in build_ticket_file(ticket, None)
