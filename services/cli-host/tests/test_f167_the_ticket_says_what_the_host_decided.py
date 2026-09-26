"""F167 — the ticket says what the host decided for each tool call, and why.

Ticket #999 said a Chrome print command "needed your approval, and it went
through". Nobody was asked. The host runs ``--unlisted-bash allow``, so the
command ran on the host's standing rule. The host told the backend only which
tool ran, never what it decided: an allow carried no reason, and a hold's
answer was never reported. Each PreToolUse event now carries the decision
(allow, ask or deny), its reason, and for a hold the operator's answer.
"""
from __future__ import annotations

import threading
import time

from automatos_cli_host import policy
from automatos_cli_host.adapters import claude as claude_adapter
from automatos_cli_host.adapters.base import ToolClass, ToolIntent
from automatos_cli_host.presets import CLAUDE
from automatos_cli_host.session import Session

_CLAUDE = claude_adapter.ClaudeAdapter(CLAUDE)
CHROME_PRINT = "google-chrome --headless --print-to-pdf=report.pdf report.html"


def _decide(tool_name, tool_input, ctx):
    return policy.decide(_CLAUDE.tool_intent(tool_name, tool_input), ctx)


def test_an_unlisted_command_the_host_ran_says_nobody_was_asked(tmp_path):
    permissive = policy.PolicyContext(cwd=tmp_path, unlisted_bash="allow")
    ran = _decide("Bash", {"command": CHROME_PRINT}, permissive)
    assert ran.behavior == "allow"
    assert ran.reason == policy.ALLOWED_UNLISTED_BASH.format(command=policy._first_words(CHROME_PRINT))
    assert "--unlisted-bash allow" in ran.reason
    # the unlisted verb's reason wins over an allowlisted verb on the same line
    assert _decide("Bash", {"command": f"ls && {CHROME_PRINT}"}, permissive).reason == ran.reason


def test_every_allow_says_why(tmp_path):
    ctx = policy.PolicyContext(cwd=tmp_path, session_tools=("board_summary",))
    assert _decide("Bash", {"command": "ls -la"}, ctx).reason == policy.ALLOWED_BASH
    assert _decide("Read", {"file_path": str(tmp_path / "notes.md")}, ctx).reason == policy.ALLOWED_FILES
    assert _decide("Grep", {"pattern": "TODO"}, ctx).reason == policy.ALLOWED_FILES
    assert _decide("WebFetch", {"url": "https://example.com"}, ctx).reason == policy.ALLOWED_NO_APPROVAL
    offered = policy.decide(ToolIntent(tool="mcp__automatos__board_summary", cls=ToolClass.PLATFORM,
                                       command="board_summary"), ctx)
    assert (offered.behavior, offered.reason) == ("allow", policy.ALLOWED_SESSION_TOOL)


def test_a_hold_or_a_refusal_keeps_its_own_reason(tmp_path):
    strict = policy.PolicyContext(cwd=tmp_path)
    permissive = policy.PolicyContext(cwd=tmp_path, unlisted_bash="allow")
    held = _decide("Bash", {"command": CHROME_PRINT}, strict)
    assert held.behavior == "ask" and "outside this ticket's Bash allowlist" in held.reason
    pushed = _decide("Bash", {"command": "git push origin main"}, permissive)
    assert pushed.behavior == "deny" and "never allowed in a session" in pushed.reason
    outside = _decide("Bash", {"command": "xxd /etc/passwd"}, permissive)
    assert outside.behavior != "allow" and "--unlisted-bash" not in outside.reason
    assert _decide("Read", {"file_path": "/etc/passwd"}, strict).behavior == "deny"


# ── the event the backend receives ──────────────────────────────────────────

def _session(tmp_path, ask_timeout=1.0, **policy_kw):
    cfg = type("Cfg", (), {"ask_timeout": ask_timeout, "sessions_dir": tmp_path, "socket_path": tmp_path / "s.sock"})()
    s = Session({"task_id": 999, "attempt": 1, "session_id": "sid"}, cfg, [str(tmp_path)], tmp_path / "s.sock",
                default_root=str(tmp_path))
    s._policy = policy.PolicyContext(cwd=tmp_path, **policy_kw)
    return s


def _call(s, command):
    return s.handle_hook({"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": command}})


def _events(s):
    out = []
    while not s.events.empty():
        out.append(s.events.get_nowait())
    return out


def _answer_holds(s, approve):
    def _run():
        answered = set()
        for _ in range(200):
            time.sleep(0.01)
            for ev in list(s.events.queue):
                if ev.get("event") == "PermissionRequest" and ev["request_id"] not in answered:
                    answered.add(ev["request_id"])
                    s.resolve_ask(ev["request_id"], approve)
                    return
    threading.Thread(target=_run, daemon=True).start()


def test_a_call_that_ran_with_nobody_asked_is_reported_as_such(tmp_path):
    s = _session(tmp_path, unlisted_bash="allow")
    assert _call(s, CHROME_PRINT)["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert _call(s, "git push origin main")["hookSpecificOutput"]["permissionDecision"] == "deny"
    ran, refused = [e for e in _events(s) if e["event"] == "PreToolUse"]
    assert (ran["decision"], ran["tool_name"], ran["subject"]) == ("allow", "Bash", CHROME_PRINT)
    assert "--unlisted-bash allow" in ran["reason"] and "answer" not in ran
    assert refused["decision"] == "deny" and "never allowed in a session" in refused["reason"]
    assert ran["event_id"] and refused["event_id"] and ran["event_id"] != refused["event_id"]  # one id per call


def test_a_held_call_is_reported_with_the_operators_answer(tmp_path):
    s = _session(tmp_path)
    _answer_holds(s, True)
    assert _call(s, CHROME_PRINT)["hookSpecificOutput"]["permissionDecision"] == "allow"
    held = _events(s)
    assert [e["event"] for e in held] == ["PermissionRequest", "PreToolUse"]  # reported once, after the answer
    assert (held[1]["decision"], held[1]["answer"]) == ("ask", "approved")
    assert held[1]["request_id"] == held[0]["request_id"]  # the backend matches the answer to its question
    assert "outside this ticket's Bash allowlist" in held[1]["reason"]

    _answer_holds(s, False)
    assert _call(s, CHROME_PRINT)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert _events(s)[-1]["answer"] == "denied"

    quiet = _session(tmp_path, ask_timeout=0.05)
    assert _call(quiet, CHROME_PRINT)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert _events(quiet)[-1]["answer"] == "no answer"


def test_a_reason_is_bounded(tmp_path):
    s = _session(tmp_path, unlisted_bash="allow")
    _call(s, "google-chrome " + "--flag " * 200)
    event = _events(s)[-1]
    assert len(event["reason"]) <= 300


def test_the_session_rules_never_say_every_other_command_is_held():
    """Under ``--unlisted-bash allow`` an unlisted command runs with nobody asked;
    "anything else is held" told #999's session a hold had happened."""
    from automatos_cli_host.session import SESSION_RULES

    rules = " ".join(SESSION_RULES.split())
    assert "anything else may be held for the operator" in rules and "anything else is held" not in rules
