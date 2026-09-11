"""CLI adapter design — the seam itself: presets are honest data, the adapter
contract holds for every row, the policy is CLI-neutral, translation round-trips,
and a CLI this host cannot run is an honest result rather than another CLI."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from automatos_cli_host import policy, presets, session
from automatos_cli_host.adapters import NotServed, adapter_for, adapters, has_adapter
from automatos_cli_host.adapters.base import LaunchContext, Prepared, PresetAdapter, Reply, ToolClass
from automatos_cli_host.adapters.claude import ClaudeAdapter
from automatos_cli_host.presets import CLAUDE, CODEX, REGISTRY, UnknownCli, preset_for


# ── the table ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("preset", list(REGISTRY.values()), ids=list(REGISTRY))
def test_every_preset_is_internally_consistent(preset, tmp_path):
    """Table-driven: what a row promises never contradicts what its adapter would
    put on a command line, and the invariant fields are filled in."""
    assert preset.tier in presets.TIERS and preset.turn_end in presets.TURN_ENDS
    assert preset.hook_events <= presets.BUS_EVENTS
    assert preset.forbidden_args, f"{preset.id}: a CLI with no forbidden arguments has no stated invariant"
    assert preset.auth_probe is not None and preset.auth_probe.code and preset.auth_probe.refusal
    assert preset.strip_env, f"{preset.id}: name the credentials this CLI must never inherit"
    if preset.tier != presets.TIER_SEED:
        assert "PreToolUse" in preset.hook_events and "Stop" in preset.hook_events
        assert preset.hook_timeout("PreToolUse") is not None
    adapter = PresetAdapter(preset, str(tmp_path / preset.binary))
    ctx = LaunchContext(cwd=tmp_path, session_dir=tmp_path / "s", ticket_path=tmp_path / "t.md",
                        system_prompt_path=tmp_path / "sp.md", task_id="1", session_id="sid", model="m", worktree_name="wt")
    args = adapter.launch_args(ctx, Prepared())
    session.assert_args_honour_invariant(args, preset.forbidden_args)      # never intersects
    for required in preset.required_args:
        assert required in args, f"{preset.id}: {required} must always be on the command line"
    assert preset.ungated_stance and all(tok in args for tok in preset.ungated_stance)


def test_the_registry_resolves_ids_and_defaults_to_claude():
    assert preset_for(None) is CLAUDE and preset_for("") is CLAUDE and preset_for(" Codex ") is CODEX
    with pytest.raises(UnknownCli):
        preset_for("grok")
    with pytest.raises(ValueError):   # a seed-tier CLI cannot end a turn on a hook it does not have
        presets.CliPreset(id="x", label="X", binary="x", tier=presets.TIER_SEED, turn_end=presets.TURN_END_STOP_HOOK)


def test_a_known_but_unserved_cli_is_honest_not_a_fallback():
    """Codex is in the registry (the picker, the claim filter) but has no adapter
    yet: it is never run as Claude Code."""
    assert has_adapter("claude") and not has_adapter("codex")
    with pytest.raises(NotServed):
        adapter_for("codex")
    with pytest.raises(UnknownCli):
        adapter_for("grok")
    assert list(adapters()) == ["claude"]


def test_a_ticket_for_an_unserved_cli_errors_before_any_process(tmp_path):
    cfg = SimpleNamespace(cli_binaries={}, sessions_dir=tmp_path, socket_path=tmp_path / "s.sock",
                          session_timeout_seconds=60, startup_timeout_seconds=5, use_worktrees=False, ask_timeout=1.0)
    (tmp_path / "ws").mkdir()
    s = session.Session({"task_id": 5, "attempt": 1, "session_id": "sid", "provider": "codex", "cwd": str(tmp_path / "ws")},
                        cfg, [str(tmp_path)], tmp_path / "s.sock", default_root=str(tmp_path))
    out = s.run()
    assert out.status == "error" and out.exit_reason == "cli_not_served" and "Codex" in (out.error or "")
    assert s.proc is None
    assert s.handle_hook({"hook_event_name": "PreToolUse"}) == {}   # nothing to answer for


# ── the gate is CLI-neutral ──────────────────────────────────────────────────

def test_tool_intent_says_what_a_call_does(tmp_path):
    a = ClaudeAdapter(CLAUDE)
    edit = a.tool_intent("Edit", {"file_path": "/w/a.py", "old_string": "x"})
    assert edit.cls is ToolClass.FILE_WRITE and edit.paths == ("/w/a.py",) and edit.subject == "/w/a.py"
    read = a.tool_intent("Grep", {"pattern": "x"})
    assert read.cls is ToolClass.FILE_READ and read.paths == () and read.subject is None
    bash = a.tool_intent("Bash", {"command": "git status", "timeout": 5})
    assert bash.cls is ToolClass.SHELL and bash.command == "git status" and bash.subject == "git status"
    assert a.tool_intent("WebSearch", {"query": "q"}).cls is ToolClass.WEB
    assert a.tool_intent("TodoWrite", {}).cls is ToolClass.BENIGN
    assert a.tool_intent("mcp__x__y", {}).cls is ToolClass.UNKNOWN
    # the base adapter knows no tools: everything is unknown, so the policy denies it
    assert PresetAdapter(CODEX).tool_intent("unified_exec", {"cmd": "ls"}).cls is ToolClass.UNKNOWN


def test_policy_verdicts_are_the_same_through_intents(tmp_path):
    """The proof the seam changed nothing: the pre-seam verdicts, now reached
    through ToolIntent, for every class the policy knows."""
    ctx = policy.PolicyContext(cwd=tmp_path, allowed_bash=policy.bash_allowlist_from_config(["make build"]))
    a = ClaudeAdapter(CLAUDE)
    d = lambda name, ti: policy.decide(a.tool_intent(name, ti), ctx)
    assert d("Edit", {"file_path": str(tmp_path / "a.py")}).allow
    assert not d("Write", {"file_path": "/etc/passwd"}).allow
    assert d("Bash", {"command": "git status"}).allow
    assert d("Bash", {"command": "git push origin main"}).behavior == "deny"
    assert d("Bash", {"command": "rm -rf build"}).behavior == "ask"
    assert d("WebFetch", {"url": "https://x"}).allow
    assert d("Task", {}).behavior == "deny"
    # a multi-path write is judged on EVERY path
    two = a.tool_intent("Edit", {"file_path": str(tmp_path / "ok.py")})
    outside = two.__class__(tool="Edit", cls=ToolClass.FILE_WRITE, paths=(str(tmp_path / "ok.py"), "/etc/x"))
    assert policy.decide(outside, ctx).behavior == "deny"


# ── translation ──────────────────────────────────────────────────────────────

def test_claude_translation_is_identity_and_renders_the_wire_shape():
    a = ClaudeAdapter(CLAUDE)
    raw = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "ls"}, "extra": 1}
    assert a.normalize_event(raw) == raw
    assert a.normalize_event({"hook_event_name": "SomethingNew"}) is None      # not on the bus → dropped
    assert a.normalize_event({}) == {}                                          # no name → passed through, ignored later
    assert a.render_response("PreToolUse", Reply.allow()) == {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}
    deny = a.render_response("PreToolUse", Reply.deny("no"))
    assert deny["hookSpecificOutput"]["permissionDecision"] == "deny" and deny["hookSpecificOutput"]["permissionDecisionReason"] == "no"
    perm = a.render_response("PermissionRequest", Reply.deny("no"))
    assert perm["hookSpecificOutput"]["decision"] == {"behavior": "deny", "message": "no"}
    assert a.render_response("UserPromptSubmit", Reply.with_context("ctx"))["hookSpecificOutput"]["additionalContext"] == "ctx"
    assert a.render_response("Stop", Reply.none()) is None and a.render_response("PostToolUse", Reply.none()) is None
    # what the shim writes on its own when the host is unreachable: a deny for the held events only
    assert a.offline_deny("PreToolUse", "gone")["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert a.offline_deny("Stop", "gone") is None


# ── preflight ────────────────────────────────────────────────────────────────

def test_preflight_names_the_missing_binary_or_the_wrong_login(tmp_path, monkeypatch):
    missing = ClaudeAdapter(CLAUDE, str(tmp_path / "nope"))
    refusal = missing.preflight()
    assert refusal.code == "claude_missing" and "not installed" in refusal.message
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\necho 1.0\n")
    fake.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude.json").write_text("{}")
    monkeypatch.setenv("HOME", str(home))
    present = ClaudeAdapter(CLAUDE, str(fake))
    refusal = present.preflight()
    assert refusal.code == "claude_not_onboarded" and "run `claude` once" in refusal.message.lower()
    info = present.detect()
    assert info["served"] is False and info["path"] == str(fake) and info["onboarded"] is False and info["tier"] == "native"
    (home / ".claude.json").write_text('{"hasCompletedOnboarding": true}')
    assert present.preflight() is None and present.detect()["served"] is True
    # the base adapter, with no probe of its own, never serves — nothing to check is not safe to spend on
    assert PresetAdapter(presets.CliPreset(id="z", label="Z", binary="z", tier=presets.TIER_SEED,
                                           turn_end=presets.TURN_END_PROCESS_EXIT, forbidden_args=("-x",),
                                           strip_env=frozenset({"K"})), str(fake)).preflight().code == "z_no_login_check"


def test_launch_args_follow_the_preset_not_the_cli_name(tmp_path):
    """Codex's row spells its launch differently in six places; the base builds it
    from the row alone (the adapter that RUNS it is a later wave)."""
    a = PresetAdapter(CODEX, "/usr/local/bin/codex")
    ctx = LaunchContext(cwd=tmp_path, session_dir=tmp_path / "s", ticket_path=tmp_path / "t.md",
                        system_prompt_path=tmp_path / "sp.md", task_id="9", session_id="sid", model="gpt-5.5", worktree_name="wt")
    args = a.launch_args(ctx, Prepared(env={"CODEX_HOME": "/h"}, args=[]))
    joined = " ".join(args)
    assert "--session-id" not in args                                      # Codex mints its own id
    assert "-C " + str(tmp_path) in joined and "-a never -s workspace-write" in joined
    assert "--dangerously-bypass-hook-trust" in args and "--enable worktrees --worktree" in joined
    assert "--append-system-prompt-file" not in args                       # the soul rides the bus
    assert "wt" not in args                                                # Codex names its own worktree
    assert "--model gpt-5.5" in joined and args[-1].startswith("Work the Automatos ticket")
    resumed = a.launch_args(LaunchContext(cwd=tmp_path, session_dir=tmp_path, ticket_path=tmp_path / "t",
                                          system_prompt_path=tmp_path / "a", task_id="9", session_id="sid",
                                          resume_session_id="old", worktree_name="wt"), Prepared())
    assert resumed[1:3] == ["resume", "old"] and "--worktree" not in resumed   # a subcommand; no worktree on resume
    session.assert_args_honour_invariant(resumed, CODEX.forbidden_args)
    with pytest.raises(RuntimeError):
        session.assert_args_honour_invariant(["codex", "exec", "x"], CODEX.forbidden_args)
