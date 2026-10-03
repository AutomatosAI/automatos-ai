"""PRD-253 W1–W2 — GitHub Copilot CLI as a session CLI, adapter by adapter.

The home is per agent and rebuilt from a whitelist (never a token), the login is
read from the account pointer or ``gh`` (never a credential), the launch carries
no allow flag, every Copilot tool says what it does to the same gate, the record
books tokens and AI credits (never a price), and Copilot's sandbox sits under
the session when the host sandboxes.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from automatos_cli_host import policy, session
from automatos_cli_host.adapters import adapter_for
from automatos_cli_host.adapters import copilot_home as home_mod
from automatos_cli_host.adapters.base import LaunchContext, Reply, ToolClass
from automatos_cli_host.adapters.copilot import CopilotAdapter, mcp_config, parse_version
from automatos_cli_host.adapters.copilot_record import read_events_usage, last_message, mcp_server_blocked
from automatos_cli_host.adapters.copilot_sandbox import missing as tools_missing  # bound before conftest stubs it
from automatos_cli_host.adapters.copilot_sandbox import unavailable_reason
from automatos_cli_host.presets import COPILOT
from automatos_cli_host.sandbox import SessionSandbox

TOKEN_MARKER = "marker-not-a-token"
ACCOUNT = {"host": "https://github.com", "login": "octocat"}


@pytest.fixture
def operator(tmp_path, monkeypatch):
    """The operator's own ~/.copilot: logged in through the keychain, with keys a
    session must never carry."""
    monkeypatch.delenv("COPILOT_HOME", raising=False)
    for name in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    (home / ".copilot").mkdir(parents=True)
    config = {"loggedInUsers": [ACCOUNT], "lastLoggedInUser": ACCOUNT, "includeCoAuthoredBy": False,
              "trustedFolders": ["/Users/me/somewhere"], "model": "gpt-5.4", "experimental": True}
    (home / ".copilot" / "config.json").write_text(json.dumps(config))
    return home


def _binary(tmp_path, version="1.0.91"):
    path = tmp_path / "copilot"
    path.write_text(f"#!/bin/sh\necho 'GitHub Copilot CLI {version}'\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _adapter(tmp_path, home, *, version="1.0.91", run=None, sandbox=None):
    return CopilotAdapter(COPILOT, _binary(tmp_path, version), home=home, sandbox=sandbox,
                          run=run or (lambda *a, **k: NS(returncode=1, stdout="", stderr="")))


def _ctx(tmp_path, **over):
    state = tmp_path / "state"
    base = dict(cwd=tmp_path / "repo", session_dir=state / "sessions" / "7", ticket_path=tmp_path / "t.md",
                system_prompt_path=tmp_path / "sp.md", task_id="7", session_id="sid-1", agent_id="58",
                state_dir=state, model=None, worktree_name=None)
    base.update(over)
    (base["cwd"]).mkdir(parents=True, exist_ok=True)
    return LaunchContext(**base)


def _tree(root: Path):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# ── the home (S1.2) ──────────────────────────────────────────────────────────

def test_the_agents_home_holds_only_the_whitelist_and_the_fixed_values(tmp_path, operator):
    before = _tree(operator / ".copilot")
    a = _adapter(tmp_path, operator)
    prepared = a.prepare(_ctx(tmp_path))
    home = Path(prepared.env["COPILOT_HOME"])
    assert home == tmp_path / "state" / "agents" / "58" / ".copilot"
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    config = json.loads((home / "config.json").read_text())
    assert config == {"loggedInUsers": [ACCOUNT], "lastLoggedInUser": ACCOUNT, "trustedFolders": []}
    settings = json.loads((home / "settings.json").read_text())
    assert settings == {"includeCoAuthoredBy": False, **home_mod.FIXED_SETTINGS}   # the operator's model is not copied
    assert settings["memory"] is False and settings["continueOnAutoMode"] is False and settings["askUser"] is False
    for name in ("config.json", "settings.json"):
        assert stat.S_IMODE((home / name).stat().st_mode) == 0o600
    assert _tree(operator / ".copilot") == before                          # the operator's home is untouched


def test_a_token_in_the_operators_config_never_reaches_the_session(tmp_path, operator):
    config = json.loads((operator / ".copilot" / "config.json").read_text())
    (operator / ".copilot" / "config.json").write_text(json.dumps({**config, "copilotTokens": {"x": TOKEN_MARKER}}))
    a = _adapter(tmp_path, operator, run=lambda *a, **k: NS(returncode=0, stdout="", stderr="Logged in to github.com account octocat"))
    prepared = a.prepare(_ctx(tmp_path))
    for blob in _tree(Path(prepared.env["COPILOT_HOME"])).values():
        assert TOKEN_MARKER.encode() not in blob
    assert a.login()[1] == "gh"                                  # a plaintext login is not the OS credential store


def test_two_tickets_of_one_agent_share_the_home_and_hooks_are_replaced(tmp_path, operator):
    a = _adapter(tmp_path, operator)
    first = Path(a.prepare(_ctx(tmp_path)).env["COPILOT_HOME"])
    (first / "hooks" / "old.json").write_text("{}")
    (first / "mcp-config.json").write_text("{}")                          # the operator's servers never ride along
    second = Path(a.prepare(_ctx(tmp_path, task_id="8", session_dir=tmp_path / "state" / "sessions" / "8")).env["COPILOT_HOME"])
    assert first == second
    assert sorted(p.name for p in (second / "hooks").iterdir()) == ["automatos.json"]
    assert not (second / "mcp-config.json").exists()


def test_trust_given_in_the_take_over_terminal_is_undone_at_the_next_spawn(tmp_path, operator):
    a = _adapter(tmp_path, operator)
    home = Path(a.prepare(_ctx(tmp_path)).env["COPILOT_HOME"])
    config = json.loads((home / "config.json").read_text())
    (home / "config.json").write_text(json.dumps({**config, "trustedFolders": [str(tmp_path / "repo")]}))
    a.prepare(_ctx(tmp_path))
    assert json.loads((home / "config.json").read_text())["trustedFolders"] == []


def test_the_hooks_file_is_claudes_format_and_outlasts_the_shim(tmp_path, operator):
    """Copilot reads Claude-format hook files unmodified and sends Claude-shaped
    payloads (1.0.6, 1.0.21, 1.0.62) — the same file Claude Code would read."""
    home = Path(_adapter(tmp_path, operator).prepare(_ctx(tmp_path)).env["COPILOT_HOME"])
    hooks = json.loads((home / "hooks" / "automatos.json").read_text())["hooks"]
    assert hooks["PreToolUse"][0]["matcher"] == "*" and "matcher" not in hooks["Stop"][0]
    pre = hooks["PreToolUse"][0]["hooks"][0]
    assert pre["type"] == "command" and pre["command"].endswith("-m automatos_cli_host.hook_shim")
    assert pre["timeout"] == 600 and hooks["Stop"][0]["hooks"][0]["timeout"] == 60
    # these also name themselves on the command line, should a payload carry no name (S0.4)
    assert hooks["PermissionRequest"][0]["hooks"][0]["command"].endswith("--event PermissionRequest")
    assert hooks["Notification"][0]["hooks"][0]["command"].endswith("--event Notification")
    assert "PostCompact" not in hooks


# ── login (S1.2) ─────────────────────────────────────────────────────────────

def test_copilots_own_login_route_reads_the_account_pointer_only(tmp_path, operator):
    a = _adapter(tmp_path, operator)
    assert a.login() == ("octocat@github.com", "copilot", None)
    assert a.preflight() is None
    detected = a.detect()
    assert detected["served"] is True and detected["login"] == "octocat@github.com" and detected["login_route"] == "copilot"


def test_the_gh_route_and_its_command(tmp_path, operator):
    (operator / ".copilot" / "config.json").write_text("{}")
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return NS(returncode=0, stdout="github.com\n  ✓ Logged in to github.com account octocat (keyring)\n", stderr="")

    a = _adapter(tmp_path, operator, run=run)
    assert a.login() == ("octocat@github.com", "gh", None)
    assert calls == [["gh", "auth", "status", "--hostname", "github.com"]]
    assert "--show-token" not in calls[0]


def test_refusals_say_how_to_log_in(tmp_path, operator, monkeypatch):
    (operator / ".copilot" / "config.json").write_text("{}")
    assert _adapter(tmp_path, operator).preflight().code == "copilot_not_logged_in"
    monkeypatch.setenv("GH_TOKEN", "x")
    refusal = _adapter(tmp_path, operator).preflight()
    assert refusal.code == "copilot_not_logged_in" and "never carry tokens" in refusal.message
    plaintext = {"lastLoggedInUser": ACCOUNT, "storeTokenPlaintext": True}
    (operator / ".copilot" / "config.json").write_text(json.dumps(plaintext))
    refusal = _adapter(tmp_path, operator).preflight()
    assert refusal.code == "copilot_plaintext_token" and "credential store" in refusal.message


# ── preflight (S1.6) ─────────────────────────────────────────────────────────

def test_a_missing_or_old_binary_is_refused(tmp_path, operator):
    missing = CopilotAdapter(COPILOT, str(tmp_path / "nope"), home=operator)
    assert missing.preflight().code == "copilot_missing" and "copilot login" in missing.preflight().message
    assert _adapter(tmp_path, operator, version="1.0.56").preflight().code == "copilot_too_old"
    assert parse_version("GitHub Copilot CLI 1.0.91.") == (1, 0, 91) and parse_version("dev") is None


def test_an_organisation_that_runs_managed_hooks_only_is_refused(tmp_path, operator, monkeypatch):
    policy_dir = tmp_path / "policy.d"
    policy_dir.mkdir()
    (policy_dir / "org.json").write_text(json.dumps({"allowManagedHooksOnly": True}))
    monkeypatch.setattr(home_mod, "MANAGED_POLICY_DIRS", (policy_dir,))
    assert _adapter(tmp_path, operator).preflight().code == "copilot_managed_hooks_only"


def test_a_repository_that_switches_hooks_off_is_refused_for_its_tickets(tmp_path, operator):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".github" / "copilot").mkdir(parents=True)
    a = _adapter(tmp_path, operator)
    assert a.refuse_here(repo) is None
    (repo / ".github" / "copilot" / "settings.local.json").write_text(json.dumps({"disableAllHooks": True}))
    src = repo / "src"
    src.mkdir()
    assert a.refuse_here(src).code == "copilot_hooks_disabled_here"      # found walking up to the git root


# ── launch (S1.3, S2.1) ──────────────────────────────────────────────────────

def _argv(tmp_path, operator, **over):
    a = _adapter(tmp_path, operator)
    ctx = _ctx(tmp_path, **over)
    return a.launch_args(ctx, a.prepare(ctx)), ctx


# What Copilot 1.0.91's own parser refuses (F236, build 6: probed against the binary;
# tests/fake_copilot.py refuses the same pairs). This test used to pin --session-id
# beside --worktree, the PRD's unverified guess, which the binary refuses.
COPILOT_REFUSES = (("--resume", "--name"), ("--resume", "--worktree"), ("--session-id", "--worktree"))


def test_a_new_session_a_worktree_session_and_a_resumed_one(tmp_path, operator):
    args, ctx = _argv(tmp_path, operator, model="claude-sonnet-4.6")
    assert args[1:3] == ["--session-id", "sid-1"] and args[args.index("--name") + 1] == "automatos #7"
    for flag in COPILOT.required_args:
        assert flag in args
    assert args[args.index("--model") + 1] == "claude-sonnet-4.6"
    assert args[-2] == "-p" and "ticket" in args[-1]
    session.assert_args_honour_invariant(args, COPILOT.forbidden_args)
    in_worktree, _ = _argv(tmp_path, operator, worktree_name="automatos-7")
    assert in_worktree[in_worktree.index("--worktree") + 1] == "automatos-7"
    assert "--session-id" not in in_worktree and in_worktree[in_worktree.index("--name") + 1] == "automatos #7"
    resumed, _ = _argv(tmp_path, operator, resume_session_id="copilot-1", worktree_name="automatos-7")
    assert resumed[1:3] == ["--resume", "copilot-1"]
    assert not {"--session-id", "--name", "--worktree"} & set(resumed)    # F236: ticket 1273's resume


def test_no_launch_carries_a_pair_copilot_refuses(tmp_path, operator):
    for over in ({}, {"resume_session_id": "x"}, {"worktree_name": "w"}, {"resume_session_id": "x", "worktree_name": "w"},
                 {"plan_first": True, "resume_session_id": "x"}):
        args, _ = _argv(tmp_path, operator, **over)
        assert not [pair for pair in COPILOT_REFUSES if set(pair) <= set(args)], over


def test_a_terminal_on_a_resumed_session_is_not_renamed(tmp_path, operator):
    a = _adapter(tmp_path, operator)
    resumed = a.terminal_args("copilot", session_id="s-1", resume=True, system_prompt_path=None, model=None, task_id="7")
    assert resumed == ["copilot", "--resume", "s-1"]
    fresh = a.terminal_args("copilot", session_id="s-1", resume=False, system_prompt_path=None, model="gpt-5", task_id="7")
    assert fresh == ["copilot", "--session-id", "s-1", "--name", "automatos #7", "--model", "gpt-5"]


def test_a_session_may_sign_in_by_itself(tmp_path, operator):
    """F233 (2 Oct, build 5): --no-auto-login switches off the stored login and the gh
    fallback. With the env tokens stripped, every session failed "No authentication
    information found"."""
    args, _ = _argv(tmp_path, operator)
    assert "--no-auto-login" not in args and "--no-auto-login" not in COPILOT.required_args


def test_no_allow_flag_in_any_launch(tmp_path, operator):
    for over in ({}, {"model": "auto"}, {"resume_session_id": "x"}, {"worktree_name": "w"}):
        args, _ = _argv(tmp_path, operator, **over)
        assert not [a for a in args if a.startswith("--allow") or a in ("--yolo", "--assisted-approval")]


def test_the_automatos_tools_ride_a_0600_file_and_the_token_never_argv(tmp_path, operator):
    tools = {"names": ["board_summary"], "url": "http://127.0.0.1:8000/api/v1/session-tools/mcp", "token": "tok-123"}
    args, ctx = _argv(tmp_path, operator, session_tools=tools)
    value = args[args.index("--additional-mcp-config") + 1]
    assert value == f"@{ctx.session_dir / 'mcp.json'}"
    path = Path(value[1:])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    server = json.loads(path.read_text())["mcpServers"]["automatos"]
    assert server == {"type": "http", "url": tools["url"], "headers": {"Authorization": "Bearer tok-123"}, "tools": ["*"]}
    session.assert_secret_not_in_args(args, "tok-123")
    assert mcp_config(None) is None and mcp_config({"url": "u"}) is None


def test_the_deliverables_folder_is_named_to_copilots_own_path_check(tmp_path, operator):
    deliverables = tmp_path / "deliverables" / "sessions" / "7"
    args, ctx = _argv(tmp_path, operator, extra_dirs=(tmp_path / "state" / "sessions" / "7", deliverables))
    dirs = [args[i + 1] for i, a in enumerate(args) if a == "--add-dir"]
    assert str(ctx.session_dir) in dirs and str(deliverables) in dirs and len(dirs) == 2


# ── tools (S1.4) ─────────────────────────────────────────────────────────────

def _verdict(tmp_path, name, tool_input, mode="edits", tools=()):
    ctx = policy.PolicyContext(cwd=tmp_path, permission_mode=mode, session_tools=tools)
    return policy.decide(CopilotAdapter(COPILOT).tool_intent(name, tool_input), ctx)


@pytest.mark.parametrize("name, tool_input, cls", [
    ("Bash", {"command": "ls"}, ToolClass.SHELL), ("powershell", {"command": "dir"}, ToolClass.SHELL),
    ("Read", {"path": "a.txt"}, ToolClass.FILE_READ), ("view", {"path": "a.txt"}, ToolClass.FILE_READ),
    ("Write", {"path": "a.txt"}, ToolClass.FILE_WRITE), ("create", {"path": "a.txt"}, ToolClass.FILE_WRITE),
    ("Edit", {"path": "a.txt"}, ToolClass.FILE_WRITE), ("str_replace_editor", {"command": "view", "path": "a"}, ToolClass.FILE_READ),
    ("apply_patch", {"input": "*** Begin Patch\n*** Add File: a.txt\n+x\n*** End Patch\n"}, ToolClass.FILE_WRITE),
    ("apply_patch", {"command": "apply_patch", "actions": [{"actionLabel": "Edit", "path": "a.txt"}]}, ToolClass.FILE_WRITE),
    ("Grep", {"pattern": "x", "glob": "*.py"}, ToolClass.FILE_READ), ("glob", {"pattern": "**/*.md"}, ToolClass.FILE_READ),
    ("WebFetch", {"url": "https://x"}, ToolClass.WEB), ("update_todo", {}, ToolClass.BENIGN),
    ("report_intent", {}, ToolClass.BENIGN), ("read_bash", {}, ToolClass.BENIGN),
    ("task", {}, ToolClass.UNKNOWN), ("write_bash", {"input": "git push"}, ToolClass.UNKNOWN),
    ("exit_plan_mode", {}, ToolClass.UNKNOWN), ("github-mcp-server/create_issue", {}, ToolClass.UNKNOWN),
])
def test_each_copilot_tool_says_what_it_does(name, tool_input, cls):
    assert CopilotAdapter(COPILOT).tool_intent(name, tool_input).cls is cls


def test_an_edit_says_every_file_it_touches():
    a = CopilotAdapter(COPILOT)
    listed = a.tool_intent("apply_patch", {"actions": [{"path": "a.txt"}, {"path": "../b.txt"}]})
    assert listed.paths == ("a.txt", "../b.txt")
    assert a.tool_intent("apply_patch", {"command": "apply_patch"}).paths == ()     # the gate refuses a write naming nothing


def test_the_gate_judges_copilot_calls_like_any_other(tmp_path):
    escape = "*** Begin Patch\n*** Add File: ../../.ssh/authorized_keys\n+key\n*** End Patch\n"
    assert _verdict(tmp_path, "Edit", {"input": escape}).behavior == "deny"
    assert _verdict(tmp_path, "view", {"path": str(Path.home() / ".aws" / "credentials")}).behavior == "deny"
    assert _verdict(tmp_path, "Bash", {"command": "git push origin main"}, mode="auto").behavior == "deny"
    assert _verdict(tmp_path, "Write", {"path": str(tmp_path / "a.txt")}).allow
    assert _verdict(tmp_path, "github-mcp-server/create_issue", {}).behavior == "deny"
    assert _verdict(tmp_path, "mcp__other__x", {}).behavior == "deny"


@pytest.mark.parametrize("name", ["mcp__automatos__board_summary", "automatos-board_summary",
                                  "automatos/board_summary", "automatos__board_summary", "automatos.board_summary",
                                  "automatos(board_summary)"])
def test_an_automatos_tool_is_allowed_by_name_in_any_spelling(tmp_path, name):
    assert _verdict(tmp_path, name, {}, tools=("board_summary",)).allow
    assert _verdict(tmp_path, name, {}, tools=("other",)).behavior == "deny"


# ── the record (S1.5, S2.1) ──────────────────────────────────────────────────

def _record(tmp_path, *events):
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events) + "not json\n")
    return path


def test_usage_is_booked_per_model_in_tokens_and_ai_credits(tmp_path):
    path = _record(
        tmp_path,
        {"type": "session.start", "data": {"sessionId": "s"}},
        {"type": "assistant.usage", "data": {"model": "claude-sonnet-4.6", "inputTokens": 100, "outputTokens": 20,
                                             "cacheReadTokens": 50, "cacheWriteTokens": 5, "reasoningTokens": 3,
                                             "copilotUsage": {"totalNanoAiu": 1_500_000_000}}},
        {"type": "assistant.usage", "data": {"model": "gpt-5.4", "inputTokens": 10, "outputTokens": 2}},
        {"type": "assistant.message", "data": {"content": "Done."}},
        {"type": "session.shutdown", "data": {"totalPremiumRequests": 2}},
    )
    usage = read_events_usage(path)
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (110, 22, 132)
    assert usage["cache_read_input_tokens"] == 50 and usage["cache_creation_input_tokens"] == 5
    assert usage["reasoning_output_tokens"] == 3 and usage["ai_credits"] == 1.5 and usage["premium_requests"] == 2
    assert set(usage["per_model"]) == {"claude-sonnet-4.6", "gpt-5.4"}
    assert usage["per_model"]["gpt-5.4"]["input_tokens"] == 10
    assert "usd" not in json.dumps(usage) and "cost" not in json.dumps(usage)
    assert last_message(path) == "Done."


def test_an_automatos_server_that_did_not_load_is_read_from_the_record(tmp_path):
    loaded = _record(tmp_path, {"type": "session.mcp_servers_loaded",
                                "data": {"servers": [{"name": "automatos", "status": "connected"}]}})
    assert mcp_server_blocked(loaded, "automatos") is None
    failed = _record(tmp_path, {"type": "session.mcp_servers_loaded", "data": {"servers": [{"name": "automatos", "status": "pending"}]}},
                     {"type": "session.mcp_server_status_changed", "data": {"serverName": "automatos", "status": "failed"}})
    assert mcp_server_blocked(failed, "automatos") == "failed"
    note = CopilotAdapter(COPILOT).record_notes(failed)
    assert len(note) == 1 and "did not load" in note[0] and "failed" in note[0]


# ── the sandbox (S2.2) ───────────────────────────────────────────────────────

def test_a_sandboxed_host_seeds_copilots_sandbox_in_the_agents_settings(tmp_path, operator):
    a = _adapter(tmp_path, operator, sandbox=SessionSandbox())
    ctx = _ctx(tmp_path)
    prepared = a.prepare(ctx)
    assert "--sandbox" not in a.launch_args(ctx, prepared)                 # a saved sandbox.enabled turns it on
    settings = json.loads((Path(prepared.env["COPILOT_HOME"]) / "settings.json").read_text())
    assert settings["experimental"] is True                                  # still an experimental feature in 1.0.91
    block = settings["sandbox"]
    assert block["enabled"] is True and block["allowBypass"] is False and block["auth"] == {"git": False, "gh": False}
    files = block["userPolicy"]["filesystem"]
    assert str(ctx.session_dir) in files["readwritePaths"]
    assert os.path.expanduser("~/.ssh") in files["deniedPaths"]
    assert str(ctx.state_dir) not in files["deniedPaths"]          # never whole: it would beat the session's grants
    network = block["userPolicy"]["network"]
    assert network["allowLocalNetwork"] is False and {"registry.npmjs.org"} <= set(network["allowedHosts"])
    assert block["userPolicy"]["seatbelt"] == {"keychainAccess": False}


def _host_state(tmp_path):
    """The host's state dir as a host leaves it: its token, agent homes, a log, the
    hook socket, this session's folder and another session's."""
    state = tmp_path / "state"
    for folder in ("agents/58/.copilot", "sessions/6", "sessions/7"):
        (state / folder).mkdir(parents=True, exist_ok=True)
    for name in ("state.json", "host.log", "hooks.sock", "sessions/6/mcp.json", "sessions/7/ticket.md"):
        (state / name).write_text("x")
    return state


def test_the_hooks_and_the_ticket_are_reachable_inside_copilots_sandbox(tmp_path, operator):
    """F234 (build 5, tickets 1266-1271): Copilot runs its hooks inside its sandbox,
    whose profile lets a process reach a Unix socket only at a read-write path —
    and writes its denials after its grants, so the denied state dir beat both the
    socket's grant and the session dir's (ticket.md). The state is denied entry by
    entry around them; the session dir is listed once."""
    state = _host_state(tmp_path)
    a = _adapter(tmp_path, operator, sandbox=SessionSandbox())
    socket_path = state / "hooks.sock"
    ctx = _ctx(tmp_path, hook_socket=socket_path, extra_dirs=(state / "sessions" / "7",))
    settings = json.loads((Path(a.prepare(ctx).env["COPILOT_HOME"]) / "settings.json").read_text())
    files = settings["sandbox"]["userPolicy"]["filesystem"]
    assert files["readwritePaths"] == [str(ctx.session_dir), str(socket_path)]       # old: no socket, the dir twice
    denied = set(files["deniedPaths"])
    assert {str(state / n) for n in ("state.json", "host.log", "agents", "sessions/6")} <= denied
    assert not {str(state), str(state / "sessions"), str(ctx.session_dir), str(socket_path)} & denied


def test_deny_all_but_covers_everything_else_under_the_root(tmp_path):
    from automatos_cli_host.adapters.copilot_sandbox import deny_all_but

    state = _host_state(tmp_path)
    assert deny_all_but(state, [state / "sessions" / "7", state / "hooks.sock"]) == [
        state / "agents", state / "host.log", state / "sessions" / "6", state / "state.json"]
    assert deny_all_but(tmp_path / "nowhere", [state]) == []                     # nothing there to protect


def test_no_session_sandbox_drops_the_block(tmp_path, operator):
    prepared = _adapter(tmp_path, operator, sandbox=SessionSandbox(enabled=False)).prepare(_ctx(tmp_path))
    settings = json.loads((Path(prepared.env["COPILOT_HOME"]) / "settings.json").read_text())
    assert "sandbox" not in settings and "experimental" not in settings


def test_a_host_that_cannot_sandbox_is_refused_with_copilots_own_prerequisites(tmp_path, monkeypatch):
    from automatos_cli_host.adapters import copilot_sandbox
    tun = tmp_path / "tun"
    assert tools_missing("Linux", "/nowhere", tun=tun) == ["bwrap", "slirp4netns", "iptables", str(tun)]
    assert tools_missing("Darwin", "/nowhere") == ["sandbox-exec"]
    monkeypatch.setattr(copilot_sandbox, "missing", tools_missing)      # the real check (conftest answers "present")
    reason = unavailable_reason(SessionSandbox(), system="Linux", path="/nowhere")
    assert reason and "slirp4netns" in reason and "--no-session-sandbox" in reason
    assert unavailable_reason(SessionSandbox(enabled=False), system="Linux", path="/nowhere") is None


def test_the_registry_serves_copilot_through_its_adapter():
    assert isinstance(adapter_for("copilot"), CopilotAdapter)
    assert os.path.basename(COPILOT.binary) == "copilot"


# ── Copilot's own permission prompt (S1.4) ───────────────────────────────────

def test_a_rejudged_permission_request_is_answered_in_claudes_shape():
    a = CopilotAdapter(COPILOT)
    allow = a.render_response("PermissionRequest", Reply.allow())
    assert allow == {"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": {"behavior": "allow"}}}
    deny = a.render_response("PermissionRequest", Reply.deny("no"))
    assert deny["hookSpecificOutput"]["decision"] == {"behavior": "deny", "message": "no"}
    assert COPILOT.permission_request == "rejudge"


def test_the_gate_answers_a_cli_that_re_asks():
    from automatos_cli_host.permission_request import AllowedCalls, answer
    seen = AllowedCalls()
    seen.add("WebFetch", {"url": "https://x"})
    assert seen.holds("WebFetch", {"url": "https://x"}) and not seen.holds("WebFetch", {"url": "https://y"})
    assert answer("rejudge", allowed_before=True, behavior="ask", reason="r") == (True, "")
    assert answer("rejudge", allowed_before=False, behavior="allow", reason="") == (True, "")
    refused, why = answer("rejudge", allowed_before=False, behavior="ask", reason="held for the operator")
    assert refused is False and "held for the operator" in why            # never a card: a held call already was one
    assert answer("deny", allowed_before=True, behavior="allow", reason="")[0] is False   # Claude, Codex: unchanged


def test_copilots_own_config_with_its_comment_header_is_read(tmp_path, operator):
    """F233: Copilot writes config.json under "//" lines, which json.loads refused,
    so the operator's account pointer never reached the agent's home."""
    config = operator / ".copilot" / "config.json"
    config.write_text("// User settings belong in settings.json.\n// This file is managed automatically.\n"
                      + json.dumps({"lastLoggedInUser": ACCOUNT, "loggedInUsers": [ACCOUNT],
                                    "firstLaunchAt": "2026-07-30T10:00:00.000Z"}, indent=2))
    a = _adapter(tmp_path, operator)
    assert a.login() == ("octocat@github.com", "copilot", None)        # old: gh probe, then refused
    ctx = _ctx(tmp_path)
    home = Path(a.prepare(ctx).env["COPILOT_HOME"])
    assert json.loads((home / "config.json").read_text())["lastLoggedInUser"] == ACCOUNT


def test_a_url_inside_a_value_is_not_a_comment(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('// managed\n{"lastLoggedInUser": {"host": "https://github.com", "login": "octocat"}}')
    assert home_mod.read_json(path)["lastLoggedInUser"]["host"] == "https://github.com"


def test_the_account_pointer_is_read_in_either_shape():
    assert home_mod.account_of({"lastLoggedInUser": ACCOUNT}) == "octocat@github.com"
    assert home_mod.account_of({"lastLoggedInUser": "octocat@ghe.example"}) == "octocat@ghe.example"
    assert home_mod.account_of({}) is None and home_mod.account_of({"lastLoggedInUser": {"host": "x"}}) is None


def test_a_resumed_session_books_only_its_own_plan_units():
    from automatos_cli_host.transcript import usage_delta
    before = {"input_tokens": 100, "output_tokens": 40, "ai_credits": 2.0, "premium_requests": 1}
    after = {"input_tokens": 200, "output_tokens": 80, "ai_credits": 3.5, "premium_requests": 2}
    delta = usage_delta(after, before)
    assert (delta["total_tokens"], delta["ai_credits"], delta["premium_requests"]) == (140, 1.5, 1)
    assert "ai_credits" not in usage_delta({"input_tokens": 1}, None)     # a CLI that books none: nothing added
