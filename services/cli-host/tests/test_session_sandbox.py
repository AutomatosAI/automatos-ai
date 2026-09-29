"""The OS sandbox under a Claude session (``sandbox.py``).

The gate judges what a command line names; an allowed ``python x.py``,
``npm run x``, ``pytest`` or ``git commit`` runs code the session wrote, which
the gate never sees. These prove the session's ``--settings`` file switches on
Claude Code's own sandbox with no way out, denies the credential stores, the
host's state and the platform's secrets, reaches only the registries — and that
a host that cannot sandbox is not served, rather than running unsandboxed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from automatos_cli_host import sandbox
from automatos_cli_host.adapters import adapter_for
from automatos_cli_host.adapters.base import LaunchContext
from automatos_cli_host.config import parse_args
from automatos_cli_host.sandbox import DEFAULT_ALLOWED_DOMAINS, SessionSandbox, claude_settings
from automatos_cli_host.sandbox import missing_tools as real_missing_tools  # before conftest stubs it
from automatos_cli_host.service import service_argv

from conftest import FAKE_CLAUDE
from test_session_fake_claude import _run, _ticket


def _missing(*tools):
    return lambda system=None, path=None: list(tools)


def _launch(short_tmp) -> LaunchContext:
    cwd = short_tmp / "ws"
    cwd.mkdir(exist_ok=True)
    session_dir = short_tmp / "state" / "sessions" / "7"
    return LaunchContext(cwd=cwd, session_dir=session_dir, ticket_path=session_dir / "ticket.md",
                         system_prompt_path=session_dir / "system_prompt.md", task_id="7", session_id="s",
                         state_dir=short_tmp / "state")


def _written_settings(short_tmp, sandboxed: Optional[SessionSandbox]) -> dict:
    adapter = adapter_for("claude", {"claude": str(FAKE_CLAUDE)}, sandboxed)
    prepared = adapter.prepare(_launch(short_tmp))
    return json.loads(Path(prepared.args[prepared.args.index("--settings") + 1]).read_text())


# ── the settings block ───────────────────────────────────────────────────────

def test_the_sandbox_is_on_with_no_way_out_and_the_gate_still_decides():
    block = claude_settings(SessionSandbox())
    assert block["enabled"] is True
    assert block["failIfUnavailable"] is True             # cannot sandbox → does not start
    assert block["allowUnsandboxedCommands"] is False     # no dangerouslyDisableSandbox retry
    assert block["autoAllowBashIfSandboxed"] is False     # the PreToolUse gate still judges every call


def test_credential_stores_host_state_and_platform_secrets_are_unreadable(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    platform_root = tmp_path / "automatos"
    deny = claude_settings(SessionSandbox(), secret_roots=(platform_root,),
                           off_limits=(tmp_path / "state",))["filesystem"]["denyRead"]
    for store in ("~/.aws", "~/.ssh", "~/.config/gh", "~/.git-credentials", "~/.claude/.credentials.json"):
        assert store in deny
    assert str(tmp_path / "state") in deny
    assert f"{platform_root}/.env" in deny and f"{platform_root}/.credential_key" in deny


def test_the_claude_login_is_unreadable_where_claude_config_dir_moves_it(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    assert str(tmp_path / "cfg" / ".credentials.json") in sandbox.deny_read()


def test_the_network_is_the_registries_plus_what_the_operator_adds_and_nothing_is_asked():
    network = claude_settings(SessionSandbox(allowed_domains=("pypi.org", "github.com")))["network"]
    assert network == {"allowedDomains": ["pypi.org", "github.com"], "strictAllowlist": True}
    assert "registry.npmjs.org" in DEFAULT_ALLOWED_DOMAINS


# ── what the session is launched with ────────────────────────────────────────

def test_a_sandboxed_host_writes_the_sandbox_into_the_session_settings(short_tmp, fake_home):
    settings = _written_settings(short_tmp, SessionSandbox())
    assert settings["sandbox"]["enabled"] is True and "PreToolUse" in settings["hooks"]
    assert str(short_tmp / "state") in settings["sandbox"]["filesystem"]["denyRead"]


def test_no_session_sandbox_writes_hooks_only(short_tmp, fake_home):
    assert "sandbox" not in _written_settings(short_tmp, SessionSandbox(enabled=False))
    assert "sandbox" not in _written_settings(short_tmp, None)  # the operator's own terminal


# ── a host that cannot sandbox is not served ─────────────────────────────────

def test_linux_needs_bubblewrap_and_socat_and_macos_needs_nothing(tmp_path):
    assert real_missing_tools(system="Linux", path=str(tmp_path)) == ["bwrap", "socat"]
    assert real_missing_tools(system="Darwin", path=str(tmp_path)) == []


def test_without_the_tools_claude_is_not_served_and_says_how_to_fix_it(short_tmp, fake_home, monkeypatch):
    monkeypatch.setattr(sandbox, "missing_tools", _missing("bwrap", "socat"))
    adapter = adapter_for("claude", {"claude": str(FAKE_CLAUDE)}, SessionSandbox())
    refusal = adapter.preflight()
    assert refusal is not None and refusal.code == "claude_sandbox_unavailable"
    assert "apt-get install bubblewrap socat" in refusal.message and "Missing: bwrap, socat" in refusal.message
    detected = adapter.detect()
    assert detected["served"] is False and detected["reason"] == refusal.message


def test_no_session_sandbox_serves_without_the_tools(short_tmp, fake_home, monkeypatch):
    monkeypatch.setattr(sandbox, "missing_tools", _missing("bwrap"))
    assert adapter_for("claude", {"claude": str(FAKE_CLAUDE)}, SessionSandbox(enabled=False)).preflight() is None


def test_a_ticket_on_a_host_that_cannot_sandbox_never_starts_a_process(short_tmp, fake_home, env_clean, monkeypatch):
    monkeypatch.setattr(sandbox, "missing_tools", _missing("bwrap"))
    workdir = short_tmp / "ws"
    workdir.mkdir()
    s, out = _run(short_tmp, _ticket(workdir))
    assert out.status == "error" and out.exit_reason == "claude_sandbox_unavailable"
    assert s.proc is None


# ── the flags, and the service that keeps them ───────────────────────────────

def test_the_sandbox_is_on_by_default_and_the_flags_shape_it():
    assert parse_args([]).session_sandbox == SessionSandbox()
    assert parse_args(["--no-session-sandbox"]).session_sandbox.enabled is False
    widened = parse_args(["--session-allow-domain", "github.com", "--session-allow-domain", "pypi.org"])
    assert widened.session_sandbox.allowed_domains == (*DEFAULT_ALLOWED_DOMAINS, "github.com")


def test_the_installed_service_keeps_the_sandbox_choice(tmp_path):
    argv = ["--dir", str(tmp_path)]
    assert "--no-session-sandbox" not in service_argv(parse_args(argv))
    assert "--no-session-sandbox" in service_argv(parse_args([*argv, "--no-session-sandbox"]))
    kept = service_argv(parse_args([*argv, "--session-allow-domain", "github.com"]))
    assert kept[kept.index("--session-allow-domain") + 1] == "github.com"
    assert kept.count("--session-allow-domain") == 1  # the registries are the default, not repeated
