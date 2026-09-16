"""One ticket through a supervised Codex session, against the fake ``codex`` —
the same turn as the Claude suite, in Codex's spelling (design §6): the isolated
per-agent home with the operator's login linked in and their config seeded,
hooks from config.toml tables, the gate on Codex's tools (a patch inside cwd
allowed, ``exec_command`` push denied), the contract as additionalContext, the
session id learned from the CLI, usage from the rollout's cumulative record,
resume by subcommand into the same home and the same rollout."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from automatos_cli_host.config import HostConfig
from automatos_cli_host.hook_server import HookServer
from automatos_cli_host.session import Session

from conftest import FAKE_CODEX


def _cfg(short_tmp, **over) -> HostConfig:
    base = dict(state_dir=short_tmp / "state", cli_binaries={"codex": str(FAKE_CODEX)}, use_worktrees=False,
                session_timeout_seconds=120)
    base.update(over)
    return HostConfig(**base)


def _ticket(workdir: Path, **over) -> dict:
    t = {"task_id": 77, "attempt": 1, "session_id": str(uuid.uuid4()), "agent_id": 15, "agent_name": "Bob",
         "title": "Say hi", "prompt": "OBJECTIVE: write hello.txt\nOUTPUT: the file\nBOUNDARIES: this dir",
         "cwd": str(workdir), "model": "gpt-5.5", "allowed_tools": [], "provider": "codex"}
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


def test_happy_turn_in_codexs_spelling(short_tmp, fake_codex_home, env_clean):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    operator_config_before = (fake_codex_home / ".codex" / "config.toml").read_bytes()
    ticket = _ticket(workdir)
    s, out = _run(short_tmp, ticket)

    assert out.status == "success", out
    assert out.exit_reason == "completed"
    assert "Done. Wrote hello.txt" in out.result_text
    assert "Push was denied by policy" in out.result_text                   # exec_command → SHELL → never-allowed
    assert "Contract seen" in out.result_text                               # additionalContext rode UserPromptSubmit
    assert (workdir / "hello.txt").read_text() == "hi\n"
    assert out.files_touched == ["hello.txt"]                               # from the patch header, via ToolIntent
    stages = sorted(d["stage"] for d in out.permission_denials)
    assert stages == ["PermissionRequest", "PreToolUse"]
    assert any("git push" in json.dumps(d) for d in out.permission_denials)
    # usage: the LAST cumulative record, the model from turn_context, cached → cache_read
    assert out.usage["input_tokens"] == 100 and out.usage["output_tokens"] == 40 and out.usage["total_tokens"] == 140
    assert out.usage["cache_read_input_tokens"] == 20 and out.usage["reasoning_output_tokens"] == 5
    assert out.usage["model"] == "gpt-5.5" and "usd" not in json.dumps(out.usage)
    # the id is the CLI's own, learned on SessionStart — not the backend's correlation id
    assert out.session_id and out.session_id != ticket["session_id"]
    assert out.transcript_path and Path(out.transcript_path).exists()
    assert s.proc is not None and s.proc.poll() is not None

    # The isolated home: per agent, under the host's state dir; the operator's ~/.codex untouched.
    home = short_tmp / "state" / "agents" / "15" / ".codex"
    assert str(Path(out.transcript_path)).startswith(str(home / "sessions"))
    assert (home / "auth.json").is_symlink() and os.readlink(home / "auth.json") == str(fake_codex_home / ".codex" / "auth.json")
    config = (home / "config.toml").read_text()
    assert config.startswith('model = "gpt-fake"')                           # seeded from the operator's own
    assert '[projects."/somewhere/else"]' in config                          # their trust carried over
    assert f'[projects."{workdir.resolve()}"]' in config or f'[projects."{workdir}"]' in config   # ours added
    assert "[[hooks.PreToolUse]]" in config and "timeout = 30" in config and "timeout = 0" not in config
    assert "[[hooks.Notification]]" not in config                            # Codex has no such event
    assert (fake_codex_home / ".codex" / "config.toml").read_bytes() == operator_config_before
    assert not (workdir / ".automatos").exists() and not (workdir / ".codex").exists()
    events = []
    while not s.events.empty():
        events.append(s.events.get_nowait())
    names = [e["event"] for e in events]
    assert names[0] == "SessionStart" and "Stop" in names and "PreToolUse" in names
    assert any(e.get("subject") == "git push origin main" for e in events)   # the subject is the command, from the intent


def test_an_api_key_login_is_refused_before_any_process(short_tmp, fake_codex_home, env_clean):
    (fake_codex_home / ".codex" / "auth.json").write_text(json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "not-a-secret-fixture"}))
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    s, out = _run(short_tmp, _ticket(workdir))
    assert out.status == "error" and out.exit_reason == "codex_api_key_login" and s.proc is None
    assert "ChatGPT" in (out.error or "") and "codex login" in (out.error or "")


def test_not_logged_in_and_not_installed_are_named(short_tmp, fake_codex_home, env_clean):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    (fake_codex_home / ".codex" / "auth.json").unlink()
    _, out = _run(short_tmp, _ticket(workdir))
    assert out.exit_reason == "codex_not_logged_in" and "codex login" in (out.error or "")
    _, out = _run(short_tmp, _ticket(workdir), cfg=_cfg(short_tmp, cli_binaries={"codex": str(short_tmp / "no-codex")}))
    assert out.exit_reason == "codex_missing" and "not installed" in (out.error or "")


def test_resume_continues_the_same_home_and_rollout_and_books_only_this_turn(short_tmp, fake_codex_home, env_clean):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    first_s, first = _run(short_tmp, _ticket(workdir))
    assert first.status == "success"
    second_ticket = _ticket(workdir, task_id=78, session_id=str(uuid.uuid4()), resume_session_id=first.session_id)
    second_s, second = _run(short_tmp, second_ticket)
    assert second.status == "success", second
    assert second.session_id == first.session_id                             # `codex resume <id>` — a subcommand
    assert second.transcript_path == first.transcript_path                   # the same rollout, appended
    assert second.usage["total_tokens"] == 140 and second.usage["input_tokens"] == 100   # this turn only, not 280
    assert second.usage["per_model"] == {"gpt-5.5": {"input_tokens": 100, "output_tokens": 40,
                                                     "cache_read_input_tokens": 20, "cache_creation_input_tokens": 0}}
    argv = list(second_s.proc.args)
    assert argv[1:3] == ["resume", first.session_id] and "--worktree" not in argv


def test_a_codex_session_exiting_early_is_an_error_with_diagnostics(short_tmp, fake_codex_home, env_clean, monkeypatch):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("FAKE_CODEX_SCENARIO", "exit-early")
    _, out = _run(short_tmp, _ticket(workdir))
    assert out.status == "error" and out.exit_reason == "exited_before_stop" and "code 3" in (out.error or "")
