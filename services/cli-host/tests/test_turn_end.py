"""How a turn ends — and that a gated CLI proves its gate loaded whatever ends it
(PRD-253 S0.2).

A print-mode CLI (``copilot -p``) ends its turn by exiting. Before this, the
proof that the gate loaded (its ``SessionStart`` hook) was only asked of CLIs
whose turn ends on ``Stop``, and any exit of a print-mode CLI counted as a
finished turn — so one whose hooks were switched off would have run ungated and
reported success. The real loop runs here against a stand-in print-mode CLI.
"""
from __future__ import annotations

import uuid

import pytest

from automatos_cli_host import adapters as adapters_pkg
from automatos_cli_host import presets, turn_end
from automatos_cli_host.adapters.base import PresetAdapter
from automatos_cli_host.config import HostConfig
from automatos_cli_host.hook_server import HookServer
from automatos_cli_host.presets import CLAUDE, CODEX
from automatos_cli_host.session import Session

from conftest import FAKE_PRINT

PRINT_CLI = presets.CliPreset(
    id="printcli", label="Print CLI", binary="printcli",
    tier=presets.TIER_HOOKS, turn_end=presets.TURN_END_PROCESS_EXIT,
    initial_prompt=presets.PROMPT_FLAG, initial_prompt_flag="-p",
    hook_events=presets.BUS_EVENTS, startup_timeout_seconds=3,
    auth_probe=presets.AuthProbe(kind="none", code="printcli_unused", refusal="unused"),
)
SEED_CLI = presets.CliPreset(id="seedcli", label="Seed CLI", binary="seedcli",
                             tier=presets.TIER_SEED, turn_end=presets.TURN_END_PROCESS_EXIT)


class _PrintAdapter(PresetAdapter):
    def logged_in(self):
        return None   # a stand-in: there is nothing to log in to


@pytest.fixture
def print_cli(monkeypatch):
    monkeypatch.setitem(presets.REGISTRY, PRINT_CLI.id, PRINT_CLI)
    monkeypatch.setitem(adapters_pkg._ADAPTERS, PRINT_CLI.id, _PrintAdapter)
    monkeypatch.setattr(turn_end, "EXIT_GRACE_AFTER_SESSION_END_SECONDS", 0.5)
    monkeypatch.delenv("FAKE_PRINT_SCENARIO", raising=False)


def _run(short_tmp, monkeypatch, scenario):
    monkeypatch.setenv("FAKE_PRINT_SCENARIO", scenario)
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = HostConfig(state_dir=short_tmp / "state", cli_binaries={"printcli": str(FAKE_PRINT)},
                     use_worktrees=False, session_timeout_seconds=60)
    ticket = {"task_id": 91, "attempt": 1, "session_id": str(uuid.uuid4()), "agent_id": 3, "agent_name": "Pat",
              "title": "Print", "prompt": "do the thing", "cwd": str(workdir), "provider": "printcli",
              "allowed_tools": []}
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


# ── the decision, as a table ─────────────────────────────────────────────────

@pytest.mark.parametrize("preset, returncode, started, stopped, expected", [
    (CLAUDE, 0, True, False, "exited_before_stop"),    # the host ends a hook-driven CLI; going first failed
    (CODEX, 0, True, True, "exited_before_stop"),
    (SEED_CLI, 0, False, False, "completed"),           # no gate to prove: the exit is the turn's end
    (SEED_CLI, 3, False, False, "completed"),
    (PRINT_CLI, 0, True, True, "completed"),
    (PRINT_CLI, None, True, True, "completed"),         # said goodbye, then lingered: the turn is over
    (PRINT_CLI, 0, False, False, "ungated_exit"),       # the gate never loaded
    (PRINT_CLI, 0, False, True, "ungated_exit"),
    (PRINT_CLI, 3, True, True, "exited_before_stop"),
    (PRINT_CLI, 0, True, False, "exited_before_stop"),  # it never reached Stop
])
def test_why_a_turn_is_over(preset, returncode, started, stopped, expected):
    assert turn_end.exit_reason(preset, returncode=returncode, session_started=started, stopped=stopped) == expected


def test_which_clis_must_prove_their_gate():
    assert turn_end.is_gated(CLAUDE) and turn_end.is_gated(CODEX) and turn_end.is_gated(PRINT_CLI)
    assert not turn_end.is_gated(SEED_CLI)
    assert turn_end.startup_timeout(PRINT_CLI, 180) == 3
    assert turn_end.startup_timeout(CLAUDE, 180) == 180     # no window of its own: the host's


def test_an_ungated_run_reports_nothing_it_produced():
    status, error = turn_end.describe("ungated_exit", cli="printcli", returncode=0, tail="tail",
                                      startup_window=3, session_timeout=60, stopped_by_host=None)
    assert status == "error" and "without Automatos' gate" in error


COPILOT_SIGNED_OUT = ("Error: No authentication information found.\n\nCopilot can be authenticated with GitHub "
                      "using an OAuth Token or a Fine-Grained Personal Access Token.")


def test_a_cli_that_could_not_sign_in_says_so_not_that_hooks_are_off():
    """F233 (build 5): tickets 1263-1265 said a repository setting or an organisation
    policy had switched the hooks off. Copilot had not signed in."""
    from automatos_cli_host.presets import COPILOT

    status, error = turn_end.describe("ungated_exit", cli="copilot", returncode=1, tail=COPILOT_SIGNED_OUT,
                                      startup_window=30, session_timeout=60, stopped_by_host=None)
    assert status == "error" and error.startswith(COPILOT.auth_probe.refusal)
    assert "without Automatos' gate" not in error and "No authentication information found" in error
    unknown = turn_end.sign_in_failure("newcli", "Error: not logged in")
    assert unknown.startswith("newcli could not sign in on this machine")


@pytest.mark.parametrize("reason", ["no_session_start", "ungated_exit"])
def test_hooks_that_could_not_reach_the_host_say_so(reason):
    """F234: 1266 said "probably showing a login screen", 1267/1268 "a repository
    setting or an organisation policy". Every call had been denied by the shim."""
    tail = ("✗ Read ticket.md (sandbox policy) ~/.automatos/cli-host/sessions/1266/ticket.md\n"
            "   Denied by preToolUse hook: Automatos CLI host is\n   unreachable — call denied")   # wrapped, as on 1266
    status, error = turn_end.describe(reason, cli="copilot", returncode=None, tail=tail, startup_window=30,
                                      session_timeout=60, stopped_by_host=None)
    assert status == "error" and error.startswith("copilot's hooks could not reach this Automatos host")
    assert "login screen" not in error and "organisation policy" not in error


COPILOT_REFUSED_ARGS = ("error: the argument '--resume [<value>]' cannot be used with '--name <name>'\n\n"
                        "Usage: copilot --resume [<value>] --prompt <text>\n\nFor more information, try '--help'.")


def test_a_command_line_the_cli_refuses_is_named_as_ours_not_a_policy():
    """F236 (build 6): ticket 1273's resume died on Copilot's own usage error, and its
    headline blamed "a repository setting or an organisation policy"."""
    status, error = turn_end.describe("ungated_exit", cli="copilot", returncode=2, tail=COPILOT_REFUSED_ARGS,
                                      startup_window=30, session_timeout=60, stopped_by_host=None)
    assert status == "error"
    assert error.startswith("copilot refused the command line this Automatos host started it with")
    assert "cannot be used with '--name <name>'" in error and "Automatos bug" in error
    assert "organisation policy" not in error and "could not sign in" not in error
    assert turn_end.usage_error("error: unknown option '--yolo'") == "unknown option '--yolo'"   # commander's shape
    assert turn_end.usage_error("Error: No session, task, or name matched 'x'.\n\nTo resume: copilot --resume=<id>") is None
    assert turn_end.usage_error(COPILOT_SIGNED_OUT) is None


def test_an_exit_before_the_session_started_names_no_cause_it_has_not_seen():
    status, error = turn_end.describe("ungated_exit", cli="copilot", returncode=1, tail="Error: something else",
                                      startup_window=30, session_timeout=60, stopped_by_host=None)
    assert status == "error" and error.startswith("copilot exited (code 1) before its session started")
    assert "organisation policy" not in error


def test_a_turn_that_ran_tools_is_never_read_as_a_sign_in_failure():
    """Past the gate, the tail can be a tool's output ("gh: not logged in")."""
    status, error = turn_end.describe("exited_before_stop", cli="copilot", returncode=1, tail="gh: not logged in",
                                      startup_window=30, session_timeout=60, stopped_by_host=None)
    assert status == "error" and error.startswith("copilot exited (code 1) before finishing the turn")


# ── the real loop against a stand-in print-mode CLI ─────────────────────────

def test_a_print_mode_turn_that_proved_its_gate_succeeds(short_tmp, monkeypatch, print_cli, env_clean):
    s, out = _run(short_tmp, monkeypatch, "happy")
    assert out.status == "success" and out.exit_reason == "completed", out
    assert out.result_text == "Done: the turn's answer."      # the Stop hook's text, not stdout
    assert s.session_started.is_set() and s.stopped.is_set()


def test_a_print_mode_cli_that_ran_without_its_hooks_reports_nothing(short_tmp, monkeypatch, print_cli, env_clean):
    s, out = _run(short_tmp, monkeypatch, "no-hooks")
    assert out.status == "error" and out.exit_reason == "ungated_exit", out
    assert out.result_text == "" and out.files_touched == []
    assert "without Automatos' gate" in (out.error or "")
    assert not s.session_started.is_set()


def test_a_print_mode_cli_that_fails_or_never_reaches_stop_is_an_error(short_tmp, monkeypatch, print_cli, env_clean):
    _, failed = _run(short_tmp, monkeypatch, "fail")
    assert failed.status == "error" and failed.exit_reason == "exited_before_stop"
    assert "exited (code 3)" in (failed.error or "")
    _, unfinished = _run(short_tmp, monkeypatch, "no-stop")
    assert unfinished.status == "error" and unfinished.exit_reason == "exited_before_stop"


def test_a_print_mode_cli_that_lingers_after_its_session_end_is_ended(short_tmp, monkeypatch, print_cli, env_clean):
    s, out = _run(short_tmp, monkeypatch, "linger")
    assert out.status == "success" and out.exit_reason == "completed", out
    assert s.proc is not None and s.proc.poll() is not None       # terminated by the host


def test_the_gate_proof_window_is_the_presets_own(short_tmp, monkeypatch, print_cli, env_clean):
    s, out = _run(short_tmp, monkeypatch, "hang")
    assert out.status == "error" and out.exit_reason == "no_session_start", out
    assert "within 3 s" in (out.error or "")
    assert s.proc is not None and s.proc.poll() is not None
