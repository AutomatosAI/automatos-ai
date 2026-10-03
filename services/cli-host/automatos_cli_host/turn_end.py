"""How a session's turn ended, and what the ticket says about it (PRD-253 S0.2).

A preset says how its CLI's turn ends (``presets.TURN_ENDS``):

* ``stop_hook`` — the CLI's own ``Stop`` hook (Claude Code, Codex). The host ends
  the interactive process itself, so an exit before ``Stop`` is a failed turn.
* ``process_exit`` — the CLI exits when its turn is over: a print-mode CLI such as
  ``copilot -p``, and the seed tier, whose stdout is the answer.

A GATED CLI is one whose hooks are the host's policy gate (tiers native, hooks,
proxy). Whatever ends its turn, it has to prove the gate loaded — its
``SessionStart`` hook — or nothing it produced is reported: a gated CLI that ran
without its hooks ran without the gate.

Pure: no I/O and no process handling. ``session.py`` asks; this answers.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from .hook_shim import NOT_THE_HOST_MARK, UNREACHABLE_MARK
from .presets import REGISTRY, TIER_HOOKS, TIER_NATIVE, TIER_PROXY, TURN_END_STOP_HOOK

GATED_TIERS = (TIER_NATIVE, TIER_HOOKS, TIER_PROXY)
NO_SESSION_START = "no_session_start"
# F234: the shim's own deny in the CLI's output — its hooks never reached this host.
HOOKS_CUT_OFF = ("{cli}'s hooks could not reach this Automatos host, so the gate denied every call and no session "
                 "started. Restart the host; if it happens again, the CLI's sandbox is blocking the hook socket. "
                 "Last output:\n{tail}")
# F233: a CLI that could not sign in exits before its hooks load. That is a login
# to fix, not hooks switched off — in the CLIs' own words.
_SIGN_IN_FAILED = re.compile(
    r"no authentication information found|authentication token found but could not be validated"
    r"|not logged in|please run /login|invalid api key", re.I)
SIGN_IN_FAILED = "{how} Last output:\n{tail}"
SIGN_IN_HOW = "{cli} could not sign in on this machine. Sign in to it in a terminal, then retry."
# F236 (build 6): a command line the CLI's own parser refuses ("error: the argument
# '--resume' cannot be used with '--name'", then its "Usage:" line) is the host's
# mistake, never a hook, a login or a policy on this machine.
USAGE_REFUSED = ("{cli} refused the command line this Automatos host started it with ({what}). That is an "
                 "Automatos bug, not a setting on this machine: report it with this ticket. Last output:\n{tail}")
_COMMANDER_USAGE = ("error: unknown option", "error: option '", "error: missing required argument",
                    "error: too many arguments")
# The process failed before its gate loaded, for a reason its output doesn't name.
EXITED_BEFORE_SESSION = ("{cli} exited (code {code}) before its session started, so its hooks never loaded and "
                         "nothing it did is reported. Last output:\n{tail}")
# A print-mode CLI fires SessionEnd, flushes and exits. One still running this
# long after SessionEnd is terminated; its turn is over all the same.
EXIT_GRACE_AFTER_SESSION_END_SECONDS = 30.0
COMPLETED = "completed"
EXITED_BEFORE_STOP = "exited_before_stop"
UNGATED_EXIT = "ungated_exit"


def is_gated(preset: Any) -> bool:
    """The CLI's hooks are the gate, so a session must prove they loaded."""
    return getattr(preset, "tier", None) in GATED_TIERS


def startup_timeout(preset: Any, host_default: float) -> float:
    """The window for the gate's proof (SessionStart): the preset's own, else the host's."""
    own = getattr(preset, "startup_timeout_seconds", None)
    return float(own) if own else float(host_default)


def exit_reason(preset: Any, *, returncode: Optional[int], session_started: bool, stopped: bool) -> str:
    """Why the turn is over once the process has gone — or lingered past its
    SessionEnd (``returncode`` None).

    A hook-driven CLI is ended by the host after ``Stop``, so going first is a
    failed turn. An ungated (seed) CLI's exit IS its turn end. A gated print-mode
    CLI has finished only when its gate loaded, its turn reached ``Stop`` and the
    process did not fail."""
    if preset.turn_end == TURN_END_STOP_HOOK:
        return EXITED_BEFORE_STOP
    if not is_gated(preset):
        return COMPLETED
    if not session_started:
        return UNGATED_EXIT
    if stopped and returncode in (0, None):
        return COMPLETED
    return EXITED_BEFORE_STOP


def sign_in_failure(cli: str, tail: str) -> Optional[str]:
    """The sentence for a CLI that exited because it could not sign in — its own
    preset's how-to — else None."""
    if not _SIGN_IN_FAILED.search(tail or ""):
        return None
    probe = getattr(REGISTRY.get(cli), "auth_probe", None)
    how = probe.refusal if probe is not None else SIGN_IN_HOW.format(cli=cli)
    return SIGN_IN_FAILED.format(how=how, tail=tail)


def usage_error(tail: str) -> Optional[str]:
    """The CLI's own complaint about its command line, else None: an ``error:`` line
    followed by its ``Usage:`` line (clap: Copilot, Codex), or one of commander's
    usage errors (Claude Code). Line by line, no pattern over the whole tail."""
    lines = [line.strip() for line in (tail or "").splitlines() if line.strip()]
    for at, line in enumerate(lines):
        lower = line.lower()
        if not lower.startswith("error: "):
            continue
        if lower.startswith(_COMMANDER_USAGE) or (at + 1 < len(lines) and lines[at + 1].startswith("Usage:")):
            return line[len("error: "):]
    return None


def cause_in_tail(reason: str, cli: str, tail: str) -> Optional[str]:
    """What the CLI's own output says stopped it before its gate loaded — a command
    line it refused, its hooks unable to reach this host, or no sign-in — else None.
    Only then: no tool has run, so the tail is the CLI's own."""
    if reason not in (UNGATED_EXIT, NO_SESSION_START):
        return None
    refused = usage_error(tail)
    if refused:
        return USAGE_REFUSED.format(cli=cli, what=refused, tail=tail)
    flat = " ".join((tail or "").split())          # a terminal wraps long lines mid-phrase
    if any(mark in flat for mark in (UNREACHABLE_MARK, NOT_THE_HOST_MARK)):
        return HOOKS_CUT_OFF.format(cli=cli, tail=tail)
    return sign_in_failure(cli, tail)


def describe(reason: str, *, cli: str, returncode: Optional[int], tail: str, startup_window: float,
             session_timeout: float, stopped_by_host: Optional[str]) -> Tuple[str, Optional[str]]:
    """The ticket's status and, unless it succeeded, the sentence that says why."""
    if reason == COMPLETED:
        return "success", None
    cause = cause_in_tail(reason, cli, tail)
    if cause:
        return "error", cause
    if reason == "cancelled":
        # F015: the operator did not cancel it — the machine stopped serving it.
        return ("host_stopped", stopped_by_host) if stopped_by_host else ("cancelled", "cancelled by the operator")
    if reason == "timeout":
        return "error", f"session exceeded {int(session_timeout)} s"
    if reason == NO_SESSION_START:
        return "error", (
            f"{cli} did not start a session within {int(startup_window)} s — "
            f"it is probably showing a login screen or a dialog. Run `{cli}` in that directory once "
            f"and log in, then retry. Last output:\n{tail}"
        )
    if reason == UNGATED_EXIT and returncode not in (0, None):
        return "error", EXITED_BEFORE_SESSION.format(cli=cli, code=returncode, tail=tail)
    if reason == UNGATED_EXIT:
        # It ran a whole turn and exited cleanly, yet no hook ever fired.
        return "error", (
            f"{cli} ran without Automatos' gate — no SessionStart hook arrived, so nothing it produced is "
            "reported. Something switched its hooks off: a repository setting or an organisation policy. "
            f"Last output:\n{tail}"
        )
    return "error", f"{cli} exited (code {returncode}) before finishing the turn. Last output:\n{tail}"


__all__ = [
    "COMPLETED", "EXITED_BEFORE_STOP", "EXIT_GRACE_AFTER_SESSION_END_SECONDS", "GATED_TIERS", "UNGATED_EXIT",
    "EXITED_BEFORE_SESSION", "HOOKS_CUT_OFF", "NO_SESSION_START", "USAGE_REFUSED", "cause_in_tail", "describe",
    "exit_reason", "is_gated", "sign_in_failure", "startup_timeout", "usage_error",
]
