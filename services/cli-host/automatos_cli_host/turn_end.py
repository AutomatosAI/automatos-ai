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

from typing import Any, Optional, Tuple

from .presets import TIER_HOOKS, TIER_NATIVE, TIER_PROXY, TURN_END_STOP_HOOK

GATED_TIERS = (TIER_NATIVE, TIER_HOOKS, TIER_PROXY)
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


def describe(reason: str, *, cli: str, returncode: Optional[int], tail: str, startup_window: float,
             session_timeout: float, stopped_by_host: Optional[str]) -> Tuple[str, Optional[str]]:
    """The ticket's status and, unless it succeeded, the sentence that says why."""
    if reason == COMPLETED:
        return "success", None
    if reason == "cancelled":
        # F015: the operator did not cancel it — the machine stopped serving it.
        return ("host_stopped", stopped_by_host) if stopped_by_host else ("cancelled", "cancelled by the operator")
    if reason == "timeout":
        return "error", f"session exceeded {int(session_timeout)} s"
    if reason == "no_session_start":
        return "error", (
            f"{cli} did not start a session within {int(startup_window)} s — "
            f"it is probably showing a login screen or a dialog. Run `{cli}` in that directory once "
            f"and log in, then retry. Last output:\n{tail}"
        )
    if reason == UNGATED_EXIT:
        return "error", (
            f"{cli} ran without Automatos' gate — no SessionStart hook arrived, so nothing it produced is "
            "reported. Something switched its hooks off: a repository setting or an organisation policy. "
            f"Last output:\n{tail}"
        )
    return "error", f"{cli} exited (code {returncode}) before finishing the turn. Last output:\n{tail}"


__all__ = [
    "COMPLETED", "EXITED_BEFORE_STOP", "EXIT_GRACE_AFTER_SESSION_END_SECONDS", "GATED_TIERS", "UNGATED_EXIT",
    "describe", "exit_reason", "is_gated", "startup_timeout",
]
