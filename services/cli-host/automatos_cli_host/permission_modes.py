"""The session's permission mode: the four Claude Code users already know.

* ``manual`` — every edit and every command off the Bash allowlist is a card.
* ``edits`` ("Edit automatically") — edits inside the session's folders run;
  a command off the allowlist is a card. The default when nothing says otherwise.
* ``plan`` — the session explores read-only and presents a plan; edits start once
  the operator approves it (PRD-253 Wave P, on EVERY CLI). Claude Code starts in
  its own plan mode and presents the plan with ``ExitPlanMode``: a card answered in
  time lets it carry on as ``edits`` in the same turn (Claude Code leaves plan mode
  when a hook approves). Every other CLI — and a Claude card nobody answered — ends
  its turn with the plan, which parks the ticket on the Plan card; the operator's
  Approve resumes the same session as ``edits``.
* ``auto`` — edits and commands off the allowlist run; only what the gate judges
  risky asks.

The gate's hard lines hold in every mode: never-allowed commands are refused,
the platform's secrets stay out of reach, the explicit ask-list and unresolved
paths still ask, and the session sandbox still bounds what a command touches.

The claim carries the agent's mode, else the workspace's (Settings → Session
mode). ``--permission-mode`` on this host overrides both for every session it runs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

MODE_MANUAL = "manual"
MODE_EDITS = "edits"
MODE_PLAN = "plan"
MODE_AUTO = "auto"
PERMISSION_MODES = (MODE_MANUAL, MODE_EDITS, MODE_PLAN, MODE_AUTO)
DEFAULT_MODE = MODE_EDITS
# ``--unlisted-bash`` (2026-09-18) is what ``--permission-mode`` replaced. Services
# installed with it restart on this code, so the old spelling still parses —
# until 2026-12-31; ``make cli-host-install`` writes the new one.
UNLISTED_BASH_MODES = {"allow": MODE_AUTO, "ask": MODE_EDITS}

log = logging.getLogger(__name__)

PLAN_FILENAME = "plan.md"
PLAN_TEXT_KEY = "plan"
# Claude Code keeps the plan in a file under its own folder and injects the text as
# ``plan``; when only ``planFilePath`` arrives, the file is read — from there only.
PLAN_FILE_KEY = "planFilePath"
PLAN_FILE_ROOT = Path.home() / ".claude"
MAX_PLAN_BYTES = 256 * 1024
MANUAL_EDIT = "Manual mode: every edit waits for the operator's approval"
PLAN_EDIT_REFUSED = ("Plan mode: edits wait until the operator approves your plan — "
                     "present it with ExitPlanMode first")
# PRD-253 Wave P: a CLI with no plan tool presents its plan as its final message.
PLAN_EDIT_REFUSED_TURN = ("Plan mode: edits wait until the operator approves your plan — "
                          "end your turn with the plan as your final message")
PLAN_CARD = "Plan mode: approve this plan to let the session start work"
# A plan card nobody answered in time: the plan goes to the operator as the Plan
# card instead, the ticket parks on it, and Approve resumes this same session.
PLAN_WITH_OPERATOR = ("Your plan has gone to the operator. Make no more changes and end your turn now — "
                      "this session resumes with their answer.")
# The event that carries a plan to the backend before the turn's result (the final
# events flush precedes the result): the backend files it as the Plan card.
PLAN_EVENT = "PlanReady"


def session_mode(host_choice: Optional[str], ticket_choice: Any, *, plan_approved: bool = False) -> str:
    """The mode this session runs in: the host's override, else the claim's, else Edit automatically.

    PRD-253 Wave P: Plan runs on every CLI, and the backend decides each claim's
    mode from the ticket's plan state — ``plan`` until the operator approves the
    plan, then ``edits`` — so the host no longer guesses from "resuming", nor runs
    a CLI without a plan mode of its own as Edit automatically. ``plan_approved``
    (from the claim) lets a host whose own override is Plan carry on once the
    ticket's plan is approved, instead of planning it again forever."""
    if host_choice in PERMISSION_MODES:
        mode = host_choice
    elif ticket_choice in PERMISSION_MODES:
        mode = ticket_choice
    else:
        mode = DEFAULT_MODE
    if mode == MODE_PLAN and plan_approved:
        return MODE_EDITS
    return mode


def plan_text(tool_input: Mapping[str, Any], plan_root: Path = PLAN_FILE_ROOT) -> str:
    """The plan a session presents with ``ExitPlanMode``; empty when the CLI sends none."""
    if not isinstance(tool_input, Mapping):
        return ""
    value = tool_input.get(PLAN_TEXT_KEY)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _plan_file_text(tool_input.get(PLAN_FILE_KEY), plan_root)


def _plan_file_text(raw: Any, plan_root: Path) -> str:
    if not isinstance(raw, str) or not raw:
        return ""
    try:
        path = Path(raw).expanduser().resolve()
        if not path.is_relative_to(plan_root.resolve()) or path.suffix != ".md":
            return ""
        with path.open("rb") as handle:
            return handle.read(MAX_PLAN_BYTES).decode("utf-8", errors="replace").strip()
    except OSError as exc:
        log.warning("the session's plan file %s was not read: %s", raw, exc)
        return ""


def save_plan(folder: Optional[Path], text: str) -> Optional[Path]:
    """Keep the whole plan beside the ticket's deliverables: the card shows only its start."""
    if folder is None or not text:
        return None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / PLAN_FILENAME
        path.write_text(text + "\n", encoding="utf-8")
        return path
    except OSError as exc:
        log.warning("the session's plan was not saved in %s: %s", folder, exc)
        return None
