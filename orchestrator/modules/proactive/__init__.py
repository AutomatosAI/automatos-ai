"""Proactive Auto (PRD-268): the check-in rules, as pure functions with no I/O.

The services (``services/proactive/``) read the database and pass in what they read;
these functions decide. Nothing here mutates its arguments.

- :mod:`modules.proactive.window`: what a check-in reads, since its last successful
  run, else the last 24 hours (§6.2.1).
- :mod:`modules.proactive.dedupe`: subject keys, normalised titles, and the rule that
  a suggestion repeats neither an open one nor one hidden in the last 14 days (§6.2.4).
- :mod:`modules.proactive.caps`: the §6.9 caps and the checks that refuse past them,
  a check-in's minimum interval among them.
- :mod:`modules.proactive.streaks`: failing streaks, and the interval that doubles
  from the third failure up to 24 hours (§6.2.2).
- :mod:`modules.proactive.provenance`: a suggestion cites only reads its own run
  made (§6.2.3).
- :mod:`modules.proactive.heartbeats`: which heartbeats become check-ins, an
  agent's ``check_in`` or ``act`` and the orchestrator's proactive levels (K10).
"""
from modules.proactive.caps import (
    CapRefusal,
    Caps,
    clamp_interval,
    interval_refusal,
    oldest_to_expire,
    run_refusal,
    suggestion_refusal,
)
from modules.proactive.dedupe import (
    Candidate,
    agent_error_key,
    board_number,
    error_class_name,
    find_duplicate,
    is_duplicate,
    normalise_title,
    ticket_key,
    title_key,
)
from modules.proactive.heartbeats import (
    CHECK_IN_LEVELS,
    HEARTBEAT_ACT,
    HEARTBEAT_CHECK_IN,
    heartbeat_mode,
    stated_mode,
)
from modules.proactive.provenance import provenance_ok
from modules.proactive.streaks import consecutive_failures, is_due, next_interval
from modules.proactive.window import check_in_window, last_success_at

__all__ = [
    "CHECK_IN_LEVELS",
    "CapRefusal",
    "Caps",
    "Candidate",
    "HEARTBEAT_ACT",
    "HEARTBEAT_CHECK_IN",
    "agent_error_key",
    "board_number",
    "check_in_window",
    "clamp_interval",
    "consecutive_failures",
    "error_class_name",
    "find_duplicate",
    "heartbeat_mode",
    "interval_refusal",
    "is_due",
    "is_duplicate",
    "last_success_at",
    "next_interval",
    "normalise_title",
    "oldest_to_expire",
    "provenance_ok",
    "run_refusal",
    "stated_mode",
    "suggestion_refusal",
    "ticket_key",
    "title_key",
]
