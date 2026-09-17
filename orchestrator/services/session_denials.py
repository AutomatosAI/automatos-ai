"""PRD-245 S0.3 — what a session's refused tool call MEANS (decision D6).

The host reports every refusal the same way (``tool``, ``stage``, ``reason``);
the backend used to read all of them as "could not do the work" and force the
ticket into review. They are not the same thing:

* ``hold`` — a shell command the policy could not judge was HELD for the operator
  and either nobody answered in time or the operator said no. The session did not
  do something it wanted to do: the ticket needs a human look (review).
* ``read_outside`` — a read of a path outside the session directory was refused.
  The guardrail worked; the work was not blocked by it.
* ``refused`` — the gate refused a command outright, with no question for anyone:
  a never-allowed command (``git push``), a path traversal, an unreadable
  ``sed``/``awk`` program, an Automatos tool this ticket may not call. A refusal
  IS the answer — nobody was asked and nobody failed to answer — so it belongs
  with the other guardrails, not with the holds.
* ``unknown_tool`` — a tool a session never has (MCP, Task, …) was refused.
* ``prompt`` — a permission prompt reached the TUI and was answered "no" by the
  host (sessions are policy-gated, not prompted).
* ``other`` — a refusal this module cannot place. Counts as a hold: fail closed.

The markers are the host's own wording (``services/cli-host/automatos_cli_host/
session.py::_ask_operator`` and ``policy.py::decide``); a parity test reads those
files so a reworded host cannot silently turn every hold into ``other``.

Pure module: no DB, no config.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

DENIAL_KIND_HOLD = "hold"
DENIAL_KIND_READ_OUTSIDE = "read_outside"
DENIAL_KIND_REFUSED = "refused"
DENIAL_KIND_UNKNOWN_TOOL = "unknown_tool"
DENIAL_KIND_PROMPT = "prompt"
DENIAL_KIND_OTHER = "other"

# The kinds that put a ticket into review (``other`` fails closed).
FORCE_REVIEW_KINDS: Tuple[str, ...] = (DENIAL_KIND_HOLD, DENIAL_KIND_OTHER)

# ``session.py::_ask_operator`` — the two ways a held command ends without "allow".
HOLD_REASON_MARKERS: Tuple[str, ...] = ("no answer from the operator", "denied by the operator")
# ``policy.py::decide`` — a Read/Write/Bash path outside the session roots.
READ_OUTSIDE_MARKER = "outside the session directory"
# ``policy.py::decide`` — a tool class the session never has.
UNKNOWN_TOOL_MARKER = "is not enabled for session tickets"
# ``policy.py`` — the outright refusals. Every one of these is a decision the
# gate made by itself; no operator was asked, so none of them is a hold.
#
# Getting this wrong is not cosmetic. Before these markers existed, every
# refusal the list below covers fell through to ``other``, which fails closed —
# so a session that merely WROTE ``..`` inside a quoted string, or reached for
# an Automatos tool under the wrong name, sent its ticket to review with nothing
# for a human to act on. Sessions start one directory above the repos, so ``..``
# is exactly what an agent types.
REFUSED_REASON_MARKERS: Tuple[str, ...] = (
    "never allowed in a session",
    "path traversal",
    "is not one this ticket may call",
    "reads its program from a file the gate cannot judge",
    "program runs a command of its own",
    "redirection outside the session directory",
)
# ``session.py::_reply_for`` — the stage of a TUI permission prompt the host denied.
PROMPT_STAGE = "PermissionRequest"

# Report headings per kind, in the order the report lists them (holds first —
# they are why the ticket is in review).
DENIAL_KIND_LABELS: Tuple[Tuple[str, str], ...] = (
    (DENIAL_KIND_HOLD, "Held for the operator — no answer in time, or denied"),
    (DENIAL_KIND_OTHER, "Refused (unclassified — treated as a hold)"),
    (DENIAL_KIND_READ_OUTSIDE, "Reads outside the session directory"),
    (DENIAL_KIND_REFUSED, "Refused by the gate (nobody was asked)"),
    (DENIAL_KIND_UNKNOWN_TOOL, "Tools a session does not have"),
    (DENIAL_KIND_PROMPT, "Permission prompts (sessions are policy-gated, not prompted)"),
)


def classify_denial(stage: Any, reason: Any) -> str:
    """The kind of one refusal from the host's stage and wording. Unplaceable → ``other``."""
    text = str(reason or "")
    if any(marker in text for marker in HOLD_REASON_MARKERS):
        return DENIAL_KIND_HOLD
    if str(stage or "") == PROMPT_STAGE:
        return DENIAL_KIND_PROMPT
    if READ_OUTSIDE_MARKER in text:
        return DENIAL_KIND_READ_OUTSIDE
    if UNKNOWN_TOOL_MARKER in text:
        return DENIAL_KIND_UNKNOWN_TOOL
    if any(marker in text for marker in REFUSED_REASON_MARKERS):
        return DENIAL_KIND_REFUSED
    return DENIAL_KIND_OTHER


def forces_review(summaries: Any) -> bool:
    """True when any summarised denial is a hold (or unplaceable). A summary
    without a ``kind`` (an older ticket) counts as a hold — fail closed."""
    for item in summaries or []:
        kind = item.get("kind", DENIAL_KIND_OTHER) if isinstance(item, Mapping) else DENIAL_KIND_OTHER
        if kind in FORCE_REVIEW_KINDS:
            return True
    return False


def group_denials_by_kind(summaries: Any) -> Dict[str, list]:
    """Summaries → ``{kind: [summary, …]}`` in ``DENIAL_KIND_LABELS`` order,
    only the kinds that occur. A summary without a ``kind`` lands under ``other``."""
    buckets: Dict[str, list] = {}
    for item in summaries or []:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind") or DENIAL_KIND_OTHER)
        buckets[kind] = buckets.get(kind, []) + [dict(item)]
    ordered = {kind: buckets[kind] for kind, _ in DENIAL_KIND_LABELS if kind in buckets}
    leftovers = {kind: rows for kind, rows in buckets.items() if kind not in ordered}
    return {**ordered, **leftovers}


def denial_kind_label(kind: str) -> str:
    return dict(DENIAL_KIND_LABELS).get(kind, f"Refused ({kind})")
