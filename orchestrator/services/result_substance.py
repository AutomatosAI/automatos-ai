"""F093 (night 3): a ticket whose result is only skipped tool calls did nothing.

Night 3's #484 was filed done/ok on a whole result of "search_knowledge:
Skipped: Tool 'search_knowledge' has reached its execution limit (5) for this
turn" — 29 model calls, 369,414 tokens. Night 1's #639 report was nothing but
"workspace_git: Skipped: … execution limit (8)" over and over. When a run ends
without an answer, the agent path builds its result from the last round's tool
messages; when every one of those was skipped (a per-tool limit, a repeat, a
policy block), the run produced nothing — and the ticket goes to review saying
so, never done.

F297 (night 8): a run that wrote no answer after its tool calls ended on "Based on
the tool results:" and its last round's raw tool output, and that was the card's
answer: #0234's ``{"written": true, "path": …}``, #0379's brand guide as JSON,
#0391's memory note twice, #0408.3's mission field note (a mission step, marked
done), and #0254's whole answer a raw Composio error ("'GMAIL' is assigned to agent
323 but is not connected for this workspace…"). Such a result is now said plainly
(``plain_no_answer``): no answer was written, what each call did, and an app that is
not connected by its name and what to do. A board card goes to review with it, a
mission step is a failed attempt (``as_step_failure``), never done.

F306 (night 9): F297 caught tool output, not a run that stopped mid-thought: "Let me
try a more specific query:" was #1879's answer twice, "Now let me get the total
kilograms per account …:" #1881's, "I'll attempt a query to list all tables in"
#1888's. A result whose last line announces a step it never took
(``modules.tools.execution.nudges.announced_step``) is said plainly the same way
(``stopped_mid_step``): it goes to review, or is a failed attempt for a mission step.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.services.ticket_reasons import NOTHING_DONE_NOTE_PREFIX

# The agent path's header when a run ends without an answer (agent_factory).
TOOL_RESULTS_HEADER = "Based on the tool results:"
NOTHING_DONE_HEADER = "No answer — the run ended with every tool call in its last round skipped:"

# PRD-252 R3: the board reads the prefix as the ticket's reason for review.
NOTHING_DONE_NOTE = (NOTHING_DONE_NOTE_PREFIX + " every tool call in the run's last round was skipped "
                     "({reasons}). Sent to review instead of done.")

NO_ANSWER_HEADER = "No answer: the agent's run ended after its tool calls without writing one."
NO_ANSWER_NOTE = (NOTHING_DONE_NOTE_PREFIX + " the agent wrote no answer, only what its tools did. "
                  "Sent to review instead of done.")
STOPPED_HEADER = "No answer: the agent's run stopped at a step it announced and never took."
STOPPED_NOTE = (NOTHING_DONE_NOTE_PREFIX + " the agent stopped at a step it announced, with no answer. "
                "Sent to review instead of done.")
WHAT_IT_WROTE = "What it wrote before it stopped:"
_DUMP_LINE = re.compile(r"^\*\*(?P<tool>[\w.\-]+)\*\*\s*:\s*(?P<body>.*)$")
_NOT_CONNECTED = re.compile(r"'(?P<app>[A-Za-z0-9_]+)' is assigned to agent \d+ but is not connected")
_NAMED_FILE = re.compile(r'"(?:path|file_path|filename)"\s*:\s*"(?P<name>[^"]{1,200})"')
_FAILED = re.compile(r'"success"\s*:\s*false|^Error executing|^\{"error"\s*:', re.IGNORECASE)

_SKIP_LINE = re.compile(
    r"^(?:\*\*)?(?P<tool>[\w.\-]+)(?:\*\*)?\s*:\s*(?:Skipped:|Blocked by policy:?)\s*(?P<reason>.*)$")


def is_skip_message(content: str) -> bool:
    """A tool message the loop wrote instead of running the tool."""
    text = (content or "").lstrip()
    return text.startswith("Skipped:") or text.startswith("Blocked by policy")


def nothing_done_note(result: str) -> Optional[str]:
    """The review note for a result that is only skipped tool calls (F093), or one
    that says the run wrote no answer (F297), else None."""
    lines: List[str] = [ln.strip() for ln in (result or "").splitlines() if ln.strip()]
    if lines and lines[0] == NO_ANSWER_HEADER:
        return NO_ANSWER_NOTE
    if lines and lines[0] == STOPPED_HEADER:
        return STOPPED_NOTE
    if lines and lines[0] in (TOOL_RESULTS_HEADER, NOTHING_DONE_HEADER):
        lines = lines[1:]
    if not lines:
        return None
    matches = [_SKIP_LINE.match(ln) for ln in lines]
    if not all(matches):
        return None
    reasons = sorted({f"{m['tool']}: {m['reason'][:120]}" for m in matches})
    return NOTHING_DONE_NOTE.format(reasons="; ".join(reasons))


def _plain_outcome(tool: str, body: str) -> str:
    """One raw tool line of a run that wrote no answer, in plain words."""
    app = _NOT_CONNECTED.search(body)
    if app:
        name = app["app"].replace("_", " ").title()
        return (f"{name} is not connected in this workspace, so the agent could not use it: "
                "connect it in Composio, or ask for the work without it.")
    if is_skip_message(body):
        return f"{tool}: skipped."
    if _FAILED.search(body):
        return f"{tool}: failed."
    named = _NAMED_FILE.search(body)
    return f"{tool}: done ({named['name']})." if named else f"{tool}: done."


def plain_no_answer(result: str) -> Optional[str]:
    """F297: the agent path's result for a run that ended on its tool calls ("Based
    on the tool results:" over the raw output of its last round) said plainly: that
    no answer was written, then one line per distinct outcome. None for any other
    result."""
    lines: List[str] = [ln.strip() for ln in (result or "").splitlines() if ln.strip()]
    if not lines or lines[0] != TOOL_RESULTS_HEADER:
        return None
    found = [_DUMP_LINE.match(ln) for ln in lines[1:]]
    outcomes = dict.fromkeys(_plain_outcome(m["tool"], m["body"]) for m in found if m)
    return "\n".join([NO_ANSWER_HEADER, *(f"- {line}" for line in outcomes)])


def stopped_mid_step(result: str) -> Optional[str]:
    """F306: a result whose last line announces a step the run never took, said
    plainly, with what the agent wrote kept below; None for any other result."""
    from modules.tools.execution.nudges import announced_step

    text = (result or "").strip()
    if not text or text.startswith((NO_ANSWER_HEADER, STOPPED_HEADER)) or not announced_step(text):
        return None
    return f"{STOPPED_HEADER}\n\n{WHAT_IT_WROTE}\n{text}"


def as_step_failure(result: Dict[str, Any]) -> Dict[str, Any]:
    """F297: a mission step whose run wrote no answer is a failed attempt, retried
    while it has attempts and failed after, never a completed step (#0408.3 was
    marked done on a line of tool output). Any other result is returned as it is."""
    text = str((result or {}).get("result") or "")
    if (result or {}).get("status") != "success" or not text.startswith((NO_ANSWER_HEADER, STOPPED_HEADER)):
        return result
    return {**result, "status": "error", "error": text}

