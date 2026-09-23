"""F093 (night 3): a ticket whose result is only skipped tool calls did nothing.

Night 3's #484 was filed done/ok on a whole result of "search_knowledge:
Skipped: Tool 'search_knowledge' has reached its execution limit (5) for this
turn" — 29 model calls, 369,414 tokens. Night 1's #639 report was nothing but
"workspace_git: Skipped: … execution limit (8)" over and over. When a run ends
without an answer, the agent path builds its result from the last round's tool
messages; when every one of those was skipped (a per-tool limit, a repeat, a
policy block), the run produced nothing — and the ticket goes to review saying
so, never done.
"""
from __future__ import annotations

import re
from typing import List, Optional

# The agent path's header when a run ends without an answer (agent_factory).
TOOL_RESULTS_HEADER = "Based on the tool results:"
NOTHING_DONE_HEADER = "No answer — the run ended with every tool call in its last round skipped:"

NOTHING_DONE_NOTE = ("Nothing was produced: every tool call in the run's last round was skipped "
                     "({reasons}). Sent to review instead of done.")

_SKIP_LINE = re.compile(
    r"^(?:\*\*)?(?P<tool>[\w.\-]+)(?:\*\*)?\s*:\s*(?:Skipped:|Blocked by policy:?)\s*(?P<reason>.*)$")


def is_skip_message(content: str) -> bool:
    """A tool message the loop wrote instead of running the tool."""
    text = (content or "").lstrip()
    return text.startswith("Skipped:") or text.startswith("Blocked by policy")


def nothing_done_note(result: str) -> Optional[str]:
    """The review note for a result that is only skipped tool calls, else None."""
    lines: List[str] = [ln.strip() for ln in (result or "").splitlines() if ln.strip()]
    if lines and lines[0] in (TOOL_RESULTS_HEADER, NOTHING_DONE_HEADER):
        lines = lines[1:]
    if not lines:
        return None
    matches = [_SKIP_LINE.match(ln) for ln in lines]
    if not all(matches):
        return None
    reasons = sorted({f"{m['tool']}: {m['reason'][:120]}" for m in matches})
    return NOTHING_DONE_NOTE.format(reasons="; ".join(reasons))
