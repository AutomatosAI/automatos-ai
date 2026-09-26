"""F198 — a redo carries every correction on the ticket, and the draft it corrects.

Night 6 (#1120, rounds 1-4): "each Reject only carries my latest note: round 1
had the voice and no café, round 2 the café and no voice, round 3 the voice and
no café again"; and a later redo wiped a correct total (#1152). A Reject
overwrote the ticket's one review_feedback field, the claim consumed it, and the
rejected draft was wiped, so every redo started over from the brief with only
the newest note.

A Reject now records its note in ``planning_data.owner_corrections`` and the
draft it sends back in ``previous_runs`` (api.board_tasks.reject_task). Both
claim paths (the dispatcher's, the CLI host's ``_ticket_prompt``) fold in
``redo_block``: that draft and every correction, oldest first, and the ask to
correct the draft rather than redo it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# keep_previous_run's reason for a Reject: the run whose draft a redo corrects.
SENT_BACK = "sent back"
# The review_feedback a Reject without a note leaves, so the redo still knows it
# is a redo (and still carries the earlier corrections and the draft).
SENT_BACK_WITHOUT_A_NOTE = "The owner sent it back without a note."
# A ticket sent back more often than this keeps its newest corrections.
MAX_CORRECTIONS_KEPT = 20


def with_correction(planning_data: Any, note: str, *, by: str, at: str) -> Dict[str, Any]:
    """``planning_data`` with ``note`` added to the ticket's corrections (rebuilt,
    never mutated in place)."""
    data = dict(planning_data) if isinstance(planning_data, dict) else {}
    corrections = list(data.get("owner_corrections") or [])
    corrections.append({"note": note, "by": by, "at": at})
    data["owner_corrections"] = corrections[-MAX_CORRECTIONS_KEPT:]
    return data


def redo_block(task: Any) -> Optional[str]:
    """What a redo is told: the draft that was sent back and every correction on
    the ticket, oldest first. None when this run is not a redo (no review_feedback
    waiting to be consumed)."""
    latest = getattr(task, "review_feedback", None)
    if not latest:
        return None
    data = task.planning_data if isinstance(getattr(task, "planning_data", None), dict) else {}
    notes = _corrections(data)
    if latest != SENT_BACK_WITHOUT_A_NOTE and (not notes or notes[-1] != latest):
        notes.append(latest)  # a note set another way (the PATCH, a stop) applies to this run too
    draft = _sent_back_draft(data)
    lines = ["## Redo: your last attempt was sent back"]
    if draft:
        lines += ["Your last attempt:", draft, ""]
    if notes:
        lines.append("Every correction on this ticket, oldest first. All of them still apply:")
        lines += [f"{n}. {note}" for n, note in enumerate(notes, 1)]
    if latest == SENT_BACK_WITHOUT_A_NOTE:
        lines.append("This time it came back without a new note." if notes else SENT_BACK_WITHOUT_A_NOTE)
    lines.append("Start from your last attempt: apply every correction and keep everything else as it was."
                 if draft else "Apply every correction.")
    return "\n".join(lines)


def _corrections(data: Dict[str, Any]) -> List[str]:
    return [c["note"] for c in data.get("owner_corrections") or [] if isinstance(c, dict) and c.get("note")]


def _sent_back_draft(data: Dict[str, Any]) -> Optional[str]:
    for run in reversed(data.get("previous_runs") or []):
        if isinstance(run, dict) and run.get("why") == SENT_BACK:
            return run.get("result") or None
    return None
