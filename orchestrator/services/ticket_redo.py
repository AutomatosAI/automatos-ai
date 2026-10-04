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

PRD-252 R2: the redo leads with the owner's words. The note that sent this
attempt back opens the block, word for word, before the draft it corrects; it was
the last line of a list under the draft.

F249 (night 7): a Reject's lesson stayed on the ticket it was written on. "Just the
email" went to the Support Agent three times (#0155, #0158, #0171), and "leave off
Perfect!" three times (#0141, #0152, #0161). ``redo_block`` now also carries the
owner's recent corrections to the ticket's agent on its other tickets, newest
first and each once, into every run of that agent's tickets. Both claim paths
read it. The ticket's own corrections stay in its redo part.

F249 (night 7b), partly fixed: the Analyst carried its note, but the Content Creator
slipped on #0199 and the newsletter helper on #0200. Two of the owner's lessons never
reached the agent: the ones written in an Approve note ("Next time put the working on
the card", "next time count"), and every lesson for a mission step or a playbook step,
whose prompts the coordinator and the playbook runner build without this block. An
Approve note that says what to do next time is a lesson now (``agent_lessons``), and
``lessons_block`` gives the same block to the mission's and the playbook's steps
(services/step_lessons.py).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

# keep_previous_run's reason for a Reject: the run whose draft a redo corrects.
SENT_BACK = "sent back"
# keep_previous_run's reason for Run now (api.board_tasks._start_now).
RUN_NOW = "run now"
# The review_feedback a Reject without a note leaves, so the redo still knows it
# is a redo (and still carries the earlier corrections and the draft).
SENT_BACK_WITHOUT_A_NOTE = "The owner sent it back without a note."
# A ticket sent back more often than this keeps its newest corrections.
MAX_CORRECTIONS_KEPT = 20
# What opens a redo whose owner said what is wrong (PRD-252 R2).
OWNER_WORDS_LEAD = "The owner sent it back with these words:"
# PRD-252 R2 (Discuss): a discussion ends with a brief the owner agreed, which
# "Update ticket and re-queue" writes onto the ticket (api.board_tasks.rebrief_task).
REBRIEFED = "re-briefed in a discussion"
BRIEF_AGREED = ("The owner agreed a new brief in a discussion; it is now this ticket's description. "
                "Work from it.")
# What a re-briefed ticket's next run is told. The drafts and notes before the
# agreed brief are what the discussion settled, so none of them is carried.
AGREED_BRIEF_BLOCK = ("## Redo: the owner agreed a new brief\n"
                      "The owner talked this ticket through and agreed the brief above. It replaces "
                      "the earlier brief, drafts and notes: work from it as written.")
MAX_BRIEF_CHARS = 8000
PREVIOUS_BRIEFS_KEPT = 5
# After this many Rejects the review panel suggests talking it through (Discuss).
DISCUSS_AFTER_REJECTS = 3
# F249: the agent's most recent distinct corrections on its other tickets, read from
# its last tickets that carry any; each is cut to a line's length.
STANDING_KEPT = 5
STANDING_TICKETS_READ = 30
STANDING_NOTE_CHARS = 300
STANDING_HEADING = "## The owner's corrections to your recent work"
STANDING_ASK = ("On your other work the owner sent drafts back with these notes, or said what to do next time; "
                "newest first. Follow each one here too, unless this brief says otherwise:")
# F249 (night 7b): an Approve note that says what to do next time is a lesson too.
NEXT_TIME = re.compile(r"\b(?:next time|from now on|in (?:the )?future|going forward|keep doing|always|never|"
                       r"don'?t|do not|no more|stop)\b", re.IGNORECASE)


def with_correction(planning_data: Any, note: str, *, by: str, at: str) -> Dict[str, Any]:
    """``planning_data`` with ``note`` added to the ticket's corrections (rebuilt,
    never mutated in place)."""
    data = dict(planning_data) if isinstance(planning_data, dict) else {}
    corrections = list(data.get("owner_corrections") or [])
    corrections.append({"note": note, "by": by, "at": at})
    data["owner_corrections"] = corrections[-MAX_CORRECTIONS_KEPT:]
    return data


def with_new_brief(planning_data: Any, old_description: Optional[str], *, by: str, at: str) -> Dict[str, Any]:
    """``planning_data`` with the brief being replaced kept in ``previous_briefs``
    (rebuilt, never mutated in place), so a re-brief loses nothing."""
    data = dict(planning_data) if isinstance(planning_data, dict) else {}
    briefs = list(data.get("previous_briefs") or [])
    briefs.append({"description": old_description or "", "by": by, "at": at})
    data["previous_briefs"] = briefs[-PREVIOUS_BRIEFS_KEPT:]
    return data


def times_sent_back(task: Any) -> int:
    """How many times the owner sent this ticket back with Reject since its brief
    was last agreed in Discuss. From DISCUSS_AFTER_REJECTS on, the review panel
    suggests Discuss (D3)."""
    data = getattr(task, "planning_data", None) or {}
    since = _brief_agreed_at(data)
    return sum(1 for run in data.get("previous_runs") or []
               if isinstance(run, dict) and run.get("why") == SENT_BACK and _after(run.get("at"), since))


def _parsed(at: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(at) if isinstance(at, str) else None
    except ValueError:
        return None


def _after(at: Any, since: Optional[datetime]) -> bool:
    if since is None:
        return True
    when = _parsed(at)
    return when is not None and when > since


def redo_block(task: Any) -> Optional[str]:
    """What a run is told of the owner's corrections. On a redo: the draft that was
    sent back and every correction on the ticket, oldest first. On any run: the
    owner's recent corrections to its agent on its other tickets (F249). None when
    there is neither."""
    parts = [_redo(task), standing_corrections(task)]
    return "\n\n".join(part for part in parts if part) or None


def _redo(task: Any) -> Optional[str]:
    """The redo part: None when this run is not a redo (no review_feedback waiting
    to be consumed)."""
    latest = getattr(task, "review_feedback", None)
    if not latest:
        return None
    if latest == BRIEF_AGREED:
        return AGREED_BRIEF_BLOCK
    data = task.planning_data if isinstance(getattr(task, "planning_data", None), dict) else {}
    since = _brief_agreed_at(data)   # PRD-252 R2: what came before an agreed brief is settled
    notes = _corrections(data, since)
    if latest != SENT_BACK_WITHOUT_A_NOTE and (not notes or notes[-1] != latest):
        notes.append(latest)  # a note set another way (the PATCH, a stop) applies to this run too
    draft = _sent_back_draft(data, since)
    lines = ["## Redo: your last attempt was sent back"]
    if latest != SENT_BACK_WITHOUT_A_NOTE:
        lines += [OWNER_WORDS_LEAD, latest, ""]
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


def kept_draft(task: Any) -> Optional[str]:
    """F243: the draft a failed redo leaves on the card's face. A redo starts clean
    (F190: the last run goes on record and off the card), so when it failed (#0171
    and #0174 ran out of credit) the draft the owner had been correcting was only
    in the card's history. While the failed run has no result of its own, this is
    the latest run's that had one."""
    if getattr(task, "status", None) != "failed" or getattr(task, "result", None):
        return None
    data = task.planning_data if isinstance(getattr(task, "planning_data", None), dict) else {}
    for run in reversed(data.get("previous_runs") or []):
        if isinstance(run, dict) and run.get("result"):
            return run["result"]
    return None


def redo_again(task: Any) -> Optional[str]:
    """F294 (night 8, #0273): Run now on a redo that failed before it answered runs
    that redo again, with the owner's words. The claim had consumed the Reject's note
    (``review_feedback``), the redo failed ("Empty response from LLM"), and Run now
    started from the brief alone: "0 kg" again, the reason gone. When the run that Run
    now replaces was a redo that left no answer (and any Run now since failed the same
    way), the newest correction goes back on the card, so the next claim folds in the
    redo block: the draft that was sent back and every correction. Returns it, or None."""
    if getattr(task, "status", None) != "failed" or getattr(task, "result", None) \
            or getattr(task, "review_feedback", None):
        return None
    data = task.planning_data if isinstance(getattr(task, "planning_data", None), dict) else {}
    if not _last_was_a_redo(data):
        return None
    notes = _corrections(data, _brief_agreed_at(data))
    task.review_feedback = notes[-1] if notes else SENT_BACK_WITHOUT_A_NOTE
    return task.review_feedback


def _last_was_a_redo(data: Dict[str, Any]) -> bool:
    """Whether the newest run on record was sent back, past any Run now that failed
    with no answer since."""
    for run in reversed(data.get("previous_runs") or []):
        if not isinstance(run, dict):
            continue
        if run.get("why") == SENT_BACK:
            return True
        if run.get("why") != RUN_NOW or run.get("status") != "failed" or run.get("result"):
            return False
    return False


def _brief_agreed_at(data: Dict[str, Any]) -> Optional[datetime]:
    """When the ticket's brief was last agreed in a discussion; None if never."""
    briefs = [b for b in data.get("previous_briefs") or [] if isinstance(b, dict)]
    return _parsed(briefs[-1].get("at")) if briefs else None


def _corrections(data: Dict[str, Any], since: Optional[datetime] = None) -> List[str]:
    return [c["note"] for c in data.get("owner_corrections") or []
            if isinstance(c, dict) and c.get("note") and _after(c.get("at"), since)]


def _sent_back_draft(data: Dict[str, Any], since: Optional[datetime] = None) -> Optional[str]:
    for run in reversed(data.get("previous_runs") or []):
        if isinstance(run, dict) and run.get("why") == SENT_BACK and _after(run.get("at"), since):
            return run.get("result") or None
    return None


def standing_corrections(task: Any) -> Optional[str]:
    """The owner's notes on this ticket's agent's other tickets, newest first and
    each once: what one Reject taught applies to the agent's next card."""
    return lessons_block(_session_of(task), getattr(task, "workspace_id", None),
                         getattr(task, "assigned_agent_id", None), but_not=getattr(task, "id", None))


def lessons_block(db: Any, workspace_id: Any, agent_id: Any, *, but_not: Any = None) -> Optional[str]:
    """The block a run of ``agent_id``'s work is given: its lessons, newest first; None without."""
    notes = agent_lessons(db, workspace_id, agent_id, but_not=but_not)
    if not notes:
        return None
    return "\n".join([STANDING_HEADING, STANDING_ASK, *(f"- {note}" for note in notes)])


def agent_lessons(db: Any, workspace_id: Any, agent_id: Any, *, but_not: Any = None) -> List[str]:
    """The owner's lessons for an agent from its recent cards (but ``but_not``), newest
    first and each once: the notes they sent its work back with, and the Approve notes
    that say what to do next time (F249)."""
    if not agent_id or db is None or workspace_id is None:
        return []
    from core.models.core import BoardTask

    query = db.query(BoardTask.planning_data, BoardTask.runtime_ref).filter(
        BoardTask.workspace_id == workspace_id, BoardTask.assigned_agent_id == agent_id)
    if but_not is not None:
        query = query.filter(BoardTask.id != but_not)
    rows = query.order_by(BoardTask.updated_at.desc(), BoardTask.id.desc()).limit(STANDING_TICKETS_READ).all()
    dated = sorted((pair for data, ref in rows for pair in _lessons(data, ref)), reverse=True)
    return _distinct([note for _, note in dated])[:STANDING_KEPT]


def _lessons(planning_data: Any, runtime_ref: Any) -> List[tuple]:
    """(when, note) for a card's corrections, and for its Approve notes that teach."""
    from services.ticket_verdict import APPROVAL_NOTE_PREFIX

    notes = runtime_ref.get("session_notes") if isinstance(runtime_ref, dict) else None
    approved = [(n.get("at") or "", n["note"][len(APPROVAL_NOTE_PREFIX):].strip())
                for n in notes or [] if isinstance(n, dict) and isinstance(n.get("note"), str)
                and n["note"].startswith(APPROVAL_NOTE_PREFIX) and NEXT_TIME.search(n["note"])]
    return [(c.get("at") or "", c["note"]) for c in _entries(planning_data)] + approved


def _entries(data: Any) -> List[Dict[str, Any]]:
    entries = (data or {}).get("owner_corrections") if isinstance(data, dict) else None
    return [c for c in entries or [] if isinstance(c, dict) and isinstance(c.get("note"), str)
            and c["note"].strip() and c["note"] != SENT_BACK_WITHOUT_A_NOTE]


def _distinct(notes: List[str]) -> List[str]:
    """Each note once (the same words, whatever their case and spacing), cut to a line."""
    seen, kept = set(), []
    for note in notes:
        key = re.sub(r"\s+", " ", note).strip().casefold()
        if key not in seen:
            seen.add(key)
            text = note.strip()
            kept.append(text if len(text) <= STANDING_NOTE_CHARS else text[:STANDING_NOTE_CHARS - 1] + "…")
    return kept


def _session_of(task: Any) -> Any:
    from sqlalchemy.orm import object_session
    from sqlalchemy.orm.exc import UnmappedInstanceError

    try:
        return object_session(task)
    except UnmappedInstanceError:    # a plain object standing in for a ticket
        return None
