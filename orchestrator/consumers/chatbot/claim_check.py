"""F187 (night 6): a reply that reports work no tool did, or names an id that
does not exist, is corrected where it is saved.

F108 nudges a claim inside the tool loop, and F099's notice under a first reply
is shown but never saved. On night 6, five first replies said work was done
("I've just assigned the Content Creator agent…", "I've just created a new task
on your board…") and ran no tool at all. They never entered the loop, the claim
was saved as the answer, and the next turn read it as fact. The reply at
02:49:35 also gave the owner "New Task ID: 1100", a ticket that did not exist.

- Tier 1: a reply that ran no tool and says an action was done (F108's families)
  enters the loop, which nudges it once. A retry that still claims is saved with
  a correction line.
- Tier 2: an id in one of three shapes that did not exist when the reply was
  written gets one re-prompt, then the correction line. The shapes are #1100,
  in a sentence about tasks or the board; "Task ID 1100", "agent ID 102" or
  "Playbook ID 101"; and a run token (exec-, cron- or rerun- plus 12 hex
  digits).
- Tier 3: a passive claim ("has been assigned") that no family reads is only
  logged, so the families can be tuned night by night.

F261 (night 8): a claim no successful action backs when tools did run was only
logged too. "I've cancelled Mission #0365" sat beside a playbook run, the owner
found #0365 still waiting, and nothing in the reply said so. F108's nudge acts
first; a reply that still claims after it is now corrected too, naming what it
claimed.

F264 (night 8): "#0226", "#0302", "#0386" are the board's numbers for its cards,
and the id check read them as ids. Task 226 was no card of this workspace, so a
right reply was re-prompted, the retry came back empty, and the owner got "I
apologize, but I encountered an issue" after the work was done. A '#' number
that is a card's number in the workspace exists.

F314 (night 9): the owner read "Correction: this reply says something was under way,
but no action in it did that, so it has not happened. Ask me to do it and check the
board after." under Auto's timer reply (chat b8d9121f), a line that read like the
system talking about Auto. Each line is now Auto's own, in plain words, and says what
did not happen for the kind of claim it was ("I didn't approve anything in this
reply", "nothing is still running from this reply, and I won't come back to this on
my own"). Since FX-007 the receipts say it (``claims_backed.not_done_line``).

PRD-256 US-002: the saved correction line keeps tier 2 only (an id that does not exist). What
was not done is said above the text from the turn's receipts (``consumers/chatbot/receipts.py``).

PRD-256 FX-007 (D10): tiers 1 and 3 are gone with the regex claim families they read
(action_claims, document_claims, shop_and_team_claims, social_post_claims): a claim is the
receipts' to answer (``claims_backed``), in the loop's nudge and above the answer. Tier 2 stays,
and a number the turn's own tool results quote is backed (night 12, A439/A593: "task 0930 does
not exist" under a reply that quoted ticket #0931's title, "Card #0930 Approval Request").

P256-FIX-RVW-8: a number the turn's automatic reads put in front of the model is quoted too. A
"what is waiting for me?" turn reads Needs you by prefetch, not by a tool call, so the note naming
"#0931 'Card #0930 Approval Request'" was in no tracker's outcomes and #0930 was still corrected, on
the first reply and in the loop. ``receipts.its_reads_are_receipted`` hands the notes the reads added
(the needs-you note, the team's findings, the retrieval-first passages) to ``reads_put_in_front``.
"""
from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from modules.tools.execution.tool_execution_tracker import TRACKERS_MADE

logger = logging.getLogger(__name__)

# F314: the owner's line under a claim no action backed, in Auto's own words (claims_backed names it).
NOT_DONE = "Just to be clear: {said}. Ask me again if you want it done."
NOTHING_DONE = NOT_DONE.format(said="I haven't done that yet, and nothing has changed")
NO_SUCH_ID = "Just to be clear: {ids} {verb} not exist — I named {it} without looking {it} up."
ID_NUDGE = (
    "Your previous reply names {ids}, which {verb} not exist in this workspace. Look it up now with a tool "
    "(platform_list_tasks, platform_list_agents, platform_list_playbooks) before naming it, or leave the number "
    "out and say plainly what was and was not done."
)

# A bare "#1234" is a task only in a sentence about the board; "Order #1042" is not.
_HASH_ID = re.compile(r"(?<![\w£$€&#/])#(\d{3,7})\b")
_BOARD_TALK = re.compile(r"\b(?:tasks?|tickets?|cards?|board)\b", re.I)
_OTHER_NUMBERING = re.compile(r"\b(?:orders?|invoices?|receipts?|issues?|pull requests?|PRs?)\b", re.I)
_ID_SHAPES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("task", re.compile(r"\b(?:task|ticket)\s+id\b\s*[:#]?\s*#?(\d{1,7})\b", re.I)),
    ("agent", re.compile(r"\bagent\s+id\b\s*[:#]?\s*#?(\d{1,7})\b", re.I)),
    ("playbook", re.compile(r"\bplaybook\s+id\b\s*[:#]?\s*#?(\d{1,7})\b", re.I)),
    ("run", re.compile(r"\b((?:exec|cron|rerun)-[0-9a-f]{12})\b")),
)
_KIND_WORD = {"task": "task", "agent": "agent", "playbook": "playbook", "run": "run"}
# A sentence that says an id is missing is a report, not an invention (02:34:31:
# "The system is telling me that agent ID 102 does not exist").
_DENIAL = re.compile(
    r"\b(?:does not|doesn't|doesn’t|do not|don't|don’t|no longer|did not|didn't|didn’t)\s+exists?\b|"
    r"\bnot\s+(?:be\s+)?found\b|\bcould(?:n't|n’t| not) (?:find|locate)\b|\bno such\b", re.I)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
_MARKDOWN = re.compile(r"[*_`]")
# RVW-8: the text this turn's automatic reads put in front of the model ('' until they ran; each turn resets it).
READS_SAID: ContextVar[str] = ContextVar("f187_reads_said", default="")

def _named_ids(text: str) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    for sentence in _SENTENCE.findall(_MARKDOWN.sub("", text or "")):
        if _DENIAL.search(sentence):
            continue
        shapes = _ID_SHAPES
        if _BOARD_TALK.search(sentence) and not _OTHER_NUMBERING.search(sentence):
            shapes = (("task", _HASH_ID),) + shapes
        for kind, shape in shapes:
            for value in shape.findall(sentence):
                if (kind, value) not in found:
                    found.append((kind, value))
    return found


def existing_ids(workspace_id: str, named: List[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """Which of ``named`` exist in the workspace now. Sync: run it off the loop."""
    from core.database.database import SessionLocal
    from core.models import Agent, BoardTask, RecipeExecution, WorkflowTemplate

    from sqlalchemy import or_

    ws = str(workspace_id)
    wanted: Dict[str, List[str]] = {}
    for kind, value in named:
        wanted.setdefault(kind, []).append(value)
    found: Set[Tuple[str, str]] = set()
    db = SessionLocal()
    try:
        # A platform agent or a marketplace playbook has no workspace and still exists.
        numbered = {"task": (BoardTask, False), "agent": (Agent, True), "playbook": (WorkflowTemplate, True)}
        found.update(_card_numbers(db, ws, wanted.get("task") or []))
        for kind, (model, shared) in numbered.items():
            if wanted.get(kind):
                owned = or_(model.workspace_id == ws, model.workspace_id.is_(None)) if shared \
                    else model.workspace_id == ws
                rows = db.query(model.id).filter(owned, model.id.in_([int(v) for v in wanted[kind]])).all()
                found.update((kind, str(row[0])) for row in rows)
        if wanted.get("run"):
            rows = db.query(RecipeExecution.execution_id).filter(
                RecipeExecution.workspace_id == ws, RecipeExecution.execution_id.in_(wanted["run"])).all()
            found.update(("run", row[0]) for row in rows)
    finally:
        db.close()
    return found


def _card_numbers(db, workspace_id: str, values: List[str]) -> Set[Tuple[str, str]]:
    """The values that are a card's number on this workspace's board (#0226: workspace_seq 226)."""
    from core.models import BoardTask

    seqs = {int(v): v for v in values}
    if not seqs:
        return set()
    rows = db.query(BoardTask.workspace_seq).filter(BoardTask.workspace_id == workspace_id,
                                                    BoardTask.workspace_seq.in_(list(seqs))).all()
    return {("task", seqs[row[0]]) for row in rows}


def _strings(value: object) -> Iterator[str]:
    """The text a tool result says (its string values, at any depth): a count, a page or a
    limit is a number the reply never quotes as an id."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for inner in value.values():
            yield from _strings(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _strings(inner)


def reads_put_in_front(notes: Iterable[Dict[str, Any]]) -> None:
    """Keep, for this turn, what the messages its automatic reads added say (RVW-8)."""
    READS_SAID.set("\n".join(said for note in notes for said in _strings(note.get("content"))))


def _quoted_this_turn() -> str:
    """What the turn's automatic reads and the tool results of the loop running now say, as
    text: a ticket's title, a card or grant number. Outside a tool loop, the reads alone."""
    made = TRACKERS_MADE.get() or []
    results = (said for tracker in made for _action, _params, result in tracker.outcomes for said in _strings(result))
    return "\n".join([READS_SAID.get(), *results])


def _named_in(value: str, text: str) -> bool:
    """Whether ``text`` names ``value`` as a whole number or token ("0930" in "Card #0930", "930")."""
    shape = rf"0*{int(value)}" if value.isdigit() else re.escape(value)
    return re.search(rf"(?<![\w-]){shape}(?![\w-])", text) is not None


def invented_ids(text: str, owner_text: str, workspace_id: str) -> List[Tuple[str, str]]:
    """The ids a reply names, in the three shapes, that do not exist in the
    workspace, leaving out any the owner named this turn and any the turn's own tool
    results (FX-007) or automatic reads (RVW-8) quote. Sync, and reads the database only
    when the reply names an id: run it off the loop (``asyncio.to_thread`` keeps the turn's
    context)."""
    owner = _MARKDOWN.sub("", owner_text or "")
    quoted = _quoted_this_turn()
    named = [(kind, value) for kind, value in _named_ids(text)
             if not _named_in(value, owner) and not _named_in(value, quoted)]
    if not named or not workspace_id:
        return []
    try:
        present = existing_ids(workspace_id, named)
    except Exception:  # noqa: BLE001 — a failed lookup never corrects a reply
        logger.warning("[F187] id lookup failed; the reply's ids are not checked", exc_info=True)
        return []
    return [pair for pair in named if pair not in present]


def _listed(ids: List[Tuple[str, str]]) -> str:
    return ", ".join(f"{_KIND_WORD[kind]} {value}" for kind, value in ids)


def id_nudge(ids: List[Tuple[str, str]]) -> str:
    return ID_NUDGE.format(ids=_listed(ids), verb="does" if len(ids) == 1 else "do")


@dataclass
class Verdict:
    """What F187 found in a turn's answer: the ids it names that do not exist (tier 2).
    ``tools`` counts the model's own tool calls (not the automatic retrieval-first search)."""

    tools: int
    ids: List[Tuple[str, str]] = field(default_factory=list)
    reprompted: bool = False

    @property
    def correction(self) -> Optional[str]:
        """Tier 2's line: the ids the answer names that do not exist (PRD-256 US-002: a claim
        no action backs is the receipts' to answer, above the text)."""
        if not self.ids:
            return None
        one = len(self.ids) == 1
        return NO_SUCH_ID.format(ids=_listed(self.ids), verb="does" if one else "do", it="it" if one else "them")

    def log(self, reply_id: object) -> None:
        """One [F187] line per id corrected, for night-by-night tuning."""
        for kind, value in self.ids:
            logger.warning(f"[F187] tier=2 id={kind}:{value} tools={self.tools} reply={reply_id} "
                           f"reprompted={self.reprompted} action=corrected")
