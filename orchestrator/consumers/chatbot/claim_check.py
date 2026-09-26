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
- Tier 3: a claim no successful action backs when tools did run (F108's nudge
  already acts there), and a passive claim ("has been assigned"), is only
  logged, so the families can be tuned night by night.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

NOTHING_DONE = "Correction: nothing was done yet — no action ran. Ask me to do it and check the board after."
NO_SUCH_ID = "Correction: {ids} {verb} not exist — I named {it} without looking {it} up."
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
# Tier 3, log only: a completion told in the passive voice.
_PASSIVE = re.compile(
    r"\b(?:has|have) been (?:created|scheduled|sent|updated|assigned|approved|added|set up|saved|filed|queued|"
    r"started|launched|drafted|posted)\b|"
    r"\b(?:is|are) now (?:on (?:your|the) board|scheduled|running|live|set up|assigned|active)\b", re.I)
# Intent or plan, not a report: "I'm going to re-assign…", "Once the agent drafts it…".
_INTENT = re.compile(r"\b(?:i'?m going to|i am going to|i will|i'll|i’ll|will be|once|when|after)\b", re.I)


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


def invented_ids(text: str, owner_text: str, workspace_id: str) -> List[Tuple[str, str]]:
    """The ids a reply names, in the three shapes, that do not exist in the
    workspace, leaving out any the owner named this turn. Sync, and reads the
    database only when the reply names an id: run it off the loop."""
    owner = _MARKDOWN.sub("", owner_text or "")
    named = [(kind, value) for kind, value in _named_ids(text)
             if not re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", owner)]
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


def passive_claim(text: str) -> bool:
    return any(_PASSIVE.search(s) and not _INTENT.search(s) for s in _SENTENCE.findall(text or ""))


@dataclass
class Verdict:
    """What F187 found in a turn's answer. ``tools`` counts the model's own
    tool calls (not the automatic retrieval-first search)."""

    tools: int
    claim: Optional[str] = None
    passive: bool = False
    ids: List[Tuple[str, str]] = field(default_factory=list)
    reprompted: bool = False

    @property
    def correction(self) -> Optional[str]:
        lines = []
        if self.claim and self.tools == 0:
            lines.append(NOTHING_DONE)
        if self.ids:
            one = len(self.ids) == 1
            lines.append(NO_SUCH_ID.format(ids=_listed(self.ids), verb="does" if one else "do",
                                           it="it" if one else "them"))
        return "\n\n".join(lines) or None

    def log(self, reply_id: object) -> None:
        """One [F187] line per finding, for night-by-night tuning."""
        if self.claim:
            tier, action = (1, "corrected") if self.tools == 0 else (3, "logged")
            logger.warning(f"[F187] tier={tier} family={self.claim} tools={self.tools} reply={reply_id} "
                           f"action={action}")
        if self.passive and not self.claim:
            logger.warning(f"[F187] tier=3 family=passive tools={self.tools} reply={reply_id} action=logged")
        for kind, value in self.ids:
            logger.warning(f"[F187] tier=2 id={kind}:{value} tools={self.tools} reply={reply_id} "
                           f"reprompted={self.reprompted} action=corrected")
