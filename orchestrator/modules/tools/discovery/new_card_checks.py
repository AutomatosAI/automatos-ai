"""What a new card's title and brief may not say (F241 and F265, night 7b).

- F241: asked "Please give #0192 to the Shopify Support Agent", Auto made a NEW card,
  #0194 "Handle task #0192" ("OBJECTIVE: Process task #0192"). #0192 stayed in the
  Inbox, and #0194 skipped the owner's review. A new card whose title is another card's
  number with only a verb like "handle" around it is a copy. It is refused, naming the
  card and the calls that act on it (platform_assign_task gives it to an agent).
- F265: Auto's brief for #0182 told the Analyst "Use `platform_update_task_status` to
  mark the task as 'review' when complete". The agent's own move was refused, and its
  answer said a table had been made where there was none. The board moves a card when
  its agent answers, so a sentence of a brief telling the agent to change the card's
  status is taken out before the card is filed, and the result says so.
- F289 (night 8): Auto made #0251 and #0386 straight into Review (platform_create_task
  {status: "review"}): nobody had worked on them, and Needs you counted them. A new card
  starts in the Inbox or with its agent, and reaches Review when its agent answers.
"""
from __future__ import annotations

import functools
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

# "#0192", "# 0192", "#0188.3": a card's number in a title.
_NUMBER = re.compile(r"#\s?(\d{1,7}(?:\.\d+)?)")
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)?")
# The words a copy wraps another card's number in: "Handle task #0192", "Redo card #0177".
COPY_WORDS = frozenset({
    "handle", "handling", "process", "do", "complete", "finish", "work", "on", "take", "over", "pick", "up",
    "redo", "re-do", "rerun", "re-run", "run", "again", "continue", "action", "deal", "with", "task", "card",
    "ticket", "the", "please", "for", "item", "number", "no", "of", "a", "this", "that",
})
A_COPY = ("{label} ('{title}') is already on the board, so no new card was made. To give it to an agent, call "
          "platform_assign_task with task_id \"{number}\" and the agent's name. To run it again, call "
          "platform_update_task_status with task_id \"{number}\" and status \"in_progress\".")

# A sentence that tells the agent to move its card: the status tool by name, or
# "mark/set/move the task (status) as/to review".
_STATUS_TOOL = re.compile(r"\bplatform_update_task(?:_status)?\b|\bupdate_task_status\b", re.IGNORECASE)
_STATUS_ORDER = re.compile(
    r"\b(?:mark|set|move|change|update|put|switch)\b[^.\n]{0,40}?\b(?:task|card|ticket|status)\b[^.\n]{0,30}?"
    r"\b(?:as|to|in|into)\b\s*['\"`]?(?:review|done|complete|completed|in[ _-]?progress|blocked|finished)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
STATUS_TAKEN_OUT = ("A sentence of the brief told the agent to change its card's status, and was taken out: the "
                    "board moves the card when the agent answers. Briefs say what to do and what to hand back.")
# A new card nobody has worked on is never filed in Review (F289).
NOT_YET_WORKED = ("review",)
STARTS_IN_THE_INBOX = ("A new card starts in the Inbox or with its agent, not in Review: it reaches Review when its "
                       "agent answers. It was filed without the status.")


def checks_the_new_card(handler: Handler) -> Handler:
    """Refuse a new card that copies an existing one; take status orders out of its brief."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        refusal = copy_refusal(db, workspace_id, (params or {}).get("title"))
        if refusal:
            return {"success": False, "error": refusal}
        params, held = _without_a_finished_status(params or {})
        brief, taken_out = without_status_orders(params.get("description"))
        out = await handler(db, workspace_id, {**params, "description": brief} if taken_out else params)
        if not (isinstance(out, dict) and out.get("success")):
            return out
        notes = {"brief_note": STATUS_TAKEN_OUT} if taken_out else {}
        return {**out, **notes, **({"status_note": STARTS_IN_THE_INBOX} if held else {})}
    return wrapped


def _without_a_finished_status(params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """The call without a status saying it was already worked on (F289), and whether one was taken out."""
    if str(params.get("status") or "").strip().lower() not in NOT_YET_WORKED:
        return params, False
    return {k: v for k, v in params.items() if k != "status"}, True


def copy_refusal(db: Session, workspace_id: Any, title: Any) -> Optional[str]:
    """Why a card titled ``title`` would copy one already on the board, or None."""
    text = str(title or "")
    numbers = _NUMBER.findall(text)
    if len(numbers) != 1:
        return None
    rest = _WORD.findall(_NUMBER.sub(" ", text).lower())
    if any(word not in COPY_WORDS for word in rest):
        return None
    return _refusal_for(db, workspace_id, f"#{numbers[0]}")


def _refusal_for(db: Session, workspace_id: Any, said: str) -> Optional[str]:
    from core.models.core import BoardTask
    from services.ticket_numbers import resolve_ticket_ref, ticket_number

    task_id = resolve_ticket_ref(db, workspace_id, said)
    card = (db.query(BoardTask).filter(BoardTask.id == task_id, BoardTask.workspace_id == workspace_id).first()
            if task_id else None)
    if card is None:
        return None
    number = ticket_number(db, card) or said
    return A_COPY.format(label=number, title=card.title, number=number)


def without_status_orders(brief: Any) -> Tuple[Any, bool]:
    """``brief`` without the sentences that tell its agent to change the card's status,
    and whether any were taken out. Anything that isn't text is left as it is."""
    if not isinstance(brief, str) or not brief.strip():
        return brief, False
    kept: List[str] = []
    taken = False
    for line in brief.split("\n"):
        sentences = _SENTENCE.split(line)
        left = [s for s in sentences if not _orders_a_move(s)]
        taken = taken or len(left) != len(sentences)
        if left or not sentences or sentences == [""]:
            kept.append(" ".join(left))
    return ("\n".join(kept).strip(), True) if taken else (brief, False)


def _orders_a_move(sentence: str) -> bool:
    return bool(_STATUS_TOOL.search(sentence) or _STATUS_ORDER.search(sentence))


__all__ = ["A_COPY", "STATUS_TAKEN_OUT", "checks_the_new_card", "copy_refusal", "without_status_orders"]
