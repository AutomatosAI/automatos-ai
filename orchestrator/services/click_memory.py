"""PRD-256 P256-FIX-N13-4 (night 13, F411): the owner's click reaches what Auto knows.

Night 13: approving #1403 moved the card to Done with the note, and Auto said itself (ask 416)
that a card approval "doesn't update what I know": the click changed the card, nothing reached
memory, and later chats (asks 392, 415) did not know the decision.

Now each click on a call's approval card (api/approval_grants ``grant_approval`` and
``deny_approval``, wrapped by :func:`remembers_the_decision`) is stored as ONE memory, after the
route has run it and committed: the owner's decision, the card's own words, the day, the note
the card showed and what came of it ("It ran: number #1403." / "It did not go through: …");
a decline says nothing ran. It goes through the existing store-memory path (the
``platform_store_memory`` handler: the exclusion policy, the write contract's metadata) as a
``decision``, workspace-shared, owned by the user who clicked (``granted_by`` / ``revoked_by``,
set by the server from the request) and evidenced by its grant.

Only a call's card is remembered (``tool_call``, never a question: an answer has its own path).
A click the route refuses (already decided, 422) raises before this runs, so one grant is one
memory. A memory that is not stored is logged with its reason; the click has already run.
"""
from __future__ import annotations

import functools
import inspect
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

Route = Callable[..., Awaitable[Any]]

APPROVED, DECLINED = "approved", "declined"
MEMORY_TYPE = "decision"
SCOPE = "workspace"
SOURCE_TYPE = "platform_verified"  # the owner's own click
EVIDENCE = "approval_grant:{id}"
USER_ACTOR = "user:"
UNKNOWN_ACTOR = "unknown"
CARD_NOTES = ("- note: ", "- comment: ")  # the note as the card showed it (card_question_text.said_line)
CARD_CHARS = 160
NOTE_CHARS = 200
ERROR_CHARS = 200
CHAT_LANE = "chat"
MEMORY = 'The owner {decision} the card "{card}" on {day}{note}. {outcome}'
NOTE = ", note: {note}"
RAN = "It ran."
RAN_WITH = "It ran: {ids}."
NOT_THROUGH = "It did not go through: {error}"
NOTHING_RAN = "Nothing ran."
NO_REASON = "no reason given"


def remembers_the_decision(decision: str) -> Callable[[Route], Route]:
    """Wrap an approval route (``grant_id``, ``ctx``, ``db``) that records ``decision``
    (APPROVED or DECLINED): once it has run the owner's click, the decision is remembered.
    The route says what was decided, never the grant's status: a single-use yes is retired
    (revoked, ``system:consumed``) by the run it allowed. A refused click raises first."""
    def wrap(route: Route) -> Route:
        # Resolved in the route's own module: FastAPI reads string annotations against the
        # wrapper's globals, so the route's signature is handed over with its types already read.
        signature = inspect.signature(route, eval_str=True)

        @functools.wraps(route)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = await route(*args, **kwargs)
            call = signature.bind(*args, **kwargs).arguments
            await remember_the_click(call.get("db"), call.get("ctx"), call.get("grant_id"), decision)
            return out
        wrapped.__signature__ = signature  # type: ignore[attr-defined]
        return wrapped
    return wrap


async def remember_the_click(db: Any, ctx: Any, grant_id: Any, decision: str) -> Optional[Dict[str, Any]]:
    """Store ``decision`` on grant ``grant_id`` in the caller's workspace: the store's answer,
    or None when the grant is not a call's card."""
    try:
        grant = _a_calls_card(db, getattr(ctx, "workspace_id", None), grant_id)
        if grant is None:
            return None
        stored = await _store(db, grant, decision_memory(grant, decision), owner_of(grant, decision))
    except Exception as exc:  # the click has run: the lost memory is logged, and the answer says so
        logger.exception("[click_memory] the decision on grant %s was not remembered", grant_id)
        return {"success": False, "error": str(exc)[:ERROR_CHARS]}
    if not (isinstance(stored, dict) and stored.get("success")):
        logger.warning("[click_memory] the decision on grant %s was not remembered: %s", grant_id,
                       stored.get("error") if isinstance(stored, dict) else stored)
    return stored


def decision_memory(grant: Any, decision: str) -> str:
    """'The owner approved the card "Approve (move to Done) #1403 'Autumn tier launch copy'" on
    12 Oct 2026, note: Ship it. It ran: number #1403.' A decline names no note and says nothing ran."""
    details = grant.details if isinstance(grant.details, dict) else {}
    if decision == DECLINED:
        return MEMORY.format(decision=decision, card=card_words(grant), day=_day(grant.revoked_at), note="",
                             outcome=NOTHING_RAN)
    return MEMORY.format(decision=decision, card=card_words(grant), day=_day(grant.granted_at),
                         note=_note(grant), outcome=_outcome(details.get("executed_result"))).strip()


def card_words(grant: Any) -> str:
    """What the card asked, its question's first line ('Approve (move to Done) #1403 …'), else the action."""
    asked = str(getattr(grant, "question_md", None) or "").strip()
    head = asked.splitlines()[0].rstrip(":.").strip() if asked else ""
    return (head or str(getattr(grant, "tool_name", None) or "the call"))[:CARD_CHARS]


def owner_of(grant: Any, decision: str) -> Optional[str]:
    """Who clicked, as the store reads its owner (a ``users.id`` or a Clerk id), from the
    actor the route set ('user:7': ``granted_by`` on a yes, ``revoked_by`` on a no); None
    when the route could not name them."""
    actor = str((grant.granted_by if decision == APPROVED else grant.revoked_by) or "")
    who = actor.removeprefix(USER_ACTOR) if actor.startswith(USER_ACTOR) else ""
    return who if who and who != UNKNOWN_ACTOR else None


def _a_calls_card(db: Any, workspace_id: Any, grant_id: Any) -> Any:
    """The grant, in this workspace only, when it is a call's approval card (not a question)."""
    from core.models.approval_grants import KIND_QUESTION, SUBJECT_TOOL_CALL, ApprovalGrant

    if db is None or workspace_id is None or grant_id is None:
        return None
    grant = (db.query(ApprovalGrant)
             .filter(ApprovalGrant.id == int(grant_id), ApprovalGrant.workspace_id == workspace_id).first())
    if grant is None or grant.kind == KIND_QUESTION or grant.subject_type != SUBJECT_TOOL_CALL:
        return None
    return grant


async def _store(db: Any, grant: Any, content: str, owner: Optional[str]) -> Dict[str, Any]:
    """Through the store-memory tool's own handler: one owner-attributed decision."""
    from modules.tools.discovery.handlers_workspace import store_memory

    details = grant.details if isinstance(grant.details, dict) else {}
    chat = details.get("conversation_id") if details.get("lane") == CHAT_LANE else None
    params = {"content": content, "type": MEMORY_TYPE, "scope": SCOPE, "source_type": SOURCE_TYPE,
              "evidence_uri": EVIDENCE.format(id=grant.id), "_user_id": owner, "_origin_chat_id": chat}
    return await store_memory(db, grant.workspace_id, params)


def _note(grant: Any) -> str:
    """', note: Ship it' when the card showed the owner a note, else ''. Read from the card's own
    lines, never the call's params: the memory keeps only what the owner saw when they clicked."""
    lines = str(getattr(grant, "question_md", None) or "").splitlines()
    said = next((line.removeprefix(mark) for line in lines for mark in CARD_NOTES if line.startswith(mark)), "")
    said = " ".join(said.split())[:NOTE_CHARS]
    return NOTE.format(note=said) if said else ""


def _outcome(result: Any) -> str:
    """What the click's run came to (``executed_result``): ran, with the ids it named, or why not."""
    if not isinstance(result, dict) or "success" not in result:
        return ""
    if not result.get("success"):
        return NOT_THROUGH.format(error=str(result.get("error") or NO_REASON)[:ERROR_CHARS])
    ids = result.get("ids") if isinstance(result.get("ids"), dict) else {}
    return RAN_WITH.format(ids=", ".join(f"{key} {value}" for key, value in ids.items())) if ids else RAN


def _day(when: Any) -> str:
    """'12 Oct 2026': the day of the click (today when the row has none)."""
    when = when if isinstance(when, datetime) else datetime.now(timezone.utc)
    return f"{when.day} {when:%b %Y}"


__all__ = ["APPROVED", "DECLINED", "card_words", "decision_memory", "owner_of", "remember_the_click", "remembers_the_decision"]
