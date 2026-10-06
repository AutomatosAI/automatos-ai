"""``platform_propose_brand_kit`` / ``platform_save_approved_brand_kit`` (PRD-255 US-014, FR-11).

The Brand designer changes the kit only after the owner approves a proposal card.

* **Propose.** The proposal (the fields ``platform_update_brand_kit`` takes) is
  checked by the kit's own validation (``brand_kit.proposed_brand_kit``) and laid over
  the stored kit WITHOUT saving it; the Brand Board is drawn from it in a child process
  (``brand_board_render``, as the Brand kit page draws it) into the ticket's folder,
  ``sessions/<ticket>/`` (F349's folder rule); and a question card is filed on the
  caller's own ticket (``kind='question'``, options Approve / Revise, free text always
  allowed) that lists what changes and links the board. The card carries the proposal
  in its ``details`` (``brand_proposal_card.PROPOSAL_MARKER``). Nothing else is written.
* **Save.** Only the newest proposal card on the caller's own ticket, only when its
  answer is a plain Approve (a question is answered by a workspace admin on the
  Questions tab, ``api/approval_grants.py``, or by a reply in the workspace's own
  Telegram chat, ``api/webhooks.py``), only once, and only while the stored kit is the one the
  proposal was drawn over. It is saved through ``platform_update_brand_kit``'s own
  handler (``update_brand_kit_tool``: the kit's validation and its one writer). The
  call carries no kit fields: what is saved is what the owner saw on the card.

Why a save action of its own: ``platform_update_brand_kit`` is ``admin_only`` (REST PUT
is workspace:manage), and an agent's call is no admin's, so an agent can never pass
it; the owner's Approve on the card is the admin's say-so for exactly that proposal.

The ticket is server-side: a session's (``_session_task_id``) or a board run's card
(``_board_task_id``), injected by ``session_ticket`` and stripped from what a caller
sends. A session's card is filed unparked (``raise_session_ask``: the turn's end parks
it); a board run's card parks the card, as ``platform_ask_human`` does.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from modules.tools.discovery import brand_proposal_card as card
from modules.tools.discovery.session_ticket import BOARD_CARD_PARAM, SESSION_TICKET_PARAM

logger = logging.getLogger(__name__)

PROPOSAL_PARAM, WHY_PARAM = "brand_kit", "why"
DESIGNER = "the Brand designer"   # the card's asker when the calling agent has no name
TERMINAL_STATUSES = ("done", "failed", "cancelled")

NEEDS_A_TICKET = ("{tool} works the ticket you are on (a session's, or a board card you are running), and this "
                  "call has none. Nothing was {done}.")
PROPOSAL_NOT_AN_OBJECT = "brand_kit is an object of kit fields, the ones platform_update_brand_kit takes."
UNKNOWN_FIELDS = ("The kit has no field {fields}: propose only the fields platform_update_brand_kit takes. "
                  "Nothing was asked.")
PROPOSAL_REFUSED = "The proposed kit is not valid, so nothing was asked: {reasons}"
NOTHING_CHANGES = "That proposal changes nothing in the kit as it is stored. Nothing was asked."
DRAW_FAILED = "The Brand Board could not be drawn from the proposal ({reason}). Nothing was asked."
TOO_LONG = ("That proposal's card would be {chars} characters, over {limit}: the owner must see every change in "
            "full, so propose fewer fields per card. Nothing was asked.")
NOT_WRITTEN = "The Brand Board was drawn, but your ticket's folder could not be written. Nothing was asked; try again."
NO_WORKSPACE = "Workspace not found"
TASK_NOT_FOUND = "ticket {ticket} is not in this workspace"
TASK_FINISHED = "ticket {ticket} is already {status}: a finished ticket cannot carry a proposal card"
NO_PROPOSAL = ("Your ticket has no proposal card: propose the change with propose_brand_kit and wait for the "
               "owner's Approve. Nothing was saved.")
KIT_CHANGED = ("The brand kit changed after you proposed (question #{ask}), so the board the owner approved is "
               "not the kit it would save. Nothing was saved: propose again over the kit as it is now.")
PROPOSED_NOTE = ("Proposal card filed on your ticket (question #{ask}); nothing is saved. Open {path} to look at "
                 "the board drawn from it. Finish what does not depend on the answer and end your turn: an "
                 "Approve lets save_approved_brand_kit save it; any other answer is a revision.")
SAVED_NOTE = ("Saved the proposal the owner approved (question #{ask}). Draw the Brand Board again "
              "(render_preview) to show the kit as it is now.")


def _failed(error: str, **extra: Any) -> Dict[str, Any]:
    return {"success": False, "error": error, **extra}


def ticket_of(params: Dict[str, Any]) -> Tuple[Optional[int], bool]:
    """The caller's ticket as the server injected it, and whether it is a session's."""
    for key, in_session in ((SESSION_TICKET_PARAM, True), (BOARD_CARD_PARAM, False)):
        ticket = params.get(key)
        if isinstance(ticket, int) and not isinstance(ticket, bool) and ticket > 0:
            return ticket, in_session
    return None, False


def _workspace(db: Session, workspace_id: UUID, lock: bool = False) -> Any:
    """The caller's workspace; ``lock`` holds its row until the commit (a save checks the kit, then writes it)."""
    from core.models.workspaces import Workspace

    query = db.query(Workspace).filter(Workspace.id == workspace_id)
    return (query.with_for_update() if lock else query).first()


def _agent_name(db: Session, workspace_id: UUID, agent_id: Any) -> Optional[str]:
    from core.models.core import Agent

    if not agent_id:
        return None
    row = db.query(Agent).filter(Agent.id == int(agent_id), Agent.workspace_id == workspace_id).first()
    return getattr(row, "name", None)


def _stored(workspace: Any) -> Any:
    from modules.documents.brand_kit import BRAND_KIT_SETTINGS_KEY

    return (getattr(workspace, "settings", None) or {}).get(BRAND_KIT_SETTINGS_KEY)


def _proposed(workspace: Any, proposal: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The kit the proposal makes of the stored one (validated, NOT saved), or why it makes none."""
    from pydantic import ValidationError

    from modules.documents.brand_kit import PATCH_FIELDS, brand_kit_errors, proposed_brand_kit

    if not isinstance(proposal, dict) or not proposal:
        return None, PROPOSAL_NOT_AN_OBJECT
    unknown = sorted(set(proposal) - set(PATCH_FIELDS))
    if unknown:
        return None, UNKNOWN_FIELDS.format(fields=", ".join(unknown))
    try:
        return proposed_brand_kit(workspace.settings, proposal), None
    except ValidationError as exc:
        reasons = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in brand_kit_errors(exc))
        return None, PROPOSAL_REFUSED.format(reasons=reasons)


def _draw_board(kit: Dict[str, Any]) -> bytes:
    """The Brand Board drawn from ``kit``, as a PNG, in a child process (blocking: run it off the loop)."""
    from modules.documents import brand_board_render
    from modules.documents.brand_fonts import brand_kit_for_media_render

    return brand_board_render.render_board_isolated(brand_kit_for_media_render(kit), brand_board_render.BOARD_PNG)


async def _board_path(workspace_id: UUID, ticket: int, proposal: Dict[str, Any],
                      kit: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """The board drawn from the proposed kit, written into the ticket's folder: its path, or why not."""
    from modules.documents.thumbnails.render import ThumbnailError
    from modules.tools.execution.session_document_folder import write_to_session

    try:
        data = await asyncio.to_thread(_draw_board, kit)
    except ThumbnailError as exc:
        logger.warning("[brand-proposal] ticket %s: the board was not drawn: %s", ticket, exc)
        return None, DRAW_FAILED.format(reason=exc)
    path = await write_to_session(workspace_id, ticket, card.board_name(proposal), data)
    return (path, None) if path else (None, NOT_WRITTEN)


def _open_ticket(db: Session, workspace_id: UUID, ticket: int) -> Tuple[Any, Optional[str]]:
    """The caller's ticket in this workspace, still open; or why a card cannot go on it."""
    from core.models.core import BoardTask

    task = db.query(BoardTask).filter(BoardTask.id == ticket, BoardTask.workspace_id == workspace_id).first()
    if task is None:
        return None, TASK_NOT_FOUND.format(ticket=ticket)
    if task.status in TERMINAL_STATUSES:
        return None, TASK_FINISHED.format(ticket=ticket, status=task.status)
    return task, None


async def _file_on_board_card(db: Session, workspace_id: UUID, task: Any, ask: Dict[str, Any]) -> Dict[str, Any]:
    """A board run's card: parked behind the question, as ``platform_ask_human`` parks it."""
    from modules.tools.discovery.handlers_asks import stage_question

    return await stage_question(db, workspace_id, subject_type="board_task", subject_id=str(task.id), park=task, **ask)


async def _file_in_session(db: Session, workspace_id: UUID, task: Any, ask: Dict[str, Any]) -> Dict[str, Any]:
    """A session's ticket: filed unparked (the turn's end parks it), one open question at a time."""
    from services.cli_host_service import raise_session_ask

    filed = await raise_session_ask(db, task_id=task.id, workspace_id=workspace_id, agent_id=ask["asked_by_agent_id"],
                                    agent_name=ask["agent_name"], question=ask["question"], options=ask["options"],
                                    details=ask["details"])
    result = filed.get("result")
    return {"success": True, **result} if filed.get("success") and isinstance(result, dict) else filed


async def _prepared(db: Session, workspace_id: UUID, ticket: int,
                    params: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The checked proposal, what it changes and the board drawn from it, or why there is no card to file."""
    from modules.documents.brand_kit import get_brand_kit
    from modules.tools.execution.session_document_folder import session_copy_path

    workspace = _workspace(db, workspace_id)
    if workspace is None:
        return None, NO_WORKSPACE
    proposal = params.get(PROPOSAL_PARAM)
    kit, problem = _proposed(workspace, proposal)
    if kit is None:
        return None, problem
    changed = card.changes(get_brand_kit(workspace.settings), kit, list(proposal))
    if not changed:
        return None, NOTHING_CHANGES
    name = _agent_name(db, workspace_id, params.get("_agent_id")) or DESIGNER
    text = card.card_text(name, str(params.get(WHY_PARAM) or ""), changed, session_copy_path(ticket, card.board_name(proposal)))
    if card.too_long(text):
        return None, TOO_LONG.format(chars=len(text), limit=card.MAX_CARD_CHARS)
    path, problem = await _board_path(workspace_id, ticket, proposal, kit)
    if path is None:
        return None, problem
    marker = {"proposal": dict(proposal), "board_path": path, "base": card.fingerprint(_stored(workspace))}
    return {"changed": changed, "path": path, "marker": marker, "text": text, "name": name}, None


def _ask(params: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    """The card: its text, the options Approve / Revise, the asker, and the proposal in its details."""
    agent_id = params.get("_agent_id")
    return {"question": prepared["text"], "options": list(card.CARD_OPTIONS),
            "asked_by_agent_id": int(agent_id) if agent_id else None, "agent_name": prepared["name"],
            "details": {card.PROPOSAL_MARKER: prepared["marker"]}}


async def propose_brand_kit(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Check a kit proposal, draw the Brand Board from it unsaved, and file the Approve / Revise card on the ticket."""
    ticket, in_session = ticket_of(params)
    if ticket is None:
        return _failed(NEEDS_A_TICKET.format(tool="propose_brand_kit", done="asked"))
    task, problem = _open_ticket(db, workspace_id, ticket)
    if task is None:
        return _failed(problem or TASK_NOT_FOUND.format(ticket=ticket))
    waiting = card.latest_proposal(_ticket_grants(db, workspace_id, ticket))
    if waiting is not None and card.status_of(waiting) == card.PENDING:
        return _failed(card.STILL_WAITING.format(ask=waiting.id), ask_id=waiting.id)
    prepared, problem = await _prepared(db, workspace_id, ticket, params)
    if prepared is None:
        return _failed(problem or NOTHING_CHANGES)
    file_it = _file_in_session if in_session else _file_on_board_card
    filed = await file_it(db, workspace_id, task, _ask(params, prepared))
    if not filed.get("success"):
        return filed
    ask_id, path = filed.get("ask_id"), prepared["path"]
    logger.info("[brand-proposal] ticket %s: proposal card #%s filed (%d changes)", ticket, ask_id, len(prepared["changed"]))
    return {"success": True, "ask_id": ask_id, "board_path": path, "changes": prepared["changed"], "saved": False,
            "note": PROPOSED_NOTE.format(ask=ask_id, path=path)}


def _ticket_grants(db: Session, workspace_id: UUID, ticket: int, lock: bool = False) -> list:
    """The question cards on this workspace's ticket; ``lock`` holds them while one is saved."""
    from core.models.approval_grants import KIND_QUESTION, SUBJECT_BOARD_TASK, ApprovalGrant

    query = db.query(ApprovalGrant).filter(
        ApprovalGrant.workspace_id == workspace_id, ApprovalGrant.subject_type == SUBJECT_BOARD_TASK,
        ApprovalGrant.subject_id == str(ticket), ApprovalGrant.kind == KIND_QUESTION)
    return (query.with_for_update() if lock else query).all()


def _mark_saved(grant: Any, marker: Dict[str, Any]) -> None:
    """Record the save on the card (a new dict), so the same Approve never saves twice."""
    saved = {**marker, "saved_at": datetime.now(timezone.utc).isoformat()}
    grant.details = {**(grant.details or {}), card.PROPOSAL_MARKER: saved}
    flag_modified(grant, "details")


async def save_approved_brand_kit(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Save the proposal the owner approved on the caller's own ticket, through platform_update_brand_kit's handler."""
    from modules.tools.discovery.handlers_documents import update_brand_kit_tool

    ticket, _in_session = ticket_of(params)
    if ticket is None:
        return _failed(NEEDS_A_TICKET.format(tool="save_approved_brand_kit", done="saved"))
    grant = card.latest_proposal(_ticket_grants(db, workspace_id, ticket, lock=True))
    if grant is None:
        return _failed(NO_PROPOSAL)
    problem = card.why_not_saved(grant)
    if problem:
        return _failed(problem, ask_id=grant.id)
    workspace = _workspace(db, workspace_id, lock=True)  # held to the commit: no kit write between check and save
    if workspace is None:
        return _failed(NO_WORKSPACE)
    marker = card.marker_of(grant)
    if card.fingerprint(_stored(workspace)) != marker.get("base"):
        return _failed(KIT_CHANGED.format(ask=grant.id), ask_id=grant.id)
    _mark_saved(grant, marker)
    saved = await update_brand_kit_tool(db, workspace_id, dict(marker["proposal"]))  # commits the kit and the mark
    if not saved.get("success"):
        db.rollback()
        return saved
    logger.info("[brand-proposal] ticket %s: the approved proposal #%s saved", ticket, grant.id)
    return {**saved, "ask_id": grant.id, "note": SAVED_NOTE.format(ask=grant.id)}


__all__ = ["propose_brand_kit", "save_approved_brand_kit", "ticket_of"]
