"""PRD-268 §6.11.5, K15 — the offer: consent first, then one calibration, posted as one message.

The person answers the consent line (§12-Q22) once per workspace; ``services.calibration.claim``
takes the claim, committed, before anything is read:

* **Not now** (consent false): the claim is taken as ``declined`` and audited. Nothing is read
  and no model is called.
* **Go ahead** (consent true): the claim is taken as ``open`` and committed; then the build
  (``services.calibration.build``: titles and kinds only, at most one model call), which the
  record holds, posted as ONE Auto message ``[text, calibration@1]`` through
  ``deliver_background_message`` (W2's parts, ``services.chat_parts``). It lands in ``chat_id``
  when that is the person's own private chat in the workspace and not a widget visitor's (a
  widget chat's ``user_id`` is the visitor's to give), else in their Auto thread; its source is
  {origin ``calibration``, calibration_id}, and it is never announced (it answers the person's
  own click). The record keeps the chat and the message, and consent is audited after the commit.
* A build or a post that fails (the model's refusal or error, an answer it could not read, a
  message that did not land, a person the messenger would find as someone else:
  ``recorded_self``) gives the claim back and raises ``OfferFailed``: the person is offered
  again, and the cause stays in the server log.
* Offered already (consented, declined or answered; also when two calls race): ``AlreadyOffered``,
  carrying the record as it is. The one exception is a claim whose claimant died before its post
  (``claim.take_over``, once it is stale): the next consent takes it over and offers again,
  unless the message did land (a claimant that died recording it), which is then recorded and
  answered as offered.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from core.models.core import Chat, Message
from core.models.proactive import CALIBRATION_DECLINED, CALIBRATION_OPEN
from services.calibration import claim
from services.calibration.audit import CONSENTED, DECLINED, audit
from services.calibration.build import build_calibration
from services.calibration.model import PART_TYPE, Calibration
from services.calibration.strings import calibration_language, calibration_text
from services.chat_messenger import deliver_background_message
from services.mobile.digest import recorded_self

__all__ = ["AlreadyOffered", "ORIGIN", "OfferFailed", "Offered", "consent", "decline", "own_chat"]

logger = logging.getLogger(__name__)

# The posted message's source origin: not in ANNOUNCED_ORIGINS (auto_message.py), so never announced.
ORIGIN = "calibration"
PRIVATE = "private"
ASSISTANT = "assistant"


class AlreadyOffered(Exception):
    """The person was offered calibration in this workspace already."""

    def __init__(self, record: Optional[Dict[str, Any]]) -> None:
        super().__init__("calibration was offered already")
        self.record = record


class OfferFailed(Exception):
    """The calibration could not be built or posted; the claim was given back."""


class _NotPosted(Exception):
    """The calibration message did not land."""


@dataclass(frozen=True)
class Offered:
    """A calibration posted: the ``Calibration@1`` and the message it was posted as."""

    calibration: Dict[str, Any]
    chat_id: str
    message_id: str

    def to_dict(self) -> Dict[str, Any]:
        """The route's answer."""
        return {"status": CALIBRATION_OPEN, "calibration": dict(self.calibration), "chat_id": self.chat_id,
                "message_id": self.message_id}


def own_chat(db: Session, workspace_id: Any, user_id: int, chat_id: Any) -> Optional[str]:
    """``chat_id`` while it is the person's own private chat in the workspace and no widget
    visitor's; None otherwise (the post then goes to their Auto thread)."""
    if chat_id is None:
        return None
    found = (
        db.query(Chat.id)
        .filter(
            Chat.id == uuid.UUID(str(chat_id)),
            Chat.workspace_id == uuid.UUID(str(workspace_id)),
            Chat.user_id == int(user_id),
            Chat.widget_key_id.is_(None),
            Chat.visibility == PRIVATE,
        )
        .first()
    )
    return str(found.id) if found is not None else None


def _existing(db: Session, workspace_id: Any, user_id: int) -> Optional[Dict[str, Any]]:
    record = claim.record_of(db, workspace_id, user_id)
    return claim.view(record) if record is not None else None


def decline(db: Session, workspace_id: Any, user_id: int, now: datetime) -> Dict[str, Any]:
    """Not now: the claim taken as declined and audited; nothing read, no model call."""
    lease = claim.claim(db, workspace_id, user_id, consent=False, now=now)
    if lease is None:
        raise AlreadyOffered(_existing(db, workspace_id, user_id))
    audit(db, workspace_id, user_id, DECLINED, lease.id, {})
    return {"status": CALIBRATION_DECLINED, "calibration": None, "chat_id": None, "message_id": None}


def _landed(db: Session, workspace_id: Any, user_id: int, calibration_id: Any) -> Optional[Message]:
    """The calibration's message, when it was posted into one of the person's chats in the workspace."""
    return (
        db.query(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .filter(
            Chat.workspace_id == uuid.UUID(str(workspace_id)),
            Chat.user_id == int(user_id),
            Message.workspace_id == uuid.UUID(str(workspace_id)),
            Message.role == ASSISTANT,
            Message.source["origin"].astext == ORIGIN,
            Message.source["calibration_id"].astext == str(calibration_id),
        )
        .first()
    )


def _taken_over(db: Session, workspace_id: Any, user_id: int, now: datetime) -> Optional[claim.Lease]:
    """The person's claim taken over when its claimant died before recording its post; None when
    there is none to take (the record is then answered as it is)."""
    record = claim.record_of(db, workspace_id, user_id)
    if record is None or record.status != CALIBRATION_OPEN or record.message_id is not None:
        return None
    ident = record.id
    message = _landed(db, workspace_id, user_id, ident)
    if message is not None:
        claim.posted(db, ident, message.chat_id, message.id)
        return None
    return claim.take_over(db, workspace_id, user_id, ident, now)


def _post(db: Session, workspace_id: Any, user_id: int, built: Calibration, saved: Dict[str, Any], said: str,
          chat_id: Any) -> Message:
    """Post the calibration as one Auto message; raises ``_NotPosted`` when it did not land, or
    before posting when the messenger would find someone else from the person's record (its owner
    check and Auto-thread fallback would act for them)."""
    person = recorded_self(db, user_id)
    if person is None:
        raise _NotPosted("the messenger would not find this person")
    message = deliver_background_message(
        db,
        workspace_id=workspace_id,
        text=calibration_text(said, built),
        source={"origin": ORIGIN, "calibration_id": built.id},
        chat_id=own_chat(db, workspace_id, user_id, chat_id),
        clerk_user_id=person,
        parts=[{"type": PART_TYPE, "calibration": saved}],
    )
    if message is None:
        raise _NotPosted("the calibration message did not land")
    return message


def consent(db: Session, workspace_id: Any, user_id: int, *, language: Optional[str], chat_id: Any,
            now: datetime) -> Offered:
    """Go ahead: claim, build, hold, post one message, and record where it was posted."""
    lease = claim.claim(db, workspace_id, user_id, consent=True, now=now) or _taken_over(db, workspace_id, user_id, now)
    if lease is None:
        raise AlreadyOffered(_existing(db, workspace_id, user_id))
    said = calibration_language(language)
    try:
        built = build_calibration(db, workspace_id, user_id, said, calibration_id=str(lease.id))
        saved = built.model_dump(mode="json")
        lease = claim.hold(db, lease, saved, datetime.now(timezone.utc))
        message = _post(db, workspace_id, user_id, built, saved, said, chat_id)
    except claim.LeaseLost:
        raise AlreadyOffered(_existing(db, workspace_id, user_id)) from None
    except Exception as failure:
        logger.exception("[calibration] workspace %s: the calibration was not offered; the claim is given back",
                         workspace_id)
        db.rollback()
        claim.release(db, lease)
        raise OfferFailed("the calibration could not be offered") from failure
    claim.posted(db, lease.id, message.chat_id, message.id)
    audit(db, workspace_id, user_id, CONSENTED, lease.id, {"lines": len(saved["lines"]), "empty": saved["empty"]})
    return Offered(calibration=saved, chat_id=str(message.chat_id), message_id=str(message.id))
