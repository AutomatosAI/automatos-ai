"""Find the reported content in the caller's workspace (PRD-266 US-016a, K8).

Each resolver answers the target's excerpt (at most ``EXCERPT_MAX_CHARS``
characters) or ``None``. ``None`` covers every way the target is not the
reporter's to report: a malformed id, another workspace's row, a user's own
message, another member's private chat, a deleted Deliverable. The route maps
all of them to one 404, so a reporter learns nothing about content they cannot
see (never a 403, never a 500).

* A **message** is an assistant message of this workspace, in a chat of this
  workspace that the reporter owns or that is public. For a server key it is a
  chat its own service user holds, tagged with the end user the key names, and
  never a public one (``reporter``).
* A **Deliverable** is one ``DeliverableService.get_deliverable`` finds in this
  workspace (it already leaves out deleted ones). A server key reports none.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from sqlalchemy import and_, or_

from core.models.core import Chat, Message
from modules.memory.thread_checkpoint import extract_message_text
from services.content_reports import EXCERPT_MAX_CHARS
from services.content_reports.reporter import Reporter
from services.deliverable_service import DeliverableService

ASSISTANT_ROLE = "assistant"
PUBLIC_VISIBILITY = "public"
# Between a Deliverable's title and its summary in the excerpt.
TITLE_SUMMARY_SEPARATOR = " — "


def canonical_target_id(raw: str) -> Optional[str]:
    """The target id in its one canonical form (a lower-case UUID), or ``None``.

    Both target kinds are keyed by UUID. Parsing first keeps a malformed id away
    from Postgres (a failed cast would abort the request's transaction) and makes
    two spellings of one id the same report for the duplicate check.
    """
    try:
        return str(UUID(str(raw).strip()))
    except (TypeError, ValueError, AttributeError):
        return None


def clip_excerpt(text: str) -> str:
    """``text`` on one line, cut to ``EXCERPT_MAX_CHARS`` characters."""
    return " ".join((text or "").split())[:EXCERPT_MAX_CHARS]


def chats_it_may_report(reporter: Reporter) -> Any:
    """The chats whose replies ``reporter`` may report, as a filter on ``Chat``.

    A person: their own chats and the workspace's public ones. A server key: the chats its
    service user holds for the end user it names (or for none, when it names none).
    """
    if not reporter.is_key:
        return or_(Chat.user_id == reporter.pk, Chat.visibility == PUBLIC_VISIBILITY)
    if reporter.end_user_id is None:
        return and_(Chat.user_id == reporter.pk, Chat.end_user_id.is_(None))
    return and_(Chat.user_id == reporter.pk, Chat.end_user_id == reporter.end_user_id)


def message_excerpt(
    db: Any, *, workspace_id: UUID, reporter: Reporter, message_id: str
) -> Optional[str]:
    """The excerpt of an assistant message the reporter may see, or ``None``."""
    row = (
        db.query(Message.parts)
        .join(Chat, Chat.id == Message.chat_id)
        .filter(
            Message.id == UUID(message_id),
            Message.workspace_id == workspace_id,
            Message.role == ASSISTANT_ROLE,
            Chat.workspace_id == workspace_id,
            chats_it_may_report(reporter),
        )
        .first()
    )
    if row is None:
        return None
    return clip_excerpt(extract_message_text(row.parts))


async def deliverable_excerpt(db: Any, *, workspace_id: UUID, deliverable_id: str) -> Optional[str]:
    """The excerpt of a live Deliverable of this workspace (title, then summary), or ``None``."""
    found = await DeliverableService(db, workspace_id).get_deliverable(deliverable_id)
    deliverable = found.get("deliverable") if found.get("success") else None
    if not deliverable:
        return None
    parts = [deliverable.get("title"), deliverable.get("summary")]
    return clip_excerpt(TITLE_SUMMARY_SEPARATOR.join(str(p) for p in parts if p))


async def resolve_target(
    db: Any, *, workspace_id: UUID, reporter: Reporter, target_type: str, target_id: str
) -> Optional[str]:
    """The excerpt of the reported content, or ``None`` when it is not the reporter's to report.

    ``target_id`` must already be canonical (``canonical_target_id``). A server key reports a
    message only.
    """
    if target_type == "message":
        return message_excerpt(
            db, workspace_id=workspace_id, reporter=reporter, message_id=target_id
        )
    if target_type == "deliverable" and not reporter.is_key:
        return await deliverable_excerpt(db, workspace_id=workspace_id, deliverable_id=target_id)
    return None


__all__ = [
    "canonical_target_id",
    "chats_it_may_report",
    "clip_excerpt",
    "deliverable_excerpt",
    "message_excerpt",
    "resolve_target",
]
