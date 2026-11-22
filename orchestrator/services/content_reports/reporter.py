"""Who files a content report: a person, or a product's server key for one of its end users.

**A person** reports as before: their ``users.id`` (``resolve_user_pk``), from any chat of theirs or
any public chat of the workspace. A person names no end user: the route refuses one (422).

**A server key** (``ak_srv_`` listing ``reports:write``, ``core/auth/server_key_gate.py``) has no
person. It reports as its own service user, the ``users`` row its chat turns run as
(``api/chat_server_key.service_user_for``), and reaches only a message in a chat that user holds,
tagged with the end user the key names (``chats.end_user_id``), or tagged with none when it names
none. It never reaches a member's chat, a public chat, another key's chat or a Deliverable
(``targets``). A key that has never chatted has no service user and so no chat: whatever it
reports is not found. Nothing here makes a service user. An end user whose consent is gone, or
whose erase has started, is not reported for (``consent_seen``, ``consent_held``), as the chat path
refuses their turn.

A key's reports count against one bucket per end user it names, so one person's reports never
use up another's; the key's own per-key limits cap its total, in the gate. The audit row names the
key, never the end user: the end-user erase leaves audit rows alone, so no report links to a
person after they are erased.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

from core.auth.principal import resolve_user_pk
from core.auth.server_key_gate import API_KEY_FILER, is_server_key

PERSON_ACTOR_TYPE = "user"
# The audit row's actor for a report a server key filed: the same name a ticket it files carries.
KEY_ACTOR_TYPE = API_KEY_FILER
# Between the key's service user and the end user in the rate limiter's subject.
END_USER_SUBJECT_SEPARATOR = ":end_user:"


@dataclass(frozen=True)
class Reporter:
    """Who a report is filed as. ``pk`` is ``None`` when there is no one to file it as."""

    pk: Optional[int]
    api_key_id: Optional[str] = None
    end_user_id: Optional[str] = None

    @property
    def is_key(self) -> bool:
        """Whether a server key files it."""
        return self.api_key_id is not None

    @property
    def actor_type(self) -> str:
        """The audit row's actor: a person, or a server key."""
        return KEY_ACTOR_TYPE if self.is_key else PERSON_ACTOR_TYPE

    @property
    def limit_subject(self) -> str:
        """The rate limiter's bucket: the person, the key's end user, or the key itself."""
        if self.is_key and self.end_user_id is not None:
            return f"{self.pk}{END_USER_SUBJECT_SEPARATOR}{self.end_user_id}"
        return str(self.pk)

    def audit_details(self) -> dict[str, Any]:
        """What the audit row adds about who filed it: for a key, the key (never its end user)."""
        return {"api_key_id": self.api_key_id} if self.is_key else {}


def reporter_of(db: Any, ctx: Any, end_user_id: Optional[str] = None) -> Reporter:
    """The reporter for this request: the key's service user, or the request's person."""
    if is_server_key(ctx):
        key_id = getattr(ctx, "api_key_id", None)
        return Reporter(pk=_service_user_of(db, key_id), api_key_id=str(key_id), end_user_id=end_user_id)
    return Reporter(pk=resolve_user_pk(db, ctx))


# What consent_seen answers when there is no end user to check: a person, or a key naming none.
NO_END_USER = ""


def consent_seen(db: Any, workspace_id: Any, reporter: Reporter) -> Optional[str]:
    """For a key naming an end user: the id of their consent record while no erase of them has begun,
    or None when their consent is gone or their erase has begun. NO_END_USER for anyone else."""
    if not reporter.is_key or reporter.end_user_id is None:
        return NO_END_USER
    from core.models.end_users import consent_record_id

    return consent_record_id(db, workspace_id, reporter.end_user_id)


def consent_held(db: Any, workspace_id: Any, reporter: Reporter, seen: str) -> bool:
    """Whether the consent record ``consent_seen`` found is still there with no erase begun.

    The record is read ``FOR SHARE``. The erase marks its start by updating that row
    (``api.end_user_memory._mark_erasing``), so it waits until this transaction ends, and a mark made
    first is seen once it commits. It must be the SAME record: an erase and a fresh consent that both
    finished meanwhile leave a new record, and the excerpt read before them is of a message the erase
    deleted. The service calls this right before it writes the audit row, in the transaction that
    commits it, so no report lands once the consent it was filed under is gone.
    """
    if seen == NO_END_USER:
        return True
    from core.models.end_users import EndUser

    held = (
        db.query(EndUser.id)
        .filter(EndUser.id == uuid.UUID(seen), EndUser.workspace_id == workspace_id,
                EndUser.end_user_id == reporter.end_user_id, EndUser.erasing_at.is_(None))
        .with_for_update(read=True)
        .first()
    )
    return held is not None


def _service_user_of(db: Any, key_id: Any) -> Optional[int]:
    """The ``users.id`` the key's chat turns run as, or ``None`` when it has never chatted."""
    from core.models.sdk_api_keys import SdkApiKey

    key = db.get(SdkApiKey, key_id) if key_id is not None else None
    return None if key is None else key.service_user_id


__all__ = ["KEY_ACTOR_TYPE", "NO_END_USER", "PERSON_ACTOR_TYPE", "Reporter", "consent_held", "consent_seen",
           "reporter_of"]
