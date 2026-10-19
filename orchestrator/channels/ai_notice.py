"""The AI notice on channel bot messages (PRD-264 EUAI-03, EU AI Act Art. 50(1)).

A person messaging a workspace on Telegram, Slack, Discord or WhatsApp is told,
on first contact, that the reply comes from an AI and for whom: "AI assistant
for {principal}." The rule for *when* lives in
:func:`modules.policy.ai_disclosure.channel_notice_due`; this module holds the
state it needs. The principal's public name comes from
:func:`core.public_principal.workspace_principal_name` (FR-15).

Both outbound channel paths call it, and nothing else sends to a channel:

* the adapter path, ``BaseChannelAdapter._send_reply`` (``channels/base.py``);
* the driver path, ``send_to_channel`` (``channels/sender.py``), which carries
  webhook replies, notifications, agent asks and destination dispatches.

State: the last-notice time per (workspace, connection, person) in Redis, with
a TTL equal to the idle gap (``config.AI_DISCLOSURE_IDLE_HOURS``). The person
id is hashed in the key (it can be a phone number). When Redis can't be read
the notice is added: losing the state only means a person sees it again. The
time is recorded only after a send went out, so a failed send never uses up
the notice.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from config import config
from core.public_principal import workspace_principal_name
from modules.policy import ai_disclosure

logger = logging.getLogger(__name__)

NOTICE_KEY_PREFIX = "ai_notice:channel"
SECONDS_PER_HOUR = 3600
MIN_TTL_SECONDS = 1


@dataclass(frozen=True)
class ChannelNotice:
    """The notice decision for one outbound message.

    ``text`` is the notice to show, or ``None`` when this message doesn't need
    one. ``key`` is the Redis key :func:`record_notice_sent` writes once the
    message went out.
    """

    key: str
    text: Optional[str]

    def apply(self, message: str) -> str:
        """``message`` with the notice as its first paragraph, when one is due."""
        return ai_disclosure.prepend_notice(self.text, message)


def _clock() -> float:
    """Wall-clock seconds; a seam the tests replace with a fake clock."""
    return time.time()


def idle_seconds() -> int:
    """The idle gap after which the notice is shown again, in seconds."""
    return int(config.AI_DISCLOSURE_IDLE_HOURS) * SECONDS_PER_HOUR


def notice_key(workspace_id: Any, connection_id: Any, person_id: Any) -> str:
    """The Redis key for one person on one connection of one workspace."""
    person = hashlib.sha256(str(person_id).encode("utf-8")).hexdigest()
    return f"{NOTICE_KEY_PREFIX}:{workspace_id}:{connection_id}:{person}"


def _get_redis() -> Any:
    """The platform Redis connection, or ``None`` when it is not available."""
    try:
        from core.redis.client import get_redis_client

        client = get_redis_client()
        return client.get_redis() if client is not None else None
    except Exception:
        logger.exception("ai notice: Redis client unavailable; the notice is added")
        return None


def _read_last_notice(key: str) -> Optional[float]:
    """The last-notice time for ``key``, or ``None`` if none or unreadable."""
    redis = _get_redis()
    if redis is None:
        logger.warning("ai notice: Redis not configured; the notice is added")
        return None
    try:
        raw = redis.get(key)
    except Exception:
        logger.exception("ai notice: Redis read failed; the notice is added")
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("ai notice: unreadable last-notice value; the notice is added")
        return None


def _write_last_notice(key: str, sent_at: float) -> None:
    """Record ``sent_at`` for ``key`` with a TTL equal to the idle gap."""
    redis = _get_redis()
    if redis is None:
        return
    try:
        redis.set(key, repr(sent_at), ex=max(MIN_TTL_SECONDS, idle_seconds()))
    except Exception:
        logger.exception("ai notice: Redis write failed; the next message repeats the notice")


def _notice_due_sync(key: str, inbound_text: Optional[str]) -> bool:
    """Read the state for ``key`` and apply the one rule in ``ai_disclosure``."""
    return ai_disclosure.channel_notice_due(
        last_notice_at=_read_last_notice(key),
        now=_clock(),
        idle_seconds=idle_seconds(),
        inbound_text=inbound_text,
    )


async def notice_for(
    db: Any,
    *,
    workspace_id: Any,
    connection_id: Any,
    person_id: Any,
    inbound_text: Optional[str] = None,
) -> ChannelNotice:
    """Decide whether a message to ``person_id`` carries the AI notice.

    ``inbound_text`` is the message being answered, when there is one: a
    question about talking to a human or a bot always gets the notice.
    """
    key = notice_key(workspace_id, connection_id, person_id)
    due = await asyncio.to_thread(_notice_due_sync, key, inbound_text)
    if not due:
        return ChannelNotice(key=key, text=None)
    return ChannelNotice(key=key, text=ai_disclosure.channel_notice(workspace_principal_name(db, workspace_id)))


async def record_notice_sent(notice: ChannelNotice) -> None:
    """Record that ``notice`` reached the person; a no-op when none was due."""
    if notice.text is None:
        return
    await asyncio.to_thread(_write_last_notice, notice.key, _clock())
