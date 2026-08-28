"""
Webhook ingestor (general workspace webhook).

Normalises an incoming webhook POST body into a RequestEnvelope
so general workspace webhooks flow through the universal router.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from uuid import UUID

from core.models.routing import (
    ChannelSource,
    RequestEnvelope,
    RequestUser,
)
from core.routing.ingestors.base import BaseIngestor


def _telegram_content(m: Any) -> str:
    """Operator-visible text carried by a Telegram message / edited_message.

    Beyond ``text``/``caption``, a media or service message carries
    attacker-controllable text in a sub-object: a group ``new_chat_title`` or a
    joiner's ``new_chat_members[].first_name`` (service messages), a
    document/audio/video/voice ``file_name``, a ``poll`` question, a ``contact``
    name, a ``venue`` title/address, a ``sticker`` emoji. Returns ``""`` for an
    update that carries no such text (a bare membership change, a delivery
    callback).
    """
    if not isinstance(m, dict):
        return ""

    v = m.get("text") or m.get("caption")
    if isinstance(v, str) and v.strip():
        return v

    # Service messages: a group rename or a joiner's self-chosen display name.
    title = m.get("new_chat_title")
    if isinstance(title, str) and title.strip():
        return title
    members = m.get("new_chat_members")
    if isinstance(members, list):
        joined = " ".join(
            p.strip()
            for member in members if isinstance(member, dict)
            for p in (member.get("first_name"), member.get("last_name"))
            if isinstance(p, str) and p.strip()
        )
        if joined:
            return joined

    poll = m.get("poll")
    if isinstance(poll, dict) and isinstance(poll.get("question"), str) and poll["question"].strip():
        return poll["question"]

    contact = m.get("contact")
    if isinstance(contact, dict):
        name = " ".join(
            p.strip() for p in (contact.get("first_name"), contact.get("last_name"))
            if isinstance(p, str) and p.strip()
        )
        if name:
            return name

    venue = m.get("venue")
    if isinstance(venue, dict):
        place = " ".join(
            p.strip() for p in (venue.get("title"), venue.get("address"))
            if isinstance(p, str) and p.strip()
        )
        if place:
            return place

    for fkey in ("document", "audio", "video", "voice"):
        f = m.get(fkey)
        if isinstance(f, dict) and isinstance(f.get("file_name"), str) and f["file_name"].strip():
            return f["file_name"]

    sticker = m.get("sticker")
    if isinstance(sticker, dict) and isinstance(sticker.get("emoji"), str) and sticker["emoji"].strip():
        return sticker["emoji"]

    return ""


def _slack_content(event: Any) -> str:
    """Slack ``event.text``, or a file-only message's ``files[].title`` / ``.name``.

    A Slack file-share carries the operator-visible directive in the uploaded
    file's (attacker-chosen) title or name, with an empty ``text`` — so keying on
    ``text`` alone lets a file upload smuggle a platform keyword past the scorer.
    """
    if not isinstance(event, dict):
        return ""
    txt = event.get("text")
    if isinstance(txt, str) and txt.strip():
        return txt
    files = event.get("files")
    if isinstance(files, list):
        for f in files:
            if isinstance(f, dict):
                for fk in ("title", "name"):
                    val = f.get(fk)
                    if isinstance(val, str) and val.strip():
                        return val
    return ""


def _whatsapp_content(body: Dict[str, Any]) -> str:
    """Meta-WhatsApp Cloud API ``entry[].changes[].value.messages[0]`` text.

    Reads ``text.body`` AND a media message's ``image``/``video``/``document``/
    ``audio`` ``caption`` and a document ``filename`` — a WhatsApp media message
    carries no ``text.body`` at all, so keying on it alone lets every media
    caption / filename fall through to the router as ``json.dumps(body)``.
    """
    entries = body.get("entry")
    if not (isinstance(entries, list) and entries and isinstance(entries[0], dict)):
        return ""
    changes = entries[0].get("changes", [])
    if not (changes and isinstance(changes[0], dict)):
        return ""
    value = changes[0].get("value", {})
    messages = value.get("messages", []) if isinstance(value, dict) else []
    if not (messages and isinstance(messages[0], dict)):
        return ""
    msg = messages[0]
    text_obj = msg.get("text")
    if isinstance(text_obj, dict) and isinstance(text_obj.get("body"), str) and text_obj["body"].strip():
        return text_obj["body"]
    for mkey in ("image", "video", "document", "audio"):
        media = msg.get(mkey)
        if isinstance(media, dict):
            cap = media.get("caption")
            if isinstance(cap, str) and cap.strip():
                return cap
            fname = media.get("filename")
            if isinstance(fname, str) and fname.strip():
                return fname
    return ""


def extract_inbound_text(body: Dict[str, Any]) -> str:
    """THE operator-visible message text a webhook body carries, or ``""``.

    Single source of truth for "what text does this inbound update carry",
    shared by :meth:`WebhookIngestor.ingest` (the content the router acts on)
    and the ``api/webhooks.py`` trust-gate scorer (``_inbound_text``). Because
    both read from THIS one function, the gate can never score empty a message
    the router would turn into routable content — the two per-field allowlists
    that P225-RVW-2 / P225-RVW-9 / P225-RVW-16 kept re-diverging are now ONE.

    Recognises, across platforms:
      - top-level string fields: ``message`` / ``text`` / ``content`` / ``body``
      - Telegram ``message`` + ``edited_message``: text, caption, and the
        text-bearing subfield of a media / service message (``new_chat_title``,
        a joiner's name, ``file_name``, poll, contact, venue, sticker)
      - Slack ``event.text`` and a file-only message's ``files[].title`` / ``.name``
      - Twilio ``Body``
      - Meta-WhatsApp ``messages[]`` ``text.body`` AND a media ``caption`` /
        document ``filename``

    Returns ``""`` for a genuinely content-less update (a delivery / status
    callback, a bare membership change). The ingestor then serialises the body
    via ``json.dumps`` as a last resort, and the gate leaves it to route — but
    that fallback blob is never handed to AutoBrain's unanchored keyword matcher
    (see the caller guard in ``api/webhooks.py``), so a keyword buried in an
    unrecognised subfield can no longer trigger a platform tool.
    """
    if not isinstance(body, dict):
        return ""

    # 1. Direct string fields (simple webhooks / curl).
    for key in ("message", "text", "content", "body"):
        v = body.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 2. Telegram message / edited_message (text, caption, media/service subfields).
    for mkey in ("message", "edited_message"):
        content = _telegram_content(body.get(mkey))
        if content.strip():
            return content.strip()

    # 3. Slack event.text OR a file-only message's file title / name.
    slack = _slack_content(body.get("event"))
    if slack.strip():
        return slack.strip()

    # 4. Twilio Body.
    twilio = body.get("Body")
    if isinstance(twilio, str) and twilio.strip():
        return twilio.strip()

    # 5. Meta-WhatsApp text.body OR a media caption / document filename.
    wa = _whatsapp_content(body)
    if wa.strip():
        return wa.strip()

    return ""


class WebhookIngestor(BaseIngestor):
    """Transform a generic webhook payload into a RequestEnvelope."""

    def ingest(
        self,
        *,
        body: Dict[str, Any],
        workspace_id: UUID,
    ) -> RequestEnvelope:
        """Build a RequestEnvelope from a webhook POST body.

        Extracts the operator-visible message text via :func:`extract_inbound_text`
        — THE single source of truth the trust gate scores against, so the router
        can never act on content the gate scored empty (P225-RVW-16). Falls back
        to a JSON stringification of the whole body only when no recognised text
        is present; that fallback blob is deliberately kept out of AutoBrain's
        unanchored platform-keyword matcher by the caller (``api/webhooks.py``).
        ``body.source`` / ``body.channel`` -> metadata; ``body.agent_id`` ->
        optional Tier-0 override.
        """
        # --- Extract content text ------------------------------------------------
        # ONE shared extractor across plain fields, Telegram, Slack, WhatsApp,
        # Twilio — the gate and the router read the exact same text.
        content = extract_inbound_text(body)

        # Fallback: stringify the full body (no recognised message text).
        if not content:
            content = json.dumps(body, default=str)

        content = str(content)

        # --- Optional Tier-0 override -------------------------------------------
        override_agent_id: Optional[int] = None
        raw_agent_id = body.get("agent_id")
        if raw_agent_id is not None:
            try:
                override_agent_id = int(raw_agent_id)
            except (ValueError, TypeError):
                pass

        # --- Metadata for routing rules ------------------------------------------
        metadata: Dict[str, Any] = {}
        for key in ("source", "channel", "event_type", "service"):
            if key in body:
                metadata[key] = body[key]

        # --- Build envelope ------------------------------------------------------
        return RequestEnvelope(
            source=ChannelSource.WEBHOOK,
            content=content,
            raw_payload=body,
            user=RequestUser(auth_type="webhook"),
            workspace_id=workspace_id,
            metadata=metadata,
            override_agent_id=override_agent_id,
        )
