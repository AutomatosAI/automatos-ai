"""PRD-266 US-011a — what a push says, and how a phone knows the server sent it.

Pure: no DB, no HTTP, no config. The fan-out (``fanout.py``) loads the grant, the ticket's
number and the workspace's settings, and calls this module once per phone.

* ``PUSH_TYPE_EVENTS`` — the only events that ever push (FR-7, K8). Every decision to push
  reads it; any other event stays in the bell.
* ``build_payload`` — K9's v2 payload: ids as strings, absent keys omitted (never null), the
  build time as ``iat`` and ``sig``. A question carries its text and up to three options so
  it can be answered from the lock screen; redaction (``redact_questions_enabled``) and every
  other event carry a generic line instead.
* ``sign`` / ``verify_signature`` — HMAC-SHA256 with the device's ``push_key`` over
  ``canonical_message``: anyone holding an Expo token can send a push, so the phone acts on a
  notification only when the signature holds and ``iat`` is fresh. automatos-go copies
  ``tests/fixtures/prd266_push_signature_vectors.json`` byte for byte and checks the same.
* ``to_message`` — the Expo message: the PRD-264 AI badge in every title (FR-9, D17), the
  question or the generic line as the body, the iOS category and the Android channel.

A push key is read only to sign; nothing here logs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Optional

from modules.policy.ai_disclosure import badge_for
from services.push.sender import PRIORITY_DEFAULT, PRIORITY_HIGH, PushMessage

# ── Which events push (FR-7, K8) ───────────────────────────────────────────
NEEDS_ANSWER = "needs_answer"
QUESTION_EVENT = "question_pending"
REVIEW_EVENT = "review_pending"

PUSH_TYPE_EVENTS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "needs_answer": ("question_pending", "approval_pending", "mission_plan_ready"),
    "review": ("review_pending",),
    "auto_message": ("auto_message",),
})
PUSH_EVENT_TYPES: frozenset[str] = frozenset(e for events in PUSH_TYPE_EVENTS.values() for e in events)
_TYPE_BY_EVENT: Mapping[str, str] = MappingProxyType(
    {event: push_type for push_type, events in PUSH_TYPE_EVENTS.items() for event in events}
)

# ── What a push says ───────────────────────────────────────────────────────
QUESTION_LINE = "A question needs your answer"
APPROVAL_LINE = "An approval is waiting for you"
MISSION_PLAN_LINE = "A Mission plan is ready for you"
REVIEW_LINE = "A ticket is ready for review"
REVIEW_NUMBERED_LINE = "A ticket is ready for review · {number}"
AUTO_MESSAGE_LINE = "Auto messaged you"
GENERIC_LINES: Mapping[str, str] = MappingProxyType({
    "question_pending": QUESTION_LINE,
    "approval_pending": APPROVAL_LINE,
    "mission_plan_ready": MISSION_PLAN_LINE,
    "review_pending": REVIEW_LINE,
    "auto_message": AUTO_MESSAGE_LINE,
})

PUSH_QUESTION_MAX_CHARS = 500
PUSH_OPTION_MAX_CHARS = 80
PUSH_MAX_OPTIONS = 3
ELLIPSIS = "…"
# How much of a question's markdown is read: bounds the regex work on a long question.
QUESTION_SCAN_MAX_CHARS = 8 * PUSH_QUESTION_MAX_CHARS

# Workspace settings: {"push": {"redact_questions": true}} sends questions generic (US-014b).
PUSH_SETTINGS_KEY = "push"
REDACT_QUESTIONS_KEY = "redact_questions"

# ── The Expo message (K9) ──────────────────────────────────────────────────
SENDER_AUTO = "Auto"
TITLE_FORMAT = "{sender} · {badge}"
QUESTION_CATEGORY_FORMAT = "automatos.question.{count}"
OPEN_CATEGORY = "automatos.open"
CHANNEL_BY_TYPE: Mapping[str, str] = MappingProxyType({
    "needs_answer": "questions",
    "review": "review",
    "auto_message": "auto",
})

# ── The signature (K9, v2) ─────────────────────────────────────────────────
PAYLOAD_VERSION = 2
SIGNATURE_VERSION = "v2"
PUSH_SIG_MAX_AGE_SECONDS = 604800  # 7 days
PUSH_SIG_MAX_FUTURE_SECONDS = 300
# The signed string fields, in K9's order; options and iat follow them.
SIGNED_TEXT_FIELDS = (
    "type", "workspace_id", "notification_id", "grant_id", "link_type", "link_id",
    "ticket_number", "question",
)
NO_PUSH_KEY = "A device without a push key cannot be sent a signed push."
NOT_A_PUSH_EVENT = "{event_type!r} never pushes (FR-7)."

# ── Markdown to plain text: the marks a question's markdown may hold ───────
# Every pattern runs in linear time: a label stops at the next '[', a link target allows one
# level of parentheses, and emphasis never spans its own delimiter. "__" is not bold here, so
# a name like __init__ keeps its underscores; "_" between word characters is never a mark.
_LINK_TARGET = r"\((?:[^()\n]|\([^()\n]*\))*\)"
_MARKDOWN_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^[ \t]*(?:```|~~~)[^\n]*$", re.M), ""),           # a fence line; its code stays
    (re.compile(r"!\[([^\[\]\n]*)\]" + _LINK_TARGET), r"\1"),         # an image: its alt text
    (re.compile(r"\[([^\[\]\n]*)\]" + _LINK_TARGET), r"\1"),          # a link: its label
    (re.compile(r"`+([^`\n]*)`+"), r"\1"),                           # inline code: its text
    (re.compile(r"^[ \t]*(?:#{1,6}[ \t]+|>[ \t]?|[-*+][ \t]+)", re.M), ""),  # heading, quote, bullet
    (re.compile(r"(?<![\w*])([*_])(?=[^\s*_])([^*_\n]*?[^\s*_])\1(?![\w*])"), r"\2"),  # italics
    (re.compile(r"\*\*(?=\S)([^*\n]*?\S)\*\*"), r"\1"),               # bold
    (re.compile(r"~~(?=\S)([^~\n]*?\S)~~"), r"\1"),                   # strikethrough
)
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def push_type_for(event_type: Any) -> Optional[str]:
    """The push type an event raises (needs_answer, review or auto_message); None: it never pushes."""
    return _TYPE_BY_EVENT.get(event_type) if isinstance(event_type, str) else None


def redact_questions_enabled(workspace_settings: Any) -> bool:
    """True when the workspace sends its questions generic: ``settings['push']['redact_questions']``."""
    push = workspace_settings.get(PUSH_SETTINGS_KEY) if isinstance(workspace_settings, Mapping) else None
    return isinstance(push, Mapping) and push.get(REDACT_QUESTIONS_KEY) is True


def plain_text(markdown: str) -> str:
    """A question's markdown as the text a notification shows: marks dropped, lines kept."""
    text = markdown[:QUESTION_SCAN_MAX_CHARS].replace("\r\n", "\n")
    for pattern, replacement in _MARKDOWN_RULES:
        text = pattern.sub(replacement, text)
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _question_text(question_md: Any) -> Optional[str]:
    """The question as plain text, at most PUSH_QUESTION_MAX_CHARS with an ellipsis; None when empty."""
    if not isinstance(question_md, str):
        return None
    text = plain_text(question_md)
    unread = question_md[QUESTION_SCAN_MAX_CHARS:].strip()
    if len(text) > PUSH_QUESTION_MAX_CHARS or unread:
        text = text[: PUSH_QUESTION_MAX_CHARS - len(ELLIPSIS)].rstrip() + ELLIPSIS
    return text or None


def _carried_options(options: Any) -> tuple[str, ...]:
    """The grant's options when a phone can offer them all as they are: 1 to 3, each short text.

    Never truncated: ``/answer`` accepts only the exact string, so a list that does not fit
    is not carried at all and the question is answered in words (category ``.question.0``).
    """
    if not isinstance(options, (list, tuple)) or not 1 <= len(options) <= PUSH_MAX_OPTIONS:
        return ()
    fits = all(
        isinstance(option, str) and option.strip() and len(option) <= PUSH_OPTION_MAX_CHARS
        for option in options
    )
    return tuple(options) if fits else ()


def _question_parts(event_type: str, grant: Any, redact: bool) -> tuple[Optional[str], tuple[str, ...]]:
    """A question's text and options, or nothing: redacted, another event, or no grant."""
    if redact or event_type != QUESTION_EVENT or grant is None:
        return None, ()
    question = _question_text(getattr(grant, "question_md", None))
    if question is None:
        return None, ()
    return question, _carried_options(getattr(grant, "options", None))


def _text(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def build_payload(
    push_key: str,
    *,
    event_type: str,
    workspace_id: Any,
    notification_id: Any = None,
    link_type: Optional[str] = None,
    link_id: Any = None,
    grant: Any = None,
    ticket_number: Optional[str] = None,
    redact: bool = False,
    now: datetime,
) -> dict[str, Any]:
    """K9's signed v2 payload for one phone.

    ``grant`` is the question's (or approval's) ``ApprovalGrant``; ``ticket_number`` is
    ``services.ticket_numbers``' number ("#0419"); ``redact`` is
    ``redact_questions_enabled(workspace.settings)``; ``now`` sets ``iat``. Every id is a
    string; a field with no value is left out, never sent as null. Raises ValueError for an
    event outside PUSH_EVENT_TYPES or a device without a push key.
    """
    push_type = push_type_for(event_type)
    if push_type is None:
        raise ValueError(NOT_A_PUSH_EVENT.format(event_type=event_type))
    question, options = _question_parts(event_type, grant, redact)
    fields = {
        "v": PAYLOAD_VERSION,
        "type": push_type,
        "workspace_id": _text(workspace_id),
        "notification_id": _text(notification_id),
        "grant_id": _text(getattr(grant, "id", None)),
        "link_type": _text(link_type),
        "link_id": _text(link_id),
        "ticket_number": _text(ticket_number),
        "question": question,
        "options": list(options) or None,
        "iat": int(now.timestamp()),
    }
    unsigned = {key: value for key, value in fields.items() if value is not None and value != ""}
    return {**unsigned, "sig": sign(push_key, unsigned)}


def canonical_message(payload: Mapping[str, Any]) -> bytes:
    """The bytes K9 signs: ``['v2', type, workspace_id, notification_id, grant_id, link_type,
    link_id, ticket_number, question, options, iat]`` as compact JSON, UTF-8, not ASCII-escaped
    (byte for byte JavaScript's ``JSON.stringify`` of the same array). A missing string field
    is ``''``, missing options ``[]``."""
    texts = [payload.get(name) if payload.get(name) is not None else "" for name in SIGNED_TEXT_FIELDS]
    options = payload.get("options")
    signed = [SIGNATURE_VERSION, *texts, options if options is not None else [], payload.get("iat")]
    return json.dumps(signed, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign(push_key: str, payload: Mapping[str, Any]) -> str:
    """Lowercase hex HMAC-SHA256 of ``canonical_message(payload)``, keyed by the push key as UTF-8."""
    if not isinstance(push_key, str) or not push_key:
        raise ValueError(NO_PUSH_KEY)
    return hmac.new(push_key.encode("utf-8"), canonical_message(payload), hashlib.sha256).hexdigest()


def _fresh(iat: Any, now: datetime) -> bool:
    if not isinstance(iat, int) or isinstance(iat, bool):
        return False
    now_s = int(now.timestamp())
    return now_s - PUSH_SIG_MAX_AGE_SECONDS <= iat <= now_s + PUSH_SIG_MAX_FUTURE_SECONDS


def verify_signature(push_key: str, payload: Mapping[str, Any], now: datetime) -> bool:
    """True only for a v2 payload whose ``iat`` is in the window and whose ``sig`` matches.

    Never raises: a payload that cannot be put into canonical form (a value JSON cannot hold,
    text that is not valid Unicode) is simply not verified.
    """
    sig = payload.get("sig")
    if payload.get("v") != PAYLOAD_VERSION or not isinstance(sig, str) or not _fresh(payload.get("iat"), now):
        return False
    if not isinstance(push_key, str) or not push_key:
        return False
    try:
        expected = sign(push_key, payload).encode("utf-8")
        given = sig.encode("utf-8")
    except (TypeError, ValueError, RecursionError):  # not JSON, or a lone surrogate: unverifiable
        return False
    return hmac.compare_digest(expected, given)


def generic_line(event_type: str, ticket_number: Optional[str] = None) -> str:
    """The line a push shows when it carries no question: one per push event."""
    if event_type == REVIEW_EVENT and ticket_number:
        return REVIEW_NUMBERED_LINE.format(number=ticket_number)
    return GENERIC_LINES[event_type]


def _category(payload: Mapping[str, Any]) -> str:
    """iOS: a question's category names how many options it offers (0 = Reply and Open)."""
    if not payload.get("question"):
        return OPEN_CATEGORY
    return QUESTION_CATEGORY_FORMAT.format(count=len(payload.get("options") or ()))


def to_message(
    device: Any, payload: Mapping[str, Any], *, event_type: str, agent_name: Optional[str] = None,
) -> PushMessage:
    """The Expo message for one phone: ``device`` holds ``expo_push_token`` and ``locale``.

    Title "<agent or Auto> · <badge>" (FR-9: the PRD-264 D17 badge for the phone's locale);
    the body is the question or the generic line; ``data`` is the signed payload.
    """
    push_type = payload["type"]
    sender = (agent_name or "").strip() or SENDER_AUTO
    return PushMessage(
        to=device.expo_push_token,
        title=TITLE_FORMAT.format(sender=sender, badge=badge_for(getattr(device, "locale", None))),
        body=payload.get("question") or generic_line(event_type, payload.get("ticket_number")),
        data=payload,
        category_id=_category(payload),
        channel_id=CHANNEL_BY_TYPE[push_type],
        priority=PRIORITY_HIGH if push_type == NEEDS_ANSWER else PRIORITY_DEFAULT,
    )
