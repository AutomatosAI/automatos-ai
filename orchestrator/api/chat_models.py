"""The ``POST /api/chat`` request body (PRD-27): ``ChatRequest`` and its parts.

PRD-266 US-007a moved these out of ``api/chat.py``, which is over AGENTS.md's 800-line limit,
so that file shrinks instead of growing; ``api/chat.py`` imports them from here.

``spokenMode`` (PRD-266 §4.7, D32): the reply is heard, not read (the app's quick-ask voice).
Auto is told so (``modules.voice.spoken_style.SPOKEN_OUTPUT_CONTRACT``) and the turn's text
frames are speechified (``modules.voice.spoken_stream``). False, the default, changes nothing.

``locale``: the person's language and region as a BCP 47 tag (``pt-BR``: Automatos Go's "I speak"
setting). Auto is told to reply in that variant (``consumers.chatbot.reply_locale``), on a spoken
turn and a typed one alike; without it the model picked the variant itself, and from Lisbon a
Brazilian speaker got European Portuguese. A value that is not a tag is dropped, never a 422, so
no client's turn fails over it; left out, the turn is exactly as before.
"""

import re
from typing import Any, List, Optional

from pydantic import BaseModel, field_validator

# A language, then up to two subtags (``pt-BR``, ``zh-Hant-TW``); RFC 5646 §4.4.1 recommends
# at most 35 characters for a tag.
LOCALE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8}){0,2}$")
LOCALE_MAX_LENGTH = 35
REGION_LETTERS = 2
SCRIPT_LETTERS = 4


def _conventional_case(subtag: str) -> str:
    """BCP 47's conventional case for a subtag: a region upper (``BR``), a script title (``Hant``)."""
    if subtag.isalpha() and len(subtag) == REGION_LETTERS:
        return subtag.upper()
    if subtag.isalpha() and len(subtag) == SCRIPT_LETTERS:
        return subtag.title()
    return subtag.lower()


def clean_locale(value: Any) -> Optional[str]:
    """``value`` as a BCP 47 tag in its conventional case (``pt-br`` becomes ``pt-BR``), or
    ``None`` when it is not one (``pt_BR``, a sentence, a number, too long)."""
    if not isinstance(value, str):
        return None
    tag = value.strip()
    if len(tag) > LOCALE_MAX_LENGTH or not LOCALE_PATTERN.fullmatch(tag):
        return None
    language, *subtags = tag.split("-")
    return "-".join([language.lower(), *(_conventional_case(subtag) for subtag in subtags)])


class MessagePart(BaseModel):
    """One part of a chat message: text, a file, or an ephemeral attachment reference."""

    type: str
    text: Optional[str] = None
    filename: Optional[str] = None
    mediaType: Optional[str] = None
    url: Optional[str] = None
    # PRD-127: ephemeral attachment reference (sent by multimodal-input.tsx)
    attachment_id: Optional[str] = None


class ChatMessageRequest(BaseModel):
    """The message being sent: AI SDK ``parts``, or a plain ``content`` string."""

    role: str = "user"
    parts: Optional[List[MessagePart]] = None
    # Compatibility with older/alternate clients
    content: Optional[str] = None
    # PRD-127: top-level list of ephemeral attachment ids for the current message
    attachment_ids: Optional[List[str]] = None


class ChatRequest(BaseModel):
    """The body of ``POST /api/chat``."""

    id: Optional[str] = None
    message: ChatMessageRequest
    # Compatibility with AI SDK "messages" payloads
    messages: Optional[List[ChatMessageRequest]] = None
    # PRD-180 S3 (F035): the per-message model selector was a placebo — nothing
    # ever read the chosen model. Field removed; the model resolves via the Auto
    # tier / the selected agent's own config, never a client-picked override.
    selectedVisibilityType: Optional[str] = "private"
    context: Optional[dict] = None
    # PRD: Unified Agent-Chat System
    agentId: Optional[int] = None  # Selected agent ID (default: system agent id=1)
    # PRD-82A: Mission mode — conversational mission planning
    missionMode: Optional[bool] = False
    # Plan mode — research and strategy output, no execution
    planMode: Optional[bool] = False
    # PRD-266 US-007a (D32): the reply is spoken aloud — short sentences, no markdown
    spokenMode: bool = False
    # The person's language and region (BCP 47, ``pt-BR``): the variant Auto replies in.
    locale: Optional[str] = None

    @field_validator("locale", mode="before")
    @classmethod
    def _not_a_tag_is_dropped(cls, value: Any) -> Optional[str]:
        return clean_locale(value)


__all__ = ["ChatMessageRequest", "ChatRequest", "MessagePart", "clean_locale"]
