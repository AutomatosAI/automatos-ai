"""The ``POST /api/chat`` request body (PRD-27): ``ChatRequest`` and its parts.

PRD-266 US-007a moved these out of ``api/chat.py``, which is over AGENTS.md's 800-line limit,
so that file shrinks instead of growing; ``api/chat.py`` imports them from here.

``spokenMode`` (PRD-266 §4.7, D32): the reply is heard, not read (the app's quick-ask voice).
Auto is told so (``modules.voice.spoken_style.SPOKEN_OUTPUT_CONTRACT``) and the turn's text
frames are speechified (``modules.voice.spoken_stream``). False, the default, changes nothing.
"""

from typing import List, Optional

from pydantic import BaseModel


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


__all__ = ["ChatMessageRequest", "ChatRequest", "MessagePart"]
