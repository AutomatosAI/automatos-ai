"""How an attachment is referenced in prompt text before it has been resolved.

PRD-223 S0.4.

A ``file`` content part gets flattened into plain prompt text at two points
that both run **before** ``AttachmentResolver`` has looked the attachment up:

- ``consumers.chatbot.prompt_analyzer._extract_message_content`` (ATOM + full)
- ``modules.context.sections.conversation._parts_to_text`` (history)

Neither of them knows whether the content arrived, so neither may state a
verdict. Both used to hardcode ``— content not available``, which then sat in
the prompt directly above the *successfully* resolved document text::

    [Attached file: PRD-223.md — content not available]   <- written first
    ### PRD-223.md
    <23KB of real document text>                          <- appended second

The model read the stale claim and reported it as truth (2026-07-31 incident:
Auto quoted the marker back to the user while holding the full document).

The resolver is the single writer of attachment availability — it emits
``### <filename>`` plus the text on success, and ``build_unavailable_marker``
on failure. These renderers therefore only speak for parts the resolver will
*not* touch on this turn: attachments from earlier turns, and parts that never
carried an ``attachment_id`` at all.

Lives in ``core`` rather than ``modules.attachments`` so the context package
can use it without opening a new feature-to-feature import edge — see the
``feature-module-independence`` contract in ``.importlinter``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

# Suffix marking a file whose content is NOT in the current prompt. Only ever
# applied to parts the resolver is not resolving on this turn.
UNAVAILABLE_SUFFIX = "content not available"


def render_unresolved_file_part(
    part: dict[str, Any],
    resolved_attachment_ids: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Render a ``file`` part as prompt text, or ``None`` to emit nothing.

    Args:
        part: The ``{"type": "file", ...}`` content part.
        resolved_attachment_ids: Attachment ids being resolved into *this*
            prompt. A part in this set is owned end-to-end by
            ``AttachmentResolver``; anything written here would either
            duplicate its ``### <filename>`` header or contradict it.

    Returns:
        ``None`` when the resolver owns the part, otherwise an explicit
        unavailable marker. Silence is not an option for unowned parts — a
        bare filename with no content and no marker is what lets a model
        infer the contents from the name.
    """
    attachment_id = part.get("attachment_id")
    if attachment_id and attachment_id in _as_id_set(resolved_attachment_ids):
        return None

    filename = part.get("filename") or part.get("name") or "file"
    return f"[Attached file: {filename} — {UNAVAILABLE_SUFFIX}]"


def _as_id_set(ids: Optional[Iterable[str]]) -> frozenset[str]:
    """Normalise an id collection to a set of strings (ids may be UUIDs)."""
    if not ids:
        return frozenset()
    return frozenset(str(i) for i in ids)
