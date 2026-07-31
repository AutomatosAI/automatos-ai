"""
AttachmentResolver — Convert attachment_ids to LLM content parts (PRD-127)

This is THE ONLY code path that constructs image_url parts or calls extract_text.
All other code (chat handler, planner, task executor, channel adapters) passes
attachment_ids into ContextService.build_context(), which delegates here.

CI enforcement: `rg "image_url.*url|ContentPart" orchestrator/ | grep -v resolver.py`
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Optional
from uuid import UUID

from modules.attachments.store import (
    AttachmentNotFoundError,
    AttachmentStore,
    MediaType,
    get_attachment_store,
)
from modules.attachments.extract import extract_text, is_extraction_failure

logger = logging.getLogger(__name__)

# Default text budget: ~20k tokens * 4 chars/token = 80k chars
DEFAULT_TEXT_BUDGET_CHARS = 80_000

# Max images per request (prevent context overflow)
MAX_IMAGES_PER_REQUEST = 20

# Threshold for inline base64 vs signed URL (500KB)
INLINE_BASE64_THRESHOLD = 500 * 1024


class VisionNotSupportedError(Exception):
    """Raised when images are attached to a text-only model."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(
            f"Model '{model_id}' does not support vision. "
            "Please switch to a vision-capable model to process images."
        )


class AttachmentResolver:
    """
    The single conversion point from attachment_ids to LLM content parts.

    No other code in the codebase should construct image_url parts or call
    extract_text directly — all paths flow through this resolver.
    """

    def __init__(
        self,
        store: Optional[AttachmentStore] = None,
        db_session: Any = None,
    ):
        self._store = store or get_attachment_store()
        self._db_session = db_session

    async def resolve(
        self,
        attachment_ids: list[UUID],
        workspace_id: UUID,
        model_id: str,
        text_budget_tokens: int = 20_000,
    ) -> tuple[list[dict], list[dict]]:
        """
        Convert attachment_ids into LLM content parts.

        Args:
            attachment_ids: List of attachment UUIDs to resolve
            workspace_id: Owning workspace (for isolation check)
            model_id: Target LLM model (for vision capability check)
            text_budget_tokens: Token budget for document text extraction

        Returns:
            (parts, failures):
            - parts: content part dicts ready to append to a user message —
              images as ``image_url`` parts, documents as text parts. When any
              attachment could not be loaded, the LAST part is an explicit
              ``[ATTACHMENT UNAVAILABLE]`` text marker (PRD-223 S0.3) so the
              model knows — and says — what it cannot see, instead of
              inferring content from a filename.
            - failures: ``[{"attachment_id", "filename", "reason"}]`` for the
              caller to surface to the user (SSE badge / log).

        Raises:
            VisionNotSupportedError: If images attached to non-vision model
        """
        if not attachment_ids:
            return [], []

        # Fetch attachment metadata. A missing/expired attachment is a
        # FAILURE the model and user must both learn about — never a silent
        # skip (PRD-223 S0.3: silence here was the fabrication opportunity).
        failures: list[dict] = []
        refs = []
        for aid in attachment_ids:
            try:
                ref = await self._store.get(aid, workspace_id)
                refs.append(ref)
            except AttachmentNotFoundError:
                logger.warning("Attachment %s not found or expired", aid)
                failures.append({
                    "attachment_id": str(aid),
                    "filename": None,
                    "reason": "not found or expired",
                })

        if not refs and not failures:
            return [], []

        # Vision capability check
        has_images = any(r.media_type == MediaType.IMAGE for r in refs)
        if has_images:
            supports_vision = await self._check_vision_support(model_id)
            if not supports_vision:
                raise VisionNotSupportedError(model_id)

        # Enforce image count limit
        image_refs = [r for r in refs if r.media_type == MediaType.IMAGE]
        if len(image_refs) > MAX_IMAGES_PER_REQUEST:
            logger.warning(
                "Too many images (%d > %d) — truncating",
                len(image_refs),
                MAX_IMAGES_PER_REQUEST,
            )
            for dropped in image_refs[MAX_IMAGES_PER_REQUEST:]:
                failures.append({
                    "attachment_id": str(dropped.attachment_id),
                    "filename": dropped.filename,
                    "reason": f"over the {MAX_IMAGES_PER_REQUEST}-image limit",
                })
            # Keep first N images, all documents
            refs = image_refs[:MAX_IMAGES_PER_REQUEST] + [
                r for r in refs if r.media_type != MediaType.IMAGE
            ]

        # Build content parts. A part that fails to build is a failure the
        # model must be told about — not a silent drop.
        parts: list[dict] = []
        budget_chars = text_budget_tokens * 4  # ~4 chars/token estimate

        for ref in refs:
            if ref.media_type == MediaType.IMAGE:
                part = await self._resolve_image(ref, workspace_id)
                if part:
                    parts.append(part)
                else:
                    failures.append({
                        "attachment_id": str(ref.attachment_id),
                        "filename": ref.filename,
                        "reason": "image could not be loaded",
                    })
            else:
                part, chars_used = await self._resolve_document(
                    ref, workspace_id, budget_chars
                )
                if part:
                    parts.append(part)
                    budget_chars -= chars_used
                else:
                    failures.append({
                        "attachment_id": str(ref.attachment_id),
                        "filename": ref.filename,
                        "reason": "text could not be extracted",
                    })

        if failures:
            parts.append(build_unavailable_marker(failures))

        return parts, failures

    async def _check_vision_support(self, model_id: str) -> bool:
        """
        Check if the model supports vision.

        Uses model_registry.get_model() if db_session available,
        otherwise falls back to OpenRouter cache or conservative default.
        """
        if not model_id:
            return False

        # Known vision models (fast path)
        vision_prefixes = (
            "gpt-4o",
            "gpt-4-vision",
            "gpt-4-turbo",
            "gpt-5",
            "claude-3",
            "claude-sonnet-4",
            "claude-opus-4",
            "claude-haiku-4",
            "gemini",
            "llava",
            "qwen-vl",
            "qwen2-vl",
        )
        model_lower = model_id.lower()
        if any(model_lower.startswith(p) or f"/{p}" in model_lower for p in vision_prefixes):
            return True

        # Known text-only models (fast path)
        text_only_prefixes = (
            "gpt-3.5",
            "claude-2",
            "claude-instant",
            "deepseek",
            "mistral",
            "mixtral",
            "llama",
            "qwen/qwen",  # base qwen without -vl
        )
        if any(model_lower.startswith(p) or f"/{p}" in model_lower for p in text_only_prefixes):
            return False

        # Try model registry if db session available
        if self._db_session:
            try:
                from core.llm.model_registry import get_model_registry

                registry = get_model_registry(self._db_session)
                model_info = registry.get_model(model_id)
                if model_info:
                    return model_info.supports_vision
            except Exception as e:
                logger.warning("Model registry lookup failed: %s", e)

        # Try OpenRouter cache
        try:
            from core.models.openrouter_cache import OpenRouterModelCache

            if self._db_session:
                cached = (
                    self._db_session.query(OpenRouterModelCache)
                    .filter(OpenRouterModelCache.model_id == model_id)
                    .first()
                )
                if cached:
                    return cached.supports_vision or False
        except Exception as e:
            logger.debug("OpenRouter cache lookup failed: %s", e)

        # Conservative default: assume no vision to avoid errors
        logger.warning(
            "Unknown model '%s' — assuming no vision support", model_id
        )
        return False

    async def _resolve_image(
        self, ref: "AttachmentRef", workspace_id: UUID
    ) -> Optional[dict]:
        """
        Resolve an image attachment to an image_url content part.

        Uses signed URL for large images, inline base64 for small ones.
        """
        try:
            # Check if we should use inline base64 (< 500KB)
            if ref.size_bytes < INLINE_BASE64_THRESHOLD:
                content = await self._store.open(ref.attachment_id, workspace_id)
                b64 = base64.b64encode(content).decode("ascii")
                data_url = f"data:{ref.mime};base64,{b64}"
                return {"type": "image_url", "image_url": {"url": data_url}}
            else:
                # Use signed URL for larger images
                url = await self._store.sign_url(ref.attachment_id, workspace_id)
                return {"type": "image_url", "image_url": {"url": url}}
        except Exception as e:
            logger.error("Failed to resolve image %s: %s", ref.attachment_id, e)
            return None

    async def _resolve_document(
        self, ref: "AttachmentRef", workspace_id: UUID, budget_chars: int
    ) -> tuple[Optional[dict], int]:
        """
        Resolve a document attachment to a text content part.

        Returns (content_part, chars_used) tuple.
        """
        if budget_chars <= 0:
            logger.warning(
                "Text budget exhausted — skipping document %s", ref.filename
            )
            return None, 0

        try:
            content = await self._store.open(ref.attachment_id, workspace_id)
            text = extract_text(
                content, ref.mime, ref.filename, max_chars=budget_chars
            )

            if not text or is_extraction_failure(text):
                # Extraction failed or empty — the caller records this as a
                # failure and the model is told via build_unavailable_marker.
                return None, 0

            formatted = f"### {ref.filename}\n\n{text}"
            return {"type": "text", "text": formatted}, len(formatted)
        except Exception as e:
            logger.error(
                "Failed to resolve document %s: %s", ref.attachment_id, e
            )
            return None, 0


# Type hint forward reference
from modules.attachments.store import AttachmentRef  # noqa: E402


def build_unavailable_marker(failures: list[dict]) -> dict:
    """Text part telling the model exactly which attachments it does NOT have.

    PRD-223 S0.3: a message that references a file whose content never
    arrived reads, to the model, like a file it should know — the engineered
    fabrication opportunity from the 2026-07-31 incident. State the gap
    explicitly and instruct honesty.
    """
    lines = []
    for failure in failures:
        name = failure.get("filename") or f"attachment {failure.get('attachment_id', 'unknown')}"
        lines.append(f"- {name}: {failure.get('reason', 'could not be loaded')}")
    return {
        "type": "text",
        "text": (
            "[ATTACHMENT UNAVAILABLE]\n"
            "The following attached file(s) could NOT be loaded into this conversation:\n"
            + "\n".join(lines)
            + "\nYou do NOT have the contents of these files. Do not infer their "
            "contents from filenames, and do not claim to have read them. If the "
            "user asks about them, say you can see the reference but do not have "
            "the contents."
        ),
    }


def inject_parts_into_last_user_message(
    messages: list[dict], parts: list[dict]
) -> list[dict]:
    """
    Inject content parts into the last user message.

    Converts `content: str` to `content: [{type: "text", text: ...}, *parts]`.

    Args:
        messages: List of message dicts (modified in place)
        parts: Content parts to inject

    Returns:
        The modified messages list
    """
    if not parts or not messages:
        return messages

    # Find last user message (search backwards)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            msg = messages[i]
            content = msg.get("content")

            if isinstance(content, str):
                # Convert string to parts list
                msg["content"] = [{"type": "text", "text": content}] + parts
            elif isinstance(content, list):
                # Append to existing parts list
                msg["content"] = content + parts
            else:
                # No content — set parts directly
                msg["content"] = parts

            return messages

    # No user message found — append as new message
    messages.append({"role": "user", "content": parts})
    return messages
