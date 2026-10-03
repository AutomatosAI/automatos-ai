"""PRD-251B Wave 3 (B9; US-B303): the style profile Auto reads from the style references.

A vision model reads the references (liked ones first, newest first, then the avoided ones,
at most MAX_READ_IMAGES, each shrunk to READ_EDGE_PX for the read) into a profile: a
palette (hex colours), a mood (a few words), the composition (a sentence) and what to avoid
(a sentence). The read goes through the platform's LLM manager (``create_llm_manager``,
request type ``brand_style_read``, so its usage is tracked for the workspace), with
``BRAND_STYLE_READ_MODEL`` (empty: the workspace's own model) and
``BRAND_STYLE_READ_TIMEOUT_SECONDS`` from config.

It is read again whenever the references change (:func:`launch_refresh`, in the
background) and on request ("Read the references again"). :func:`style_prompt` is the
profile as one paragraph: the composer, the plan's research and Template Studio's image
prompts carry it. Liked images themselves go to an AI tool only when its action accepts a
reference image and the workspace allows it (``send_liked``).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import UUID

from config import config
from modules.documents import brand_references as refs

logger = logging.getLogger(__name__)

SERVICE_NAME = "brand_style"
REQUEST_TYPE = "brand_style_read"
MAX_READ_IMAGES = 8
READ_EDGE_PX = 768
# Without Pillow to shrink it, an image goes to the model only when it is this small.
SMALL_ENOUGH_BYTES = 1_500_000
MAX_PALETTE = 6
MAX_MOOD_WORDS = 8
MAX_WORD_CHARS = 24
MAX_SENTENCE_CHARS = 400
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_FENCED = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)  # the composer's own

SYSTEM = (
    "You read a brand's style references for its social media team. Each image is marked LIKE "
    "(the brand wants posts to look like it) or AVOID (posts must never look like it), with the "
    "person's note. Answer with one JSON object and nothing else: "
    '{"palette": [up to 6 hex colours the liked images share, most used first], '
    '"mood": [up to 8 short words for their mood], "composition": "one sentence on how the liked '
    'images are composed", "avoid": "one sentence on what the avoided images have that posts must not"}. '
    "Describe only what the images show; never name a person or a brand you recognise."
)


class StyleReadFailed(Exception):
    """The model's answer could not be read as a profile (502); a timeout is its own (504)."""


def _shrunk(data: bytes, content_type: str) -> Optional[Tuple[bytes, str]]:
    """The image as a small JPEG for the read, or as it is when Pillow is not there and it is small."""
    try:
        from PIL import Image  # optional: without it the read takes small images as they are
    except ImportError:
        return (data, content_type) if len(data) <= SMALL_ENOUGH_BYTES else None
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail((READ_EDGE_PX, READ_EDGE_PX))
        out = io.BytesIO()
        image.save(out, "JPEG", quality=80)
        return out.getvalue(), "image/jpeg"
    except (OSError, ValueError):
        logger.warning("[BrandStyle] a style reference could not be shrunk for the read; it is left out", exc_info=True)
        return None


def images_for_read(references: List[Dict[str, Any]]) -> List[Tuple[str, str, bytes, str]]:
    """(stance, note, bytes, type) of the references the read sends: liked first, newest first."""
    liked, avoided = refs.split_by_stance(references)
    ordered = [*reversed(liked), *reversed(avoided)]
    images: List[Tuple[str, str, bytes, str]] = []
    for ref in ordered:
        if len(images) >= MAX_READ_IMAGES:
            break
        data = refs.load_reference(ref)
        small = _shrunk(data, ref.get("content_type") or "image/png") if data else None
        if small is not None:
            images.append((ref.get("stance") or refs.LIKE, ref.get("note") or "", small[0], small[1]))
    return images


def build_messages(images: List[Tuple[str, str, bytes, str]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": "The brand's style references:"}]
    for index, (stance, note, data, content_type) in enumerate(images, start=1):
        content.append({"type": "text", "text": f"Image {index}: {stance.upper()}. Note: {note or 'none'}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"}})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def _words(value: Any, limit: int) -> List[str]:
    items = value if isinstance(value, list) else []
    return [" ".join(str(item).split())[:MAX_WORD_CHARS] for item in items if str(item).strip()][:limit]


def parse_profile(text: Any) -> Optional[Dict[str, Any]]:
    """The model's answer as a profile, checked and trimmed; ``None`` when it is not one."""
    if not isinstance(text, str) or not text.strip():
        return None
    fenced = _FENCED.search(text)
    body = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        raw = json.loads(body)
    except ValueError:
        return None
    if not isinstance(raw, dict):
        return None
    palette = [colour.upper() for colour in _words(raw.get("palette"), MAX_PALETTE) if HEX.match(colour)]
    profile = {
        "palette": palette,
        "mood": _words(raw.get("mood"), MAX_MOOD_WORDS),
        "composition": " ".join(str(raw.get("composition") or "").split())[:MAX_SENTENCE_CHARS],
        "avoid": " ".join(str(raw.get("avoid") or "").split())[:MAX_SENTENCE_CHARS],
    }
    return profile if palette or profile["mood"] or profile["composition"] else None


async def read_profile(workspace_id: Any, images: List[Tuple[str, str, bytes, str]]) -> Dict[str, Any]:
    """The profile the model reads from ``images``; StyleReadFailed, or asyncio.TimeoutError."""
    from core.llm import create_llm_manager

    llm = create_llm_manager(
        service_name=SERVICE_NAME, model=config.BRAND_STYLE_READ_MODEL or None, workspace_id=workspace_id, request_type=REQUEST_TYPE,
    )
    response = await asyncio.wait_for(llm.generate_response(build_messages(images)), timeout=float(config.BRAND_STYLE_READ_TIMEOUT_SECONDS))
    profile = parse_profile(getattr(response, "content", None))
    if profile is None:
        raise StyleReadFailed("The model's answer could not be read as a style profile. Try again.")
    return profile


def with_profile(style: Mapping[str, Any], profile: Optional[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The style settings with ``profile`` (stamped with when it was read and from which references)."""
    stamped = None if profile is None else {
        **profile, "read_at": datetime.now(timezone.utc).isoformat(), "reference_ids": [ref["id"] for ref in references],
    }
    return {**style, "profile": stamped}


def style_prompt(settings: Optional[Mapping[str, Any]]) -> str:
    """The profile as one paragraph for an image prompt; empty when there is none."""
    profile = refs.style_of(settings)["profile"] or {}
    parts = []
    if profile.get("palette"):
        parts.append(f"palette {', '.join(profile['palette'])}")
    if profile.get("mood"):
        parts.append(f"mood: {', '.join(profile['mood'])}")
    if profile.get("composition"):
        parts.append(f"composition: {profile['composition'].rstrip('.')}")
    if profile.get("avoid"):
        parts.append(f"avoid: {profile['avoid'].rstrip('.')}")
    return f"Brand style (from the brand kit's references): {'; '.join(parts)}." if parts else ""


def _load_references(workspace_id: UUID) -> Optional[List[Dict[str, Any]]]:
    from core.database.database import SessionLocal
    from core.models.workspaces import Workspace

    with SessionLocal() as db:
        workspace = db.get(Workspace, workspace_id)
        return None if workspace is None else refs.style_of(workspace.settings)["references"]


def same_references(current: List[Dict[str, Any]], read: List[Dict[str, Any]]) -> bool:
    """Whether the references are still the ones a read covered (each id, stance and note)."""
    def shape(found: List[Dict[str, Any]]) -> List[Tuple[Any, Any, Any]]:
        return [(ref.get("id"), ref.get("stance"), ref.get("note")) for ref in found]

    return shape(current) == shape(read)


def _store_profile(workspace_id: UUID, profile: Optional[Dict[str, Any]], references: List[Dict[str, Any]]) -> bool:
    """Store the profile read from ``references``, the workspace row locked; nothing when the
    references changed meanwhile (the change launched a read of its own, which stores)."""
    from core.database.database import SessionLocal
    from core.models.workspaces import Workspace

    with SessionLocal() as db:
        workspace = db.query(Workspace).filter(Workspace.id == workspace_id).with_for_update().first()
        style = refs.style_of(workspace.settings) if workspace is not None else None
        if style is None or not same_references(style["references"], references):
            db.rollback()
            return False
        refs.save_style(db, workspace, with_profile(style, profile, references))
        return True


async def refresh_profile(workspace_id: UUID) -> Optional[Dict[str, Any]]:
    """Read the workspace's references again and store the profile (no references: no profile).
    The database work runs off the event loop, on sessions of its own."""
    references = await asyncio.to_thread(_load_references, workspace_id)
    if references is None:
        return None
    images = await asyncio.to_thread(images_for_read, references) if references else []
    profile = await read_profile(workspace_id, images) if images else None
    await asyncio.to_thread(_store_profile, workspace_id, profile, references)
    return profile


async def _refresh_quietly(workspace_id: UUID) -> None:
    try:
        await refresh_profile(workspace_id)
    except (StyleReadFailed, asyncio.TimeoutError):
        logger.warning("[BrandStyle] the style profile of workspace %s was not read again", workspace_id, exc_info=True)


def launch_refresh(workspace_id: UUID) -> None:
    """Read the profile again in the background (the references changed). Call it from the event loop."""
    from core.utils.background_tasks import launch_guarded

    launch_guarded(_refresh_quietly(workspace_id), subsystem="brand_kit", operation="style_read", workspace_id=workspace_id)
