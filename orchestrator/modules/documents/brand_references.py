"""PRD-251B Wave 3 (B9; US-B302): the brand kit's style references.

Images a person likes, or wants to avoid, each with a note on why: at most
MAX_REFERENCES of them, PNG, JPEG or WebP, each at most MAX_REFERENCE_BYTES. The type is
read from the file's own bytes, never its name or its declared type. They are stored like
the logo (the workspace's brand files, ``brand_logo.store_brand_file``) and listed under a
settings key of their own, ``workspace.settings['brand_style']``, beside the style profile
Auto reads from them (``brand_style.py``): the kit's PUT never touches them. Every lookup is
the workspace's own, so another workspace's reference is "not found".
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import UUID, uuid4

from modules.documents.brand_logo import delete_brand_file, load_brand_file, store_brand_file

REFERENCES_ROUTE = "/api/documents/brand-kit/references"
STYLE_SETTINGS_KEY = "brand_style"
MAX_REFERENCES = 24
MAX_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_NOTE_CHARS = 300
LIKE, AVOID = "like", "avoid"
STANCES = (LIKE, AVOID)
# extension → media type, for the three kinds a reference may be.
IMAGE_TYPES = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}


class BrandReferenceError(ValueError):
    """A reference refused; ``status`` is the HTTP answer (413 too big, 415 not an image, 422 otherwise)."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def sniff(data: bytes) -> Optional[str]:
    """``png``, ``jpg`` or ``webp`` from the file's first bytes; ``None`` for anything else."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def reference_path(workspace_id: UUID, ref_id: str, extension: str) -> str:
    return f"{workspace_id}/brand/references/{ref_id}.{extension}"


def style_of(settings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The workspace's style settings: its references, its profile and whether liked images go to AI tools."""
    raw = (settings or {}).get(STYLE_SETTINGS_KEY) or {}
    refs = [dict(ref) for ref in raw.get("references") or [] if isinstance(ref, Mapping) and ref.get("id")]
    profile = raw.get("profile") if isinstance(raw.get("profile"), Mapping) else None
    return {"references": refs, "profile": dict(profile) if profile else None, "send_liked": raw.get("send_liked") is not False}


def save_style(db: Any, workspace: Any, style: Mapping[str, Any]) -> Dict[str, Any]:
    """Store ``style`` as the workspace's style settings and commit (settings reassigned, so the JSON change is seen)."""
    workspace.settings = {**(workspace.settings or {}), STYLE_SETTINGS_KEY: dict(style)}
    db.commit()
    return dict(style)


def _note(note: Any) -> str:
    text = " ".join(str(note or "").split())
    if len(text) > MAX_NOTE_CHARS:
        raise BrandReferenceError(f"a note has at most {MAX_NOTE_CHARS} characters")
    return text


def _stance(stance: Any) -> str:
    if stance not in STANCES:
        raise BrandReferenceError(f"stance must be one of {', '.join(STANCES)}")
    return stance


def add_reference(workspace_id: UUID, references: List[Dict[str, Any]], data: bytes, *, note: Any, stance: Any) -> List[Dict[str, Any]]:
    """Store an uploaded image and return the new list, its entry last: checked for type, size and count."""
    if len(references) >= MAX_REFERENCES:
        raise BrandReferenceError(f"The brand kit holds {MAX_REFERENCES} style references at most: remove one first.")
    if len(data) > MAX_REFERENCE_BYTES:
        raise BrandReferenceError(f"A style reference is at most {MAX_REFERENCE_BYTES // (1024 * 1024)} MB.", status=413)
    extension = sniff(data)
    if extension is None:
        raise BrandReferenceError("A style reference is a PNG, JPEG or WebP image.", status=415)
    ref_id = uuid4().hex
    entry = {
        "id": ref_id, "path": reference_path(workspace_id, ref_id, extension), "content_type": IMAGE_TYPES[extension],
        "bytes": len(data), "note": _note(note), "stance": _stance(stance), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    store_brand_file(entry["path"], data, entry["content_type"])
    return [*references, entry]


def find_reference(references: List[Dict[str, Any]], ref_id: str) -> Optional[Dict[str, Any]]:
    return next((ref for ref in references if ref.get("id") == ref_id), None)


def update_reference(references: List[Dict[str, Any]], ref_id: str, changes: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The list with the reference's note and stance changed, or ``None`` when there is no such reference."""
    found = find_reference(references, ref_id)
    if found is None:
        return None
    edited = {**found}
    if "note" in changes:
        edited["note"] = _note(changes["note"])
    if "stance" in changes:
        edited["stance"] = _stance(changes["stance"])
    return [edited if ref is found else ref for ref in references]


def remove_reference(references: List[Dict[str, Any]], ref_id: str) -> Optional[List[Dict[str, Any]]]:
    """The list without the reference (its file removed), or ``None`` when there is no such reference."""
    found = find_reference(references, ref_id)
    if found is None:
        return None
    delete_brand_file(found.get("path") or "")
    return [ref for ref in references if ref is not found]


def load_reference(ref: Mapping[str, Any]) -> Optional[bytes]:
    return load_brand_file(ref.get("path") or "", MAX_REFERENCE_BYTES)


def listed(ref: Mapping[str, Any]) -> Dict[str, Any]:
    """A reference as the routes answer it: its image is served by its own route, never by its storage path."""
    return {
        "id": ref["id"], "note": ref.get("note") or "", "stance": ref.get("stance") or LIKE,
        "content_type": ref.get("content_type"), "bytes": ref.get("bytes"), "created_at": ref.get("created_at"),
        "url": f"{REFERENCES_ROUTE}/{ref['id']}/image",
    }


def split_by_stance(references: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    liked = [ref for ref in references if ref.get("stance") == LIKE]
    avoided = [ref for ref in references if ref.get("stance") == AVOID]
    return liked, avoided
