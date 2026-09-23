"""F087 (night 3): a file uploaded again under the same name REPLACES the document.

Night 3: re-uploading a corrected sheet created a second document beside the
first — no "replace?", no version, nothing marking which copy was current. Six
copies of the Christmas sheet sat side by side; agents read whichever they
found, and Auto answered from the old price after the new one was uploaded.
The only way to replace was delete-then-upload.

Now an upload whose name matches an earlier upload of the same workspace, in
the same team scope, is ingested under that document's id: references to it
keep working, retrieval sees only the new text, and the old source is kept in
``doc_metadata.versions`` so nothing is lost. Agents' own reports and
cloud-synced files keep their own identities and are never replaced this way.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import or_
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

NOT_REPLACED_SOURCE_TYPES = ("agent_output",)


def replaceable_document(db: Session, workspace_id: Any, filename: str, team_access: Sequence[str]):
    """The earlier upload a new upload of ``filename`` replaces, or None."""
    from core.models import Document
    from core.models.cloud_sync import CloudDocument

    candidates = (
        db.query(Document)
        .filter(
            Document.workspace_id == workspace_id,
            Document.filename == filename,
            or_(Document.source_type.is_(None), Document.source_type.notin_(NOT_REPLACED_SOURCE_TYPES)),
            ~db.query(CloudDocument.id).filter(CloudDocument.document_id == Document.id).exists(),
        )
        .order_by(Document.id.desc())
        .all()
    )
    scope = sorted(team_access or [])
    return next((d for d in candidates if sorted(d.team_access or []) == scope), None)


def record_replacement(document: Any, *, file_path: str, file_size: int, content_hash: str, file_type: str,
                       replaced_by: Optional[str] = None, now: Optional[datetime] = None) -> int:
    """Point the document at its new source and keep the old one on record.
    Rebuilds ``doc_metadata`` (never mutates it). Returns the new version number."""
    meta: Dict[str, Any] = dict(document.doc_metadata or {})
    versions: List[Dict[str, Any]] = list(meta.get("versions") or [])
    versions.append({
        "file_path": document.file_path,
        "content_hash": document.content_hash,
        "file_size": document.file_size,
        "replaced_at": (now or datetime.now(timezone.utc)).isoformat(),
        "replaced_by": replaced_by,
    })
    meta.pop("kept_pct", None)              # measured again when the new text is ingested
    document.doc_metadata = {**meta, "versions": versions}
    document.file_path = file_path
    document.file_size = file_size
    document.content_hash = content_hash
    document.file_type = file_type
    document.status = "processing"
    return len(versions) + 1


async def replace_document(db: Session, document: Any, *, workspace_id: Any, file_path: str, file_size: int,
                           content_hash: str, file_type: str, replaced_by: Optional[str] = None,
                           tags: Optional[Sequence[str]] = None, description: Optional[str] = None) -> int:
    """Re-ingest ``document`` from its new source under the same id: what the old
    text stored is cleared first, so retrieval sees only the new one. Tags and a
    description sent with the upload replace the old ones; left out, they stay.
    Returns the new version number; ``document.status`` says how ingestion went."""
    from api.documents import get_document_manager, processing_type

    version = record_replacement(document, file_path=file_path, file_size=file_size, content_hash=content_hash,
                                 file_type=file_type, replaced_by=replaced_by)
    if tags:
        document.tags = list(tags)
    if description:
        document.description = description
    db.commit()
    try:
        manager = get_document_manager(str(workspace_id))
        manager.clear_chunks(document.id)
        await manager._process_document(document.id, file_path, processing_type(file_type))
        db.refresh(document)
    except Exception as exc:
        logger.error("F087: replacing document %s failed: %s", document.id, exc, exc_info=True)
        document.status = "failed"
        db.commit()
    return version


def replaced_message(filename: str, version: int, status: str) -> str:
    if status == "failed":
        return (f"Replacing {filename} failed — its previous source is kept in the document's history; "
                "upload it again or reprocess the document.")
    return (f"Replaced {filename} (version {version}) — agents now read the new copy; "
            "the previous one is kept in the document's history.")
