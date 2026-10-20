"""A media-render file's mark, checked and recorded (PRD-264 EUAI-06b, EUAI-10).

media-render marks and signs every file with an AI element, and says so: each marked
output of a finished job names its ``digital_source_type``, the ``sha256`` of the bytes it
ships and its C2PA ``manifest_id`` (``None`` when signing failed and the file carries its
XMP mark alone); a marked ``POST /jpeg`` answer carries the same as the
``X-Digital-Source-Type``, ``X-Content-SHA256`` and ``X-C2PA-Manifest-Id`` headers. The
service keeps no database, so the orchestrator writes each kept file's provenance row
(``services.content_provenance.record_produced_file``), and the public check
(``GET /api/public/provenance/{sha256}``) can say Automatos AI made it.

* :func:`marked_file` and :func:`marked_from_headers` read a file's mark: ``None`` for a
  file with no AI element, which gets no row.
* :meth:`MarkedFile.names` proves the bytes that arrived are the bytes media-render marked
  (the client refuses a file whose digest differs: its row would name a hash no copy has).
* :func:`record_marked` writes the row, best effort and off the event loop.

No hash is ever logged here (FR-17). ``services`` is imported when a file is recorded,
never when this module loads.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

# What a marked /jpeg answer's headers are called (services/media-render/media_render/server.py).
SOURCE_TYPE_HEADER = "X-Digital-Source-Type"
SHA256_HEADER = "X-Content-SHA256"
MANIFEST_HEADER = "X-C2PA-Manifest-Id"

# What a marked output (or /tts line) names (media_render/content_credentials.py, MarkRecord.fields).
SOURCE_TYPE_FIELD = "digital_source_type"
SHA256_FIELD = "sha256"
MANIFEST_FIELD = "manifest_id"

# What a media-render file's provenance row says made it: the service, never a person or a prompt.
MEDIA_RENDER_GENERATOR = "Automatos AI media-render"
# The provenance kind (core.models.content_provenance.PROVENANCE_KINDS) of each file media-render ships.
KIND_BY_SUFFIX = {".mp4": "video", ".png": "image", ".jpg": "image", ".jpeg": "image", ".wav": "audio"}


@dataclass(frozen=True)
class MarkedFile:
    """What media-render says about a file it marked: its IPTC term, digest and manifest id."""

    digital_source_type: str
    sha256: Any
    manifest_id: Any = None

    def names(self, digest: str) -> bool:
        """True when this mark is of exactly the bytes whose SHA-256 is ``digest``, with a
        well-formed manifest id (a non-empty string, or none when signing failed)."""
        manifest_ok = self.manifest_id is None or (isinstance(self.manifest_id, str) and bool(self.manifest_id))
        return isinstance(self.sha256, str) and self.sha256 == digest and manifest_ok


def marked_file(fields: Mapping[str, Any]) -> Optional[MarkedFile]:
    """The mark an output (or a /tts line) names; ``None`` when it names none (no AI element)."""
    source_type = fields.get(SOURCE_TYPE_FIELD)
    if not isinstance(source_type, str) or not source_type:
        return None
    return MarkedFile(source_type, fields.get(SHA256_FIELD), fields.get(MANIFEST_FIELD))


def marked_from_headers(headers: Mapping[str, str]) -> Optional[MarkedFile]:
    """The mark a /jpeg answer's headers carry (``httpx.Headers`` reads them case-insensitively)."""
    return marked_file({
        SOURCE_TYPE_FIELD: headers.get(SOURCE_TYPE_HEADER),
        SHA256_FIELD: headers.get(SHA256_HEADER),
        MANIFEST_FIELD: headers.get(MANIFEST_HEADER),
    })


def provenance_kind(file_name: str) -> Optional[str]:
    """The provenance kind of a file media-render ships, by its extension; ``None`` for another."""
    return KIND_BY_SUFFIX.get(PurePosixPath(file_name).suffix.lower())


def record_marked(workspace_id: Any, mark: Optional[MarkedFile], file_name: str) -> bool:
    """Write the provenance row of a marked file the orchestrator keeps (best effort, off the
    event loop: ``record_produced_file``). False when the file carries no mark, so it gets
    no row, or when its kind is not one the record holds (logged by extension, never by hash)."""
    if mark is None:
        return False
    kind = provenance_kind(file_name)
    if kind is None:
        logger.error("[provenance] a marked %s file of workspace %s has no provenance kind; it is not recorded",
                     PurePosixPath(file_name).suffix or "extensionless", workspace_id)
        return False
    from services.content_provenance import record_produced_file

    record_produced_file(workspace_id, mark.sha256, kind, MEDIA_RENDER_GENERATOR, manifest_id=mark.manifest_id)
    return True


__all__ = [
    "KIND_BY_SUFFIX",
    "MANIFEST_HEADER",
    "MEDIA_RENDER_GENERATOR",
    "SHA256_HEADER",
    "SOURCE_TYPE_HEADER",
    "MarkedFile",
    "marked_file",
    "marked_from_headers",
    "provenance_kind",
    "record_marked",
]
