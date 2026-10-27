"""PRD-251 S1.1c (D6): the rendered file records a finished render writes into a post's ``media``.

A render writes ``media`` as ``{aspect: [file records]}``, each with its Deliverable id and the
sha256 of its bytes, so the content hash (and an approval) binds to the exact files. Records
come only from a render (``service.finish_render``); an edit may set ``media`` only to Deliverable
ids, so no client can forge a digest. Pure checks: the caller passes the error it raises
(``service.InvalidPost``), so this module imports nothing of the lifecycle.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Type

# A rendered file record in ``media``: its Deliverable, the sha256 of its bytes and what the
# renderer measured.
RENDERED_FILE_REQUIRED = ("deliverable_id", "name", "sha256", "bytes")
RENDERED_FILE_OPTIONAL = ("content_type", "duration", "width", "height")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MEASURED = ("duration", "width", "height")


def _optional_number(record: Mapping[str, Any], key: str, where: str, error: Type[Exception]) -> Optional[float]:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise error(f"{where}.{key} must be a non-negative number")
    return value


def _required(record: Mapping[str, Any], where: str, error: Type[Exception]) -> Dict[str, Any]:
    """The record's required keys, each checked: its Deliverable, name, digest and size."""
    for key in ("deliverable_id", "name"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise error(f"{where}.{key} is required")
    if not isinstance(record["sha256"], str) or not SHA256_PATTERN.match(record["sha256"]):
        raise error(f"{where}.sha256 must be a lowercase sha256")
    size = record["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise error(f"{where}.bytes must be a positive whole number")
    return {key: record[key] for key in RENDERED_FILE_REQUIRED}


def rendered_file(record: Any, where: str, error: Type[Exception]) -> Dict[str, Any]:
    """One rendered file record, checked, as a new dict; ``error`` for anything else."""
    if not isinstance(record, dict):
        raise error(f"{where} must be a rendered file record")
    unknown = [k for k in record if k not in RENDERED_FILE_REQUIRED + RENDERED_FILE_OPTIONAL]
    missing = [k for k in RENDERED_FILE_REQUIRED if k not in record]
    if unknown or missing:
        raise error(f"{where} has unknown keys {unknown!r} or lacks {missing!r}")
    clean = _required(record, where, error)
    if record.get("content_type") is not None:
        if not isinstance(record["content_type"], str):
            raise error(f"{where}.content_type must be a string")
        clean["content_type"] = record["content_type"]
    for key in MEASURED:
        value = _optional_number(record, key, where, error)
        if value is not None:
            clean[key] = value
    return clean


def rendered_media(media: Any, error: Type[Exception]) -> Dict[str, List[Dict[str, Any]]]:
    """``{aspect: [rendered file records]}``, at least one file; ``error`` for anything else."""
    if not isinstance(media, Mapping) or not media:
        raise error("a render must produce at least one file")
    clean: Dict[str, List[Dict[str, Any]]] = {}
    for aspect, records in media.items():
        if not isinstance(aspect, str) or not aspect.strip():
            raise error("a rendered aspect must be named")
        if not isinstance(records, (list, tuple)) or not records:
            raise error(f"media.{aspect} must list the rendered files")
        clean[aspect] = [rendered_file(r, f"media.{aspect}[{i}]", error) for i, r in enumerate(records)]
    return clean


__all__ = ["RENDERED_FILE_OPTIONAL", "RENDERED_FILE_REQUIRED", "SHA256_PATTERN", "rendered_file", "rendered_media"]
