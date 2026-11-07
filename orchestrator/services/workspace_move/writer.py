"""PRD-273 §5.1 — the ``.automatos-workspace`` zip, written part by part to a file on disk.

Each part is streamed into its zip entry as it is produced, so memory stays flat however
large the workspace is: a table's rows arrive in batches, a file in chunks. Every part's
row count, plaintext size and SHA-256 are counted on the way through and go into
``manifest.json``, written last. Without a passphrase an entry is deflated by the zip;
with one it is sealed (``crypto.SealingStream``) and stored as it is.

A row becomes one JSON line: sorted keys, no spaces, UTF-8. A value JSON has no type for is
written as text (a time or a date in ISO 8601, a UUID, a decimal) or base64 (bytes); the
manifest's ``columns`` gives each table column's SQL type, so a reader knows which.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, BinaryIO, Dict, Iterable, Iterator, Mapping, Optional
from uuid import UUID

from services.workspace_move.crypto import FileKey, SealingStream

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
LINE_END = b"\n"
# A zip stores times from 1980; an entry's time is the export's.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class ExportTooLarge(RuntimeError):
    """The export holds more data than the install allows (``WORKSPACE_EXPORT_MAX_TOTAL_MB``)."""


def json_value(value: Any) -> Any:
    """``json.dumps``' fallback for the types a row holds that JSON has none for."""
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (UUID, Decimal)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    raise TypeError(f"a value of type {type(value).__name__} has no JSON form")


def json_bytes(document: Any) -> bytes:
    """``document`` as compact, key-sorted UTF-8 JSON."""
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=json_value)
    return text.encode("utf-8")


class PartSink:
    """One entry being written: counts lines, bytes and the SHA-256 of what passes through."""

    def __init__(self, raw: BinaryIO, sealer: Optional[SealingStream], budget: "_Budget") -> None:
        self._raw = raw
        self._sealer = sealer
        self._budget = budget
        self._sha = hashlib.sha256()
        self.size = 0
        self.rows = 0

    def write(self, data: bytes) -> None:
        """Append plaintext bytes to the entry."""
        if not data:
            return
        self._budget.spend(len(data))
        self._sha.update(data)
        self.size += len(data)
        if self._sealer is not None:
            self._sealer.write(data)
        else:
            self._raw.write(data)

    def write_row(self, row: Mapping[str, Any]) -> None:
        """Append one JSON line."""
        self.write(json_bytes(row) + LINE_END)
        self.rows += 1

    def summary(self) -> Dict[str, Any]:
        """The manifest's record of the entry: its rows, plaintext bytes and SHA-256."""
        return {"rows": self.rows, "bytes": self.size, "sha256": self._sha.hexdigest()}


class _Budget:
    """The plaintext an export may still write; ExportTooLarge once it is spent."""

    def __init__(self, max_bytes: Optional[int]) -> None:
        self._left = max_bytes

    def spend(self, size: int) -> None:
        if self._left is None:
            return
        self._left -= size
        if self._left < 0:
            raise ExportTooLarge("The workspace holds more data than one export may carry on this install.")


class MoveFileWriter:
    """The zip of one export. ``part`` opens an entry; ``finish`` writes the manifest and
    closes the file; ``parts`` is every entry's record so far, by name."""

    def __init__(self, out: BinaryIO, *, key: Optional[FileKey], stamp: datetime,
                 max_bytes: Optional[int] = None) -> None:
        self._zip = zipfile.ZipFile(out, mode="w", allowZip64=True)
        self._key = key
        self._stamp = max(stamp.timetuple()[:6], ZIP_EPOCH)
        self._budget = _Budget(max_bytes)
        self._parts: Dict[str, Dict[str, Any]] = {}

    @property
    def encrypted(self) -> bool:
        return self._key is not None

    @property
    def parts(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._parts)

    def has(self, name: str) -> bool:
        return name in self._parts

    def _info(self, name: str, *, sealed: bool) -> zipfile.ZipInfo:
        info = zipfile.ZipInfo(name, date_time=self._stamp)
        info.compress_type = zipfile.ZIP_STORED if sealed else zipfile.ZIP_DEFLATED
        return info

    @contextmanager
    def part(self, name: str) -> Iterator[PartSink]:
        """An entry named ``name``, open for writing; its record is kept once it closes."""
        if name in self._parts or name == MANIFEST_NAME:
            raise ValueError(f"{name} is already in the file")
        sealed = self._key is not None
        with self._zip.open(self._info(name, sealed=sealed), mode="w", force_zip64=True) as raw:
            sealer = SealingStream(self._key, name, raw) if sealed else None
            sink = PartSink(raw, sealer, self._budget)
            yield sink
            if sealer is not None:
                sealer.close()
        self._parts[name] = sink.summary()

    def write_rows(self, name: str, rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        """An entry of JSON lines, one per row; its record."""
        with self.part(name) as sink:
            for row in rows:
                sink.write_row(row)
        return self._parts[name]

    def write_json(self, name: str, document: Any) -> Dict[str, Any]:
        """An entry holding one JSON document; its record."""
        with self.part(name) as sink:
            sink.write(json_bytes(document))
        return self._parts[name]

    def finish(self, manifest: Mapping[str, Any]) -> None:
        """Write ``manifest.json`` (never sealed) and close the zip."""
        self._zip.writestr(self._info(MANIFEST_NAME, sealed=False), json_bytes(manifest))
        self._zip.close()

    def abort(self) -> None:
        """Close the zip after a failure; the caller deletes the file. A close that fails too is
        logged, so the export's own error is the one raised."""
        try:
            self._zip.close()
        except (OSError, ValueError):
            logger.exception("[workspace-move] the unfinished export file did not close cleanly")


__all__ = ["ExportTooLarge", "MANIFEST_NAME", "MoveFileWriter", "PartSink", "json_bytes", "json_value"]
