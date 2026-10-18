"""PRD-261 — the hashes that name a theme package and each of its files.

* Each file: the sha256 of its bytes, as lower-case hex (``file_manifest``).
* The package: the sha256 of its sorted ``path\\0sha256\\n`` lines (``package_digest``), so the
  digest depends on the files alone, never on the zip's member order, timestamps or
  compression. The files route serves ``/api/themes/files/<digest>/<path>`` by it.
* A core page file a clone tracks: the sha256 of its bytes with CRLF read as LF
  (``page_source_sha256``), so a Windows checkout hashes the same as CI's.

Standard library only: ``tests/fixtures/themes/build.py`` loads this file by path when it
runs as a command in the browser lane, where the orchestrator's packages aren't installed.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Tuple, Union

FileEntry = Dict[str, Union[str, int]]

DIGEST_HEX_CHARS = 64
PATH_SEPARATOR = "\0"
LINE_END = "\n"


def sha256_hex(data: bytes) -> str:
    """The sha256 of ``data`` as 64 lower-case hex characters."""
    return hashlib.sha256(data).hexdigest()


def manifest_entry(data: bytes, content_type: str) -> FileEntry:
    """One file's row in a file manifest: its checked type, its size and its sha256."""
    return {"type": content_type, "size": len(data), "sha256": sha256_hex(data)}


def file_manifest(files: Mapping[str, Tuple[bytes, str]]) -> Dict[str, FileEntry]:
    """``{path: {type, size, sha256}}`` for ``{path: (bytes, content type)}``, in path order."""
    return {path: manifest_entry(data, content_type) for path, (data, content_type) in sorted(files.items())}


def package_digest(file_hashes: Mapping[str, str]) -> str:
    """The package's digest from ``{path: sha256 hex}``: sha256 of the sorted ``path\\0sha\\n`` lines."""
    lines = "".join(f"{path}{PATH_SEPARATOR}{file_hashes[path]}{LINE_END}" for path in sorted(file_hashes))
    return sha256_hex(lines.encode("utf-8"))


def digest_of_manifest(manifest: Mapping[str, Mapping[str, Union[str, int]]]) -> str:
    """The package digest of a file manifest (each row's ``sha256``)."""
    return package_digest({path: str(row["sha256"]) for path, row in manifest.items()})


def page_source_sha256(data: bytes) -> str:
    """A core page file's sha256, CRLF read as LF (``services/themes/data/core-pages.json``)."""
    return sha256_hex(data.replace(b"\r\n", b"\n"))
