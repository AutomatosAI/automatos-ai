"""PRD-259 §3 — the .autoskin package reader: a zip in, a validated skin and its checked files out.

:func:`read_package` refuses, each with a code and a clear message:

* a package over ``SKIN_MAX_PACKAGE_MB`` (``PACKAGE_TOO_LARGE``: the route answers 413)
  or anything but a zip (``NOT_A_ZIP``, zip64 included: a skin never needs it);
* a central directory bigger than ``MAX_MEMBERS`` entries could need, measured from the
  end record BEFORE zipfile parses it (a flood of tiny entries would cost memory first);
* a member that is a symlink, has an absolute path, a '..' segment or a backslash,
  repeats a name, is encrypted, or isn't ``skin.json`` or one of the asset paths in
  ``services/skins/assets.py``;
* a total uncompressed size over the package cap: checked from the central directory
  before anything is read, and again while reading (each read stops one byte past
  what is left);
* a skin.json the validator (``services/skins/validate.py``) refuses, a file it names
  that is missing, a file it does not name, a 'custom' mascot without all its files;
* an asset whose bytes are not what its path says, or over its cap.

Nothing here touches storage or the database: the route stores what this returns.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import zipfile
import zlib
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from config import config
from services.skins.assets import (
    MB, asset_kind, checked_type, mascot_paths, too_large_error,
)
from services.skins.fields import SkinError, echo, skin_error
from services.skins.schema import FONT_FILE_PATTERN, FONT_SLOTS
from services.skins.storage import StoredFile
from services.skins.validate import validate_skin

SKIN_IMPORT_ROUTE = "/api/skins/import"
SKIN_JSON_PATH = "skin.json"
SKIN_JSON_MAX_BYTES = 64 * 1024
ZIP_SIGNATURE = b"PK\x03\x04"
# Folder entries a zip tool writes on its own; they hold nothing and are skipped.
DIRECTORY_ENTRIES = frozenset({"assets/", "assets/fonts/", "assets/mascot/"})
MAX_MEMBERS = 2 + len(FONT_SLOTS) + len(mascot_paths()) + len(DIRECTORY_ENTRIES)
# The end-of-central-directory record: where it is, how big the directory says it is.
EOCD_SIGNATURE = b"PK\x05\x06"
EOCD_BYTES = 22
EOCD_CD_SIZE_AT = 12
ZIP_COMMENT_MAX = 0xFFFF
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_LOCATOR_BYTES = 20
# A central entry is 46 bytes plus its name and extra fields; a skin's names are short.
CENTRAL_ENTRY_MAX_BYTES = 1024
CENTRAL_DIRECTORY_MAX_BYTES = MAX_MEMBERS * CENTRAL_ENTRY_MAX_BYTES
ENCRYPTED_FLAG = 0x1
DRIVE_PREFIX = re.compile(r"[A-Za-z]:")
# What a damaged or hostile member raises while it is decompressed.
READ_ERRORS = (zipfile.BadZipFile, zlib.error, NotImplementedError, RuntimeError, EOFError, ValueError)

PACKAGE_ERROR_CODES: Tuple[str, ...] = (
    "PACKAGE_TOO_LARGE", "NOT_A_ZIP", "TOO_MANY_FILES", "SYMLINK", "UNSAFE_PATH", "DUPLICATE_FILE",
    "ENCRYPTED", "NOT_ALLOWED", "UNCOMPRESSED_TOO_LARGE", "CORRUPT_FILE", "MISSING_SKIN_JSON", "BAD_JSON",
    "UNDECLARED_FILE", "MISSING_FILE", "MASCOT_INCOMPLETE", "NOT_WOFF2", "BAD_FONT_FILE", "FONT_TOO_LARGE",
    "NOT_PNG", "PREVIEW_TOO_LARGE", "BAD_MASCOT_FILE", "MASCOT_TOO_LARGE",
)


class SkinPackageError(ValueError):
    """The package was refused; ``errors`` are ``{code, path, message}`` like the validator's."""

    def __init__(self, errors: List[SkinError]) -> None:
        super().__init__("; ".join(error["message"] for error in errors))
        self.errors = errors

    @property
    def too_large(self) -> bool:
        """True when the package itself is over the cap (the route's 413)."""
        return any(error["code"] == "PACKAGE_TOO_LARGE" for error in self.errors)


class SkinPackage(NamedTuple):
    """A read package: the validated skin, its asset manifest and the files to store."""

    skin: Dict[str, Any]
    manifest: Dict[str, Dict[str, Any]]
    files: Dict[str, StoredFile]


def package_max_bytes() -> int:
    """The package cap in bytes (``config.SKIN_MAX_PACKAGE_MB``)."""
    return config.SKIN_MAX_PACKAGE_MB * MB


def _refuse(code: str, path: str, message: str) -> SkinPackageError:
    return SkinPackageError([skin_error(code, path, message)])


def _central_directory_bytes(data: bytes) -> Optional[int]:
    """The central directory's size from the end record zipfile will read; None for no record or zip64."""
    at = data.rfind(EOCD_SIGNATURE, max(0, len(data) - EOCD_BYTES - ZIP_COMMENT_MAX))
    if at < 0 or at + EOCD_BYTES > len(data):
        return None
    locator_at = at - ZIP64_LOCATOR_BYTES
    if locator_at >= 0 and data[locator_at:locator_at + len(ZIP64_LOCATOR_SIGNATURE)] == ZIP64_LOCATOR_SIGNATURE:
        return None
    return int.from_bytes(data[at + EOCD_CD_SIZE_AT:at + EOCD_CD_SIZE_AT + 4], "little")


def _open_zip(data: bytes) -> zipfile.ZipFile:
    if len(data) > package_max_bytes():
        raise _refuse("PACKAGE_TOO_LARGE", "", f"A skin package must be {config.SKIN_MAX_PACKAGE_MB} MB or smaller.")
    not_zip = _refuse("NOT_A_ZIP", "", "A skin package must be a .autoskin (zip) file.")
    directory_bytes = _central_directory_bytes(data) if data.startswith(ZIP_SIGNATURE) else None
    if directory_bytes is None:
        raise not_zip
    if directory_bytes > CENTRAL_DIRECTORY_MAX_BYTES:
        raise _refuse("TOO_MANY_FILES", "", f"A skin package holds at most {MAX_MEMBERS} entries.")
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, zlib.error, EOFError, ValueError) as exc:
        raise not_zip from exc


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def _is_unsafe(name: str) -> bool:
    if not name or name.startswith("/") or "\\" in name or DRIVE_PREFIX.match(name):
        return True
    return ".." in name.split("/") or any(ord(ch) < 32 or ord(ch) == 127 for ch in name)


def _is_allowed(name: str) -> bool:
    return name == SKIN_JSON_PATH or name in DIRECTORY_ENTRIES or asset_kind(name) is not None


def _member_error(info: zipfile.ZipInfo, seen: Set[str]) -> Optional[SkinError]:
    name, shown = info.filename, echo(info.filename)
    if _is_symlink(info):
        return skin_error("SYMLINK", name, f"{shown} is a link: a skin package can't hold links.")
    if _is_unsafe(name):
        return skin_error("UNSAFE_PATH", name, f"{shown} is not a plain relative path inside the package.")
    if name in seen:
        return skin_error("DUPLICATE_FILE", name, f"{shown} is in the package more than once.")
    if info.flag_bits & ENCRYPTED_FLAG:
        return skin_error("ENCRYPTED", name, f"{shown} is encrypted: a skin package can't be.")
    if not _is_allowed(name):
        return skin_error(
            "NOT_ALLOWED", name,
            f"{shown} can't be in a skin: only skin.json, assets/preview.png, assets/fonts/<name>.woff2 "
            "and assets/mascot/<state>.webm|.mp4|.png.",
        )
    return None


def _checked_members(zf: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    """The package's files (folder entries dropped), once every member passed the path rules."""
    infos = zf.infolist()
    if len(infos) > MAX_MEMBERS:
        raise _refuse("TOO_MANY_FILES", "", f"A skin package holds at most {MAX_MEMBERS} entries.")
    seen: Set[str] = set()
    errors: List[SkinError] = []
    for info in infos:
        error = _member_error(info, seen)
        seen.add(info.filename)
        if error is not None:
            errors.append(error)
    if errors:
        raise SkinPackageError(errors)
    if sum(info.file_size for info in infos) > package_max_bytes():
        raise _refuse("UNCOMPRESSED_TOO_LARGE", "", "The package's files unpack to more than the package limit.")
    return [info for info in infos if info.filename not in DIRECTORY_ENTRIES]


def _read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> Optional[bytes]:
    """The member's bytes, or None when it unpacks to more than ``limit`` (never reads past it)."""
    try:
        with zf.open(info) as handle:
            data = handle.read(limit + 1)
    except READ_ERRORS as exc:
        raise _refuse("CORRUPT_FILE", info.filename, f"{info.filename} is damaged and can't be unpacked.") from exc
    return data if len(data) <= limit else None


def _read_skin_json(zf: zipfile.ZipFile, members: Dict[str, zipfile.ZipInfo]) -> Dict[str, Any]:
    info = members.get(SKIN_JSON_PATH)
    if info is None:
        raise _refuse("MISSING_SKIN_JSON", SKIN_JSON_PATH, "A skin package must hold skin.json at its top level.")
    raw = _read_member(zf, info, SKIN_JSON_MAX_BYTES)
    if raw is None:
        raise _refuse("BAD_JSON", SKIN_JSON_PATH, f"skin.json must be {SKIN_JSON_MAX_BYTES // 1024} KB or smaller.")
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _refuse("BAD_JSON", SKIN_JSON_PATH, "skin.json is not valid JSON text.") from exc
    result = validate_skin(data)
    if not result["ok"]:
        raise SkinPackageError(result["errors"])
    return result["skin"]


def declared_assets(skin: Dict[str, Any]) -> Set[str]:
    """Every asset path a validated skin names: its font files, its preview, a custom mascot's files."""
    fonts = {value for value in (skin.get("fonts") or {}).values() if FONT_FILE_PATTERN.fullmatch(value)}
    preview = {skin["preview"]} if skin.get("preview") else set()
    custom = (skin.get("mascot") or {}).get("mode") == "custom"
    return fonts | preview | (set(mascot_paths()) if custom else set())


def _declaration_errors(declared: Set[str], present: Iterable[str]) -> List[SkinError]:
    found = set(present) - {SKIN_JSON_PATH}
    undeclared = [
        skin_error("UNDECLARED_FILE", path, f"{path} is in the package but skin.json doesn't use it.")
        for path in sorted(found - declared)
    ]
    missing = declared - found
    missing_mascot = missing & set(mascot_paths())
    mascot = [skin_error(
        "MASCOT_INCOMPLETE", "mascot",
        f'A "custom" mascot needs every state as .webm, .mp4 and .png: {len(missing_mascot)} files are missing.',
    )] if missing_mascot else []
    absent = [
        skin_error("MISSING_FILE", path, f"skin.json uses {path}, which isn't in the package.")
        for path in sorted(missing - missing_mascot)
    ]
    return undeclared + absent + mascot


def _read_asset(zf: zipfile.ZipFile, info: zipfile.ZipInfo, budgets: Dict[str, int]) -> Tuple[bytes, str]:
    """One asset's bytes and checked type; ``budgets`` are the bytes left (total, and per shared-cap kind)."""
    path = info.filename
    kind = asset_kind(path)
    cap = budgets[kind.name] if kind.shared_cap else kind.cap_bytes()
    data = _read_member(zf, info, min(cap, budgets["total"]))
    if data is None:
        too_large = cap <= budgets["total"]
        raise SkinPackageError([too_large_error(kind, path)] if too_large else [skin_error(
            "UNCOMPRESSED_TOO_LARGE", path, "The package's files unpack to more than the package limit.",
        )])
    content_type, error = checked_type(kind, path, data)
    if error is not None:
        raise SkinPackageError([error])
    return data, content_type


def _read_assets(zf: zipfile.ZipFile, infos: List[zipfile.ZipInfo], used: int) -> Dict[str, StoredFile]:
    budgets = {"total": package_max_bytes() - used, "mascot": config.SKIN_MAX_MASCOT_MB * MB}
    files: Dict[str, StoredFile] = {}
    for info in infos:
        data, content_type = _read_asset(zf, info, budgets)
        kind = asset_kind(info.filename)
        budgets = {**budgets, "total": budgets["total"] - len(data)}
        if kind.shared_cap:
            budgets = {**budgets, kind.name: budgets[kind.name] - len(data)}
        files = {**files, info.filename: StoredFile(data, content_type)}
    return files


def asset_manifest(files: Dict[str, StoredFile]) -> Dict[str, Dict[str, Any]]:
    """``{path: {type, size, sha256}}`` for the stored files: what the assets route may serve."""
    return {
        path: {"type": stored.content_type, "size": len(stored.data), "sha256": hashlib.sha256(stored.data).hexdigest()}
        for path, stored in sorted(files.items())
    }


def read_package(data: bytes) -> SkinPackage:
    """Read and check a .autoskin package; :class:`SkinPackageError` says why one is refused."""
    with _open_zip(data) as zf:
        infos = _checked_members(zf)
        members = {info.filename: info for info in infos}
        skin = _read_skin_json(zf, members)
        errors = _declaration_errors(declared_assets(skin), members)
        if errors:
            raise SkinPackageError(errors)
        assets = [info for info in infos if info.filename != SKIN_JSON_PATH]
        files = _read_assets(zf, assets, members[SKIN_JSON_PATH].file_size)
    return SkinPackage(skin=skin, manifest=asset_manifest(files), files=files)
