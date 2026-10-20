"""PRD-261 §4.2 — the .autotheme reader: a zip in, a checked theme package out.

Started from PRD-259's skin package reader (#1136): the same zip checks, with the file
whitelist widened to the theme layout. :func:`read_theme_package` refuses, each
with a code, the path and a plain sentence:

* a package over ``THEME_MAX_PACKAGE_MB`` (``PACKAGE_TOO_LARGE``, the route's 413), anything but a
  zip, a zip64 archive, a central directory bigger than ``MAX_MEMBERS`` entries could need
  (measured from the end record before zipfile parses it);
* a member that is a symlink, has an absolute path, a '..' segment or a backslash, repeats a
  name, is encrypted, is compressed other than stored or deflated, is an SVG (none in v1, PRD
  §6), or is not on the whitelist in ``FILE_KINDS``;
* files that unpack past the package's limits, a file over its own cap (the entry over
  ``THEME_MAX_ENTRY_KB``, a clip over ``THEME_MAX_CLIP_KB``), JS plus CSS over
  ``THEME_MAX_CODE_KB``, bytes that are not what the path says (M-02: a phone's font,
  ``assets/fonts/mobile/<NAME>.ttf`` or ``.otf``, is one complete sfnt face, ``_sfnt_problem``);
* a theme.json the manifest check refuses (``services/themes/manifest.py``); a missing entry,
  look schema or preset, or a preset the schema refuses (``look_schema.py``); a prepaint value
  outside the CSS-value grammar (``css_value.py``); CSS outside the theme's class, with
  ``!important`` or ``@import`` (``css_scope.py``); a preview, font, wordmark or mobile font the
  manifest names that is not in the package;
* M-02 (host 1.1.0, plan §2.2.11): a ``mobile`` block a phone can't draw without the theme's
  code: ``fromLooks`` with no ``boot/prepaint.json``, or a table whose ``defaults`` look, painted as
  the web paints it (``mobile_tokens.paint_look``: nothing when the defaults don't fit the table),
  gives no ``--background`` or ``--foreground`` (``MOBILE_CANNOT_PAINT``); a ``clips`` or ``stills``
  mascot missing a state's ``assets/mascot/<state>.mp4`` or ``.png`` (``MISSING_FILE``).

The result carries every file with its checked type, the file manifest
``{path: {type, size, sha256}}`` and the package digest (``digest.py``). Nothing here touches
storage or the database: the import route stores what this returns.
"""

from __future__ import annotations

import io
import re
import stat
import zipfile
import zlib
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple

from config import config
from modules.documents.brand_fonts import BrandFontError, check_woff2
from services.themes.contract import (
    CAP_SETTINGS, KB, MANIFEST_PATH, MB, ThemeError, echo, parse_json, theme_error,
)
from services.themes.css_scope import STYLE_PATH, css_scope_errors
from services.themes.css_value import PREPAINT_PATH, prepaint_errors
from services.themes.digest import digest_of_manifest, file_manifest
from services.themes.look_schema import look_file_errors, preset_ids
from services.themes.manifest import referenced_files, validate_manifest
from services.themes.mobile_manifest import MASCOT_NONE, MOBILE_FIELD
from services.themes.mobile_tokens import boot_table, missing_tokens, paint_look

THEME_IMPORT_ROUTE = "/api/themes/import"
PACKAGE_ERROR_CODES: Tuple[str, ...] = (
    "PACKAGE_TOO_LARGE", "NOT_A_ZIP", "ZIP64", "TOO_MANY_FILES", "SYMLINK", "UNSAFE_PATH", "DUPLICATE_FILE",
    "ENCRYPTED", "NOT_ALLOWED", "SVG_NOT_ALLOWED", "UNCOMPRESSED_TOO_LARGE", "CORRUPT_FILE", "FILE_TOO_LARGE",
    "ENTRY_TOO_LARGE", "CODE_TOO_LARGE", "CLIP_TOO_LARGE", "BAD_FILE_TYPE", "NOT_UTF8", "BAD_JSON", "MISSING_FILE",
    "NO_PRESET", "PRESET_ID_MISMATCH", "UNKNOWN_DEFAULT_LOOK", "MOBILE_CANNOT_PAINT",
)
MOBILE_CANNOT_PAINT = "MOBILE_CANNOT_PAINT"
ENTRY_PATH = "dist/entry.js"

# The plugin upload's limits (core/services/plugin_upload_service.py) hold for themes too.
MAX_MEMBERS = 500
UNCOMPRESSED_MAX_BYTES = 50 * MB
JSON_MAX_BYTES = 64 * KB
PREPAINT_MAX_BYTES = 256 * KB
TEXT_MAX_BYTES = 256 * KB
SOURCE_MAX_BYTES = 1 * MB
FONT_MAX_BYTES = 400 * KB
IMAGE_MAX_BYTES = 512 * KB

ZIP_SIGNATURE = b"PK\x03\x04"
EOCD_SIGNATURE = b"PK\x05\x06"
EOCD_BYTES = 22
EOCD_CD_SIZE_AT = 12
ZIP_COMMENT_MAX = 0xFFFF
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_LOCATOR_BYTES = 20
ZIP64_MARKERS = (b"\xff\xff", b"\xff\xff\xff\xff")
CENTRAL_ENTRY_MAX_BYTES = 1024
CENTRAL_DIRECTORY_MAX_BYTES = MAX_MEMBERS * CENTRAL_ENTRY_MAX_BYTES
ENCRYPTED_FLAG = 0x1
ALLOWED_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
DRIVE_PREFIX = re.compile(r"[A-Za-z]:")
READ_ERRORS = (zipfile.BadZipFile, zlib.error, NotImplementedError, RuntimeError, EOFError, ValueError)

NAME = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
LOOK_ID = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
SOURCE_EXTENSIONS = r"(?:ts|tsx|js|jsx|mjs|cjs|css|json|md|txt)"
TEXT_TYPE = "text/plain; charset=utf-8"
CODE_KINDS = frozenset({"entry", "chunk", "style"})

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
WEBM_SIGNATURE = b"\x1a\x45\xdf\xa3"

# M-02: a phone's font is one sfnt face, as the woff2 a browser gets is: TrueType outlines (00 01 00 00, or
# Apple's "true") or CFF ("OTTO"). A collection ("ttcf") is refused. The header is the version, then
# numTables and three search fields (two bytes each); each table record is tag, checksum, offset, length.
SFNT_VERSIONS = frozenset({b"\x00\x01\x00\x00", b"true", b"OTTO"})
SFNT_COLLECTION = b"ttcf"
SFNT_VERSION_BYTES = 4
SFNT_HEADER_BYTES = 12
SFNT_RECORD_BYTES = 16
SFNT_COUNT_BYTES = 2
SFNT_FIELD_BYTES = 4
SFNT_SPAN_AT = 8  # where a record's offset sits; its length follows
SFNT_MIN_TABLES = 1
SFNT_MAX_TABLES = 64

# M-02 (PB-36): the files a phone draws each mascot state from, by the block's mode (absent is "none").
# MASCOT_KEY is both theme.json's top-level mascot ({"states": [...]}) and the block's mode.
MASCOT_FOLDER = "assets/mascot"
MASCOT_KEY = "mascot"
STATES_KEY = "states"
MOBILE_MASCOT_FILES: Dict[str, Tuple[str, ...]] = {"clips": ("mp4", "png"), "stills": ("png",), MASCOT_NONE: ()}
MOBILE_FROM_LOOKS = "fromLooks"
PREPAINT_DEFAULTS = "defaults"


def cap_bytes(name: str) -> int:
    """One of ``CAP_SETTINGS`` in bytes: config's value once B2-wire names it, else the default."""
    setting = CAP_SETTINGS[name]
    return int(getattr(config, setting.config_name, setting.default)) * setting.unit


def _woff2_problem(data: bytes) -> Optional[str]:
    try:
        check_woff2(data)
    except BrandFontError as exc:
        return str(exc)
    return None


def _big_endian(data: bytes, at: int, size: int) -> int:
    return int.from_bytes(data[at:at + size], "big")


def _sfnt_tables(data: bytes) -> Tuple[int, Optional[str]]:
    """The face's table count once its header and table directory fit the file, else why not."""
    version = data[:SFNT_VERSION_BYTES]
    if not data:
        return 0, "the file is empty"
    if version == SFNT_COLLECTION:
        return 0, "it is a font collection (ttcf), and a theme ships one face per file"
    if version not in SFNT_VERSIONS:
        return 0, "its bytes are not a TrueType or OpenType font"
    if len(data) < SFNT_HEADER_BYTES:
        return 0, "its font header is cut short"
    tables = _big_endian(data, SFNT_VERSION_BYTES, SFNT_COUNT_BYTES)
    if not SFNT_MIN_TABLES <= tables <= SFNT_MAX_TABLES:
        return 0, f"its header lists {tables} tables, and a font holds {SFNT_MIN_TABLES} to {SFNT_MAX_TABLES}"
    if SFNT_HEADER_BYTES + tables * SFNT_RECORD_BYTES > len(data):
        return 0, "its table directory runs past the end of the file"
    return tables, None


def _sfnt_problem(data: bytes) -> Optional[str]:
    """Why ``data`` is not one complete sfnt face (a .ttf or .otf), or None: every table lies inside the file."""
    tables, problem = _sfnt_tables(data)
    if problem is not None:
        return problem
    for index in range(tables):
        at = SFNT_HEADER_BYTES + index * SFNT_RECORD_BYTES + SFNT_SPAN_AT
        offset = _big_endian(data, at, SFNT_FIELD_BYTES)
        length = _big_endian(data, at + SFNT_FIELD_BYTES, SFNT_FIELD_BYTES)
        if offset + length > len(data):
            return f"its table {index + 1} of {tables} runs past the end of the file"
    return None


def _sniff(test: Callable[[bytes], bool], what: str) -> Callable[[bytes], Optional[str]]:
    return lambda data: None if test(data) else f"its bytes are not {what}"


IS_PNG = _sniff(lambda data: data.startswith(PNG_SIGNATURE), "a PNG image")
IS_WEBP = _sniff(lambda data: data[:4] == b"RIFF" and data[8:12] == b"WEBP", "a WebP image")
IS_WEBM = _sniff(lambda data: data.startswith(WEBM_SIGNATURE), "a WebM video")
IS_MP4 = _sniff(lambda data: data[4:8] == b"ftyp", "an MP4 video")


class FileKind(NamedTuple):
    """One kind of file a package may hold: its paths, served type, cap and byte check.

    ``check`` returns why the bytes are wrong, or None; text kinds (``check`` None) must be UTF-8.
    """

    name: str
    pattern: "re.Pattern[str]"
    content_type: str
    cap: Callable[[], int]
    too_large_code: str
    check: Optional[Callable[[bytes], Optional[str]]] = None


def _kind(name: str, path: str, content_type: str, cap: Callable[[], int], code: str = "FILE_TOO_LARGE",
          check: Optional[Callable[[bytes], Optional[str]]] = None) -> FileKind:
    return FileKind(name, re.compile(path), content_type, cap, code, check)


# PRD §4.2's layout. Order matters only for readability: no path matches two kinds.
FILE_KINDS: Tuple[FileKind, ...] = (
    _kind("manifest", r"theme\.json", "application/json", lambda: JSON_MAX_BYTES),
    _kind("entry", r"dist/entry\.js", "text/javascript", lambda: cap_bytes("entry"), "ENTRY_TOO_LARGE"),
    _kind("chunk", rf"dist/chunks/{NAME}\.js", "text/javascript", lambda: cap_bytes("code"), "CODE_TOO_LARGE"),
    _kind("style", r"dist/theme\.css", "text/css", lambda: cap_bytes("code"), "CODE_TOO_LARGE"),
    _kind("prepaint", r"boot/prepaint\.json", "application/json", lambda: PREPAINT_MAX_BYTES),
    _kind("look-schema", r"looks/schema\.json", "application/json", lambda: JSON_MAX_BYTES),
    _kind("preset", rf"looks/presets/{LOOK_ID}\.look\.json", "application/json", lambda: JSON_MAX_BYTES),
    _kind("font", rf"assets/fonts/{NAME}\.woff2", "font/woff2", lambda: FONT_MAX_BYTES, check=_woff2_problem),
    _kind("mobile-font", rf"assets/fonts/mobile/{NAME}\.ttf", "font/ttf", lambda: FONT_MAX_BYTES, check=_sfnt_problem),
    _kind("mobile-font", rf"assets/fonts/mobile/{NAME}\.otf", "font/otf", lambda: FONT_MAX_BYTES, check=_sfnt_problem),
    _kind("licence", rf"assets/fonts/{NAME}\.(?:txt|md)", TEXT_TYPE, lambda: TEXT_MAX_BYTES),
    _kind("clip", rf"assets/mascot/{NAME}\.webm", "video/webm", lambda: cap_bytes("clip"), "CLIP_TOO_LARGE", IS_WEBM),
    _kind("clip", rf"assets/mascot/{NAME}\.mp4", "video/mp4", lambda: cap_bytes("clip"), "CLIP_TOO_LARGE", IS_MP4),
    _kind("still", rf"assets/mascot/{NAME}\.png", "image/png", lambda: IMAGE_MAX_BYTES, check=IS_PNG),
    _kind("brand", rf"assets/brand/{NAME}\.png", "image/png", lambda: IMAGE_MAX_BYTES, check=IS_PNG),
    _kind("brand", rf"assets/brand/{NAME}\.webp", "image/webp", lambda: IMAGE_MAX_BYTES, check=IS_WEBP),
    _kind("preview", rf"assets/previews/{NAME}\.png", "image/png", lambda: IMAGE_MAX_BYTES, check=IS_PNG),
    _kind("source", rf"source/(?:{NAME}/){{0,7}}{NAME}\.{SOURCE_EXTENSIONS}", TEXT_TYPE, lambda: SOURCE_MAX_BYTES),
    _kind("doc", r"(?:README\.md|CHANGELOG\.md|LICENSE(?:\.md|\.txt)?)", TEXT_TYPE, lambda: TEXT_MAX_BYTES),
)


class ThemeFile(NamedTuple):
    """One file of a theme package, with the content type its bytes were checked as."""

    data: bytes
    content_type: str


class ThemePackage(NamedTuple):
    """A read package: its checked manifest, its files, their manifest and the package digest."""

    manifest: Dict[str, Any]
    files: Dict[str, ThemeFile]
    file_manifest: Dict[str, Dict[str, Any]]
    digest: str


class ThemePackageError(ValueError):
    """The package was refused; ``errors`` are ``{code, path, message}``."""

    def __init__(self, errors: List[ThemeError]) -> None:
        super().__init__("; ".join(error["message"] for error in errors))
        self.errors = errors

    @property
    def too_large(self) -> bool:
        """True when the package itself is over the cap (the route's 413)."""
        return any(error["code"] == "PACKAGE_TOO_LARGE" for error in self.errors)


def file_kind(path: str) -> Optional[FileKind]:
    """The kind of file ``path`` names, or None when nothing may live there."""
    return next((kind for kind in FILE_KINDS if kind.pattern.fullmatch(path)), None)


def _refuse(code: str, path: str, message: str) -> ThemePackageError:
    return ThemePackageError([theme_error(code, path, message)])


def _end_record(data: bytes) -> Tuple[Optional[int], bool]:
    """The central directory's size from the end record, and whether the archive is zip64."""
    at = data.rfind(EOCD_SIGNATURE, max(0, len(data) - EOCD_BYTES - ZIP_COMMENT_MAX))
    if at < 0 or at + EOCD_BYTES > len(data):
        return None, False
    locator_at = at - ZIP64_LOCATOR_BYTES
    has_locator = locator_at >= 0 and data[locator_at:locator_at + 4] == ZIP64_LOCATOR_SIGNATURE
    fields = (data[at + 8:at + 10], data[at + 10:at + 12], data[at + 12:at + 16], data[at + 16:at + 20])
    zip64 = has_locator or any(field in ZIP64_MARKERS for field in fields)
    return int.from_bytes(data[at + EOCD_CD_SIZE_AT:at + EOCD_CD_SIZE_AT + 4], "little"), zip64


def _open_zip(data: bytes) -> zipfile.ZipFile:
    package_cap = cap_bytes("package")
    if len(data) > package_cap:
        raise _refuse("PACKAGE_TOO_LARGE", "", f"A theme package must be {package_cap // MB} MB or smaller.")
    not_zip = _refuse("NOT_A_ZIP", "", "A theme package must be a .autotheme (zip) file.")
    directory_bytes, zip64 = _end_record(data) if data.startswith(ZIP_SIGNATURE) else (None, False)
    if zip64:
        raise _refuse("ZIP64", "", "The package is a zip64 archive: a theme package never needs one. Zip it plainly.")
    if directory_bytes is None:
        raise not_zip
    if directory_bytes > CENTRAL_DIRECTORY_MAX_BYTES:
        raise _refuse("TOO_MANY_FILES", "", f"A theme package holds at most {MAX_MEMBERS} entries.")
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, zlib.error, EOFError, ValueError) as exc:
        raise not_zip from exc


def _is_unsafe(name: str) -> bool:
    if not name or name.startswith("/") or "\\" in name or DRIVE_PREFIX.match(name):
        return True
    return ".." in name.split("/") or any(ord(ch) < 32 or ord(ch) == 127 for ch in name)


def _member_error(info: zipfile.ZipInfo, seen: set) -> Optional[ThemeError]:
    name, shown = info.filename, echo(info.filename)
    if stat.S_ISLNK(info.external_attr >> 16):
        return theme_error("SYMLINK", name, f"{shown} is a link: a theme package can't hold links.")
    if _is_unsafe(name):
        return theme_error("UNSAFE_PATH", name, f"{shown} is not a plain relative path inside the package.")
    if name.lower() in seen:
        return theme_error(
            "DUPLICATE_FILE", name,
            f"{shown} is in the package more than once (names that differ only in capitals count as the same file).",
        )
    if info.flag_bits & ENCRYPTED_FLAG:
        return theme_error("ENCRYPTED", name, f"{shown} is encrypted: a theme package can't be.")
    if info.compress_type not in ALLOWED_COMPRESSION:
        return theme_error("NOT_ALLOWED", name, f"{shown} is compressed a way a theme can't be: only stored or deflated.")
    if name.lower().endswith(".svg"):
        return theme_error("SVG_NOT_ALLOWED", name, f"{shown} is an SVG: a theme can't ship SVG files. Use PNG or WebP.")
    if not name.endswith("/") and file_kind(name) is None:
        return theme_error("NOT_ALLOWED", name, f"{shown} isn't a file a theme package may hold (see PRD-261 §4.2).")
    return None


def _checked_members(zf: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    """The package's files (folder entries dropped), once every member passed the path rules."""
    infos = zf.infolist()
    if len(infos) > MAX_MEMBERS:
        raise _refuse("TOO_MANY_FILES", "", f"A theme package holds at most {MAX_MEMBERS} entries.")
    errors: List[ThemeError] = []
    for index, info in enumerate(infos):
        seen = {earlier.filename.lower() for earlier in infos[:index]}
        error = _member_error(info, seen)
        errors += [error] if error is not None else []
    if errors:
        raise ThemePackageError(errors)
    if sum(info.file_size for info in infos) > UNCOMPRESSED_MAX_BYTES:
        raise _refuse("UNCOMPRESSED_TOO_LARGE", "", "The package's files unpack to more than the package limit.")
    return [info for info in infos if not info.filename.endswith("/")]


def _read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> Optional[bytes]:
    """The member's bytes, or None when it unpacks to more than ``limit`` (never reads past it)."""
    try:
        with zf.open(info) as handle:
            data = handle.read(limit + 1)
    except READ_ERRORS as exc:
        raise _refuse("CORRUPT_FILE", info.filename, f"{echo(info.filename)} is damaged and can't be unpacked.") from exc
    return data if len(data) <= limit else None


def _content_error(kind: FileKind, path: str, data: bytes) -> Optional[ThemeError]:
    """Why ``data`` isn't what ``path`` says, or None."""
    if kind.check is not None:
        problem = kind.check(data)
        return theme_error("BAD_FILE_TYPE", path, f"{path}: {problem.rstrip('.')}.") if problem else None
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return theme_error("NOT_UTF8", path, f"{path} must be UTF-8 text.")
    return None


def _read_file(zf: zipfile.ZipFile, info: zipfile.ZipInfo, left: int) -> Tuple[Optional[bytes], Optional[ThemeError]]:
    """One file's bytes within its cap and what is left of the total, or why it is refused."""
    path, kind = info.filename, file_kind(info.filename)
    cap = kind.cap()
    data = _read_member(zf, info, min(cap, left))
    if data is None and cap <= left:
        return None, theme_error(kind.too_large_code, path, f"{path} is over its limit of {cap // KB} KB.")
    if data is None:
        return None, theme_error("UNCOMPRESSED_TOO_LARGE", path, "The package's files unpack to more than the package limit.")
    return data, _content_error(kind, path, data)


def _read_files(zf: zipfile.ZipFile, infos: List[zipfile.ZipInfo]) -> Dict[str, ThemeFile]:
    """Every file read and type-checked; all refusals at once."""
    files: Dict[str, ThemeFile] = {}
    errors: List[ThemeError] = []
    left = UNCOMPRESSED_MAX_BYTES
    for info in infos:
        data, error = _read_file(zf, info, left)
        errors += [error] if error is not None else []
        if data is not None:
            left -= len(data)
            files = {**files, info.filename: ThemeFile(data, file_kind(info.filename).content_type)}
    if errors:
        raise ThemePackageError(errors)
    return files


def _manifest(files: Mapping[str, ThemeFile]) -> Tuple[Optional[Dict[str, Any]], List[ThemeError]]:
    if MANIFEST_PATH not in files:
        return None, [theme_error("MISSING_FILE", MANIFEST_PATH, f"A theme package must hold {MANIFEST_PATH} at its top level.")]
    document, error = parse_json(files[MANIFEST_PATH].data, MANIFEST_PATH)
    if error is not None:
        return None, [error]
    result = validate_manifest(document)
    return result.manifest, [{**item, "path": f"{MANIFEST_PATH}#{item['path']}"} for item in result.errors]


def _code_errors(files: Mapping[str, ThemeFile]) -> List[ThemeError]:
    errors = [] if ENTRY_PATH in files else [
        theme_error("MISSING_FILE", ENTRY_PATH, f"A theme package must hold its code at {ENTRY_PATH}."),
    ]
    code = sum(len(item.data) for path, item in files.items() if file_kind(path).name in CODE_KINDS)
    if code > cap_bytes("code"):
        errors += [theme_error("CODE_TOO_LARGE", "dist/", f"The theme's JS and CSS together must be {cap_bytes('code') // KB} KB or smaller.")]
    return errors


def _prepaint_errors(files: Mapping[str, ThemeFile]) -> List[ThemeError]:
    if PREPAINT_PATH not in files:
        return []
    document, error = parse_json(files[PREPAINT_PATH].data, PREPAINT_PATH)
    return [error] if error is not None else prepaint_errors(document)


def _manifest_link_errors(manifest: Mapping[str, Any], files: Mapping[str, ThemeFile]) -> List[ThemeError]:
    """The files the manifest names exist; its default look is one of the presets; its CSS is scoped."""
    errors = [
        theme_error("MISSING_FILE", path, f"theme.json names {path}, which isn't in the package.")
        for path in referenced_files(manifest) if path not in files
    ]
    look = manifest.get("look")
    if look is not None and look not in preset_ids(files):
        errors += [theme_error("UNKNOWN_DEFAULT_LOOK", f"{MANIFEST_PATH}#look", f'"look" names "{echo(look)}", which isn\'t a preset.')]
    if STYLE_PATH in files:
        # utf-8-sig, as theme.json is read: a leading BOM is not part of the first selector.
        errors += css_scope_errors(files[STYLE_PATH].data.decode("utf-8-sig"), manifest["htmlClass"])
    return errors


def _paint_errors(block: Mapping[str, Any], files: Mapping[str, ThemeFile]) -> List[ThemeError]:
    """``fromLooks``: the prepaint table's ``defaults`` look paints the phone's ground and its text.

    A table that isn't a JSON object is refused once, by ``_prepaint_errors``.
    """
    if block.get(MOBILE_FROM_LOOKS) is not True:
        return []
    if PREPAINT_PATH not in files:
        return [theme_error(MOBILE_CANNOT_PAINT, PREPAINT_PATH, (
            f'"mobile.fromLooks" paints a phone from {PREPAINT_PATH}, which isn\'t in the package. '
            'Add the table, or give "mobile.tokens" instead.'
        ))]
    table, error = parse_json(files[PREPAINT_PATH].data, PREPAINT_PATH)
    if error is not None or not isinstance(table, dict):
        return []
    # No stored look: the web paints the defaults (boot.ts :164), and nothing at all from a table they don't fit.
    missing = missing_tokens(paint_look(table, None))
    if not missing:
        return []
    return [theme_error(MOBILE_CANNOT_PAINT, f"{PREPAINT_PATH}#{PREPAINT_DEFAULTS}", _cannot_paint(table, missing))]


def _cannot_paint(table: Mapping[str, Any], missing: List[str]) -> str:
    """Why the defaults paint no phone: they don't fit the table (the web drops it whole), or they lack tokens."""
    if boot_table(table) is None:
        return (
            f'{PREPAINT_PATH}: "{PREPAINT_DEFAULTS}" doesn\'t name an option of every knob table and a cell of every '
            'pair table, with every number rule\'s value in range, so the web paints nothing from the table and a '
            'phone can\'t draw the theme. Fix the defaults, or give "mobile.tokens" instead.'
        )
    return (
        f'{PREPAINT_PATH}: the default look paints no {" or ".join(missing)}, so a phone can\'t draw the theme '
        'from its looks. Give the default choices those tokens, or give "mobile.tokens" instead.'
    )


def _mascot_file_errors(
    manifest: Mapping[str, Any], block: Mapping[str, Any], files: Mapping[str, ThemeFile],
) -> List[ThemeError]:
    """``clips``: every state's .mp4 and .png; ``stills``: every state's .png (PB-36). No .webm: a phone plays mp4."""
    mode = block.get(MASCOT_KEY, MASCOT_NONE)
    states = (manifest.get(MASCOT_KEY) or {}).get(STATES_KEY) or []
    suffixes = MOBILE_MASCOT_FILES.get(mode, ())
    needed = [f"{MASCOT_FOLDER}/{state}.{suffix}" for state in states for suffix in suffixes]
    return [
        theme_error("MISSING_FILE", path, f'"mobile.mascot" "{mode}" needs {path}, which isn\'t in the package.')
        for path in needed if path not in files
    ]


def _mobile_link_errors(manifest: Mapping[str, Any], files: Mapping[str, ThemeFile]) -> List[ThemeError]:
    """A ``mobile`` block a phone draws without the theme's code: its colours paint, its mascot files exist."""
    block = manifest.get(MOBILE_FIELD)
    if not isinstance(block, dict):
        return []
    return _paint_errors(block, files) + _mascot_file_errors(manifest, block, files)


def _content_errors(files: Mapping[str, ThemeFile]) -> Tuple[Optional[Dict[str, Any]], List[ThemeError]]:
    manifest, errors = _manifest(files)
    _, look_errors = look_file_errors({path: item.data for path, item in files.items()})
    errors = errors + _code_errors(files) + look_errors + _prepaint_errors(files)
    if manifest is not None:
        errors += _manifest_link_errors(manifest, files) + _mobile_link_errors(manifest, files)
    return manifest, errors


def read_theme_package(data: bytes) -> ThemePackage:
    """Read and check a .autotheme package; :class:`ThemePackageError` says why one is refused."""
    with _open_zip(data) as zf:
        files = _read_files(zf, _checked_members(zf))
    manifest, errors = _content_errors(files)
    if errors or manifest is None:
        raise ThemePackageError(errors)
    rows = file_manifest(files)
    return ThemePackage(manifest=manifest, files=files, file_manifest=rows, digest=digest_of_manifest(rows))
