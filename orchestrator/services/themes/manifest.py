"""PRD-261 §4.2 — a theme's ``theme.json``, schema ``automatos-theme/v1``, checked field by field.

Required: ``schema``, ``id`` (``publisher/slug``), ``name``, ``version`` (``MAJOR.MINOR.PATCH``),
``engines`` (``{automatos, sdk}``, each exact or caret, ``*`` refused, and met by this host),
``base``, ``htmlClass`` (the pattern, and no reserved name: ``play`` only for ``automatos/play``,
PA-8), ``minWidth``, ``tone`` and ``slots`` (known, versioned slots that include
``lookEditor@1`` and ``styleSheet@1``, PA-14).

Optional: ``description``; ``pages`` (only the Q3 routes, each ``{mode: own|clone,
keepCorePage?, clonedFrom?}``, a clone naming one of its route's ``PAGE_SOURCES`` files and that
file's sha256; host-owned pages refused); ``routes`` (theme-only pages at ``/x/<route>``);
``nav`` (``{id, label, path}``, a ``/x/`` path naming a declared route); ``uses`` (SDK areas);
``pricing`` (a label only, Q12); ``previews`` and ``fonts`` (package files); ``brand``
(``{name?, wordmark?}``); ``look`` (the default preset's id); ``mascot`` (``null`` or
``{states}``); ``mobile`` (M-01, host 1.1.0: the data a phone draws the theme from, checked by
``services/themes/mobile_manifest.py``; its fonts are package files). Every other field is refused.

The TypeScript twin (``frontend/lib/theme-host/package-manifest.ts``, A-06b) is held to the
same corpus (``tests/fixtures/themes/cases.json``, ``manifests``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from services.themes.contract import (
    HOST_OWNED_ROUTES, HTML_CLASS_OWNERS, HTML_CLASS_PATTERN, KNOWN_SLOTS, MANIFEST_SCHEMA, PAGE_MODES,
    PAGE_SCOPE, PAGE_SOURCES, REQUIRED_SLOTS, RESERVED_HTML_CLASSES, SDK_VERSION, THEME_BASES,
    THEME_HOST_VERSION, THEME_TONES, ThemeError, echo, is_whole_number, plain_text_problem, theme_error,
)
from services.themes.mobile_manifest import mobile_errors, mobile_font_files

MANIFEST_ERROR_CODES: Tuple[str, ...] = (
    "NOT_AN_OBJECT", "UNKNOWN_FIELD", "MISSING_FIELD", "BAD_SCHEMA", "BAD_ID", "BAD_TEXT", "BAD_VERSION",
    "BAD_ENGINES", "ENGINES_MISMATCH", "BAD_HTML_CLASS", "RESERVED_HTML_CLASS", "BAD_BASE", "BAD_MIN_WIDTH",
    "BAD_TONE", "BAD_SLOTS", "UNKNOWN_SLOT", "MISSING_SLOT", "BAD_PAGE", "BAD_PAGE_MODE", "BAD_KEEP_CORE_PAGE",
    "BAD_CLONED_FROM", "CLONED_FROM_OUT_OF_SCOPE", "PAGE_OUT_OF_SCOPE", "HOST_OWNED_PAGE", "BAD_ROUTES", "BAD_NAV",
    "BAD_USES", "BAD_ASSET_PATH", "BAD_BRAND", "BAD_LOOK_ID", "BAD_MASCOT", "BAD_MOBILE",
)
REQUIRED_FIELDS: Tuple[str, ...] = (
    "schema", "id", "name", "version", "engines", "base", "htmlClass", "minWidth", "tone", "slots",
)
OPTIONAL_FIELDS: Tuple[str, ...] = (
    "description", "pages", "routes", "nav", "uses", "pricing", "previews", "fonts", "brand", "look", "mascot",
    "mobile",
)
ENGINE_FIELDS: Tuple[str, ...] = ("automatos", "sdk")
PAGE_FIELDS: Tuple[str, ...] = ("mode", "keepCorePage", "clonedFrom")
CLONED_FROM_FIELDS: Tuple[str, ...] = ("path", "sha")
NAV_FIELDS: Tuple[str, ...] = ("id", "label", "path")
BRAND_FIELDS: Tuple[str, ...] = ("name", "wordmark")
MASCOT_FIELDS: Tuple[str, ...] = ("states",)

ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,38}/[a-z0-9][a-z0-9-]{0,62}")
VERSION_PATTERN = re.compile(r"(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})")
RANGE_PATTERN = re.compile(r"(\^?)([0-9]{1,6})\.([0-9]{1,6})\.([0-9]{1,6})")
ROUTE_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,31}(?:/[a-z0-9][a-z0-9-]{0,31}){0,3}")
NAV_ID_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,31}")
NAV_PATH_PATTERN = re.compile(r"/(?:[a-z0-9][a-z0-9-]{0,63}(?:/[a-z0-9][a-z0-9-]{0,63}){0,7})?")
PARAM_PATTERN = re.compile(r"[a-z][A-Za-z0-9_-]{0,31}")
USES_PATTERN = re.compile(r"[a-z][A-Za-z0-9.-]{0,63}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
STATE_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,31}")
LOOK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
ASSET_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
PREVIEW_PATTERN = re.compile(rf"assets/previews/{ASSET_NAME}\.png")
FONT_PATTERN = re.compile(rf"assets/fonts/{ASSET_NAME}\.woff2")
WORDMARK_PATTERN = re.compile(rf"assets/brand/{ASSET_NAME}\.(?:png|webp)")
X_ROUTE_PREFIX = "/x/"

NAME_MAX = 60
DESCRIPTION_MAX = 280
LABEL_MAX = 40
MIN_WIDTH_MAX = 3840
MAX_ROUTES = 16
MAX_NAV = 16
MAX_KEEP = 16
MAX_USES = 64
MAX_FILES_LISTED = 16
MAX_STATES = 32


class ManifestResult(NamedTuple):
    """The checked manifest (a copy of the input) when ``errors`` is empty, else None."""

    manifest: Optional[Dict[str, Any]]
    errors: List[ThemeError]


def _error(code: str, path: str, message: str) -> List[ThemeError]:
    return [theme_error(code, path, message)]


def _unknown(record: Mapping[str, Any], known: Tuple[str, ...], where: str) -> List[ThemeError]:
    """UNKNOWN_FIELD for every key outside ``known``; ``where`` is the record's dotted path."""
    return [
        theme_error("UNKNOWN_FIELD", f"{where}.{echo(key)}" if where else echo(key),
                    f'"{echo(key)}" isn\'t part of {where or "theme.json"}. Remove it.')
        for key in record if key not in known
    ]


def _is_string_list(value: Any, pattern: "re.Pattern[str]", max_items: int) -> bool:
    """A list of at most ``max_items`` different strings, each matching ``pattern``."""
    if not isinstance(value, list) or len(value) > max_items:
        return False
    strings = all(isinstance(item, str) and pattern.fullmatch(item) is not None for item in value)
    return strings and len(set(value)) == len(value)


def _text_errors(value: Any, field: str, max_chars: int) -> List[ThemeError]:
    problem = plain_text_problem(value, field, max_chars)
    return _error("BAD_TEXT", field, problem) if problem else []


def satisfies_range(range_text: Any, host_version: str) -> bool:
    """True when an exact ``X.Y.Z`` or a caret ``^X.Y.Z`` accepts ``host_version`` (lib/theme-host/engines.ts)."""
    match = RANGE_PATTERN.fullmatch(range_text) if isinstance(range_text, str) else None
    host = RANGE_PATTERN.fullmatch(host_version)
    if match is None or host is None:
        return False
    wanted = tuple(int(part) for part in match.groups()[1:])
    have = tuple(int(part) for part in host.groups()[1:])
    if not match.group(1):
        return wanted == have
    return wanted[0] == have[0] and wanted[1:] <= have[1:]


def _engines_errors(engines: Any) -> List[ThemeError]:
    if not isinstance(engines, dict):
        return _error("BAD_ENGINES", "engines", '"engines" must be {"automatos": "^1.0.0", "sdk": "^1.0.0"}.')
    errors = _unknown(engines, ENGINE_FIELDS, "engines")
    hosts = {"automatos": THEME_HOST_VERSION, "sdk": SDK_VERSION}
    for field in ENGINE_FIELDS:
        value, where = engines.get(field), f"engines.{field}"
        if not isinstance(value, str) or RANGE_PATTERN.fullmatch(value) is None:
            errors += _error("BAD_ENGINES", where, f'"{where}" must be an exact version or a caret range such as ^1.0.0; "*" is refused.')
        elif not satisfies_range(value, hosts[field]):
            errors += _error("ENGINES_MISMATCH", where, f'"{where}" asks for {echo(value)}, and this host is {hosts[field]}.')
    return errors


def _identity_errors(document: Mapping[str, Any]) -> List[ThemeError]:
    errors: List[ThemeError] = []
    if document.get("schema") != MANIFEST_SCHEMA:
        errors += _error("BAD_SCHEMA", "schema", f'"schema" must be "{MANIFEST_SCHEMA}".')
    theme_id = document.get("id")
    if not isinstance(theme_id, str) or ID_PATTERN.fullmatch(theme_id) is None:
        errors += _error("BAD_ID", "id", '"id" must be publisher/slug in lower-case letters, digits and dashes.')
    version = document.get("version")
    if not isinstance(version, str) or VERSION_PATTERN.fullmatch(version) is None:
        errors += _error("BAD_VERSION", "version", '"version" must be MAJOR.MINOR.PATCH, such as 1.0.0.')
    errors += _text_errors(document.get("name"), "name", NAME_MAX)
    if "description" in document:
        errors += _text_errors(document["description"], "description", DESCRIPTION_MAX)
    return errors


def _html_class_errors(document: Mapping[str, Any]) -> List[ThemeError]:
    html_class = document.get("htmlClass")
    if not isinstance(html_class, str) or HTML_CLASS_PATTERN.fullmatch(html_class) is None:
        return _error("BAD_HTML_CLASS", "htmlClass", (
            '"htmlClass" must be 3 to 32 lower-case letters, digits or dashes, starting with a letter.'
        ))
    owner = HTML_CLASS_OWNERS.get(html_class)
    if html_class in RESERVED_HTML_CLASSES and (owner is None or owner != document.get("id")):
        return _error("RESERVED_HTML_CLASS", "htmlClass", f'"htmlClass" "{html_class}" is reserved by the host.')
    return []


def _layout_errors(document: Mapping[str, Any]) -> List[ThemeError]:
    errors: List[ThemeError] = []
    if document.get("base") not in THEME_BASES:
        errors += _error("BAD_BASE", "base", f'"base" must be one of {", ".join(THEME_BASES)}.')
    width = document.get("minWidth")
    if not is_whole_number(width) or not 0 <= width <= MIN_WIDTH_MAX:
        errors += _error("BAD_MIN_WIDTH", "minWidth", f'"minWidth" must be a whole number of pixels from 0 to {MIN_WIDTH_MAX}.')
    if document.get("tone") not in THEME_TONES:
        errors += _error("BAD_TONE", "tone", f'"tone" must be one of {", ".join(THEME_TONES)}.')
    return errors


def _slot_errors(slots: Any) -> List[ThemeError]:
    if not isinstance(slots, list) or not all(isinstance(slot, str) for slot in slots) or len(set(slots)) != len(slots):
        return _error("BAD_SLOTS", "slots", '"slots" must be a list of different slot names, such as "frame@1".')
    unknown = [slot for slot in slots if slot not in KNOWN_SLOTS]
    errors = [theme_error("UNKNOWN_SLOT", "slots", f'"{echo(slot)}" is not a slot this host serves.') for slot in unknown]
    missing = [slot for slot in REQUIRED_SLOTS if slot not in slots]
    return errors + [
        theme_error("MISSING_SLOT", "slots", f'Every theme ships "{slot}": add it to "slots" and register it.')
        for slot in missing
    ]


def _is_host_owned(route: str) -> bool:
    return any(route == owned or route.startswith(f"{owned}/") for owned in HOST_OWNED_ROUTES)


def _cloned_from_errors(route: str, entry: Mapping[str, Any], where: str) -> List[ThemeError]:
    cloned = entry.get("clonedFrom")
    if entry.get("mode") != "clone":
        return _error("BAD_CLONED_FROM", where, f'"{where}" names "clonedFrom" only for mode "clone".') if cloned is not None else []
    if not isinstance(cloned, dict) or set(cloned) != set(CLONED_FROM_FIELDS):
        return _error("BAD_CLONED_FROM", f"{where}.clonedFrom", f'"{where}.clonedFrom" must be {{"path", "sha"}}.')
    if not isinstance(cloned["sha"], str) or SHA256_PATTERN.fullmatch(cloned["sha"]) is None:
        return _error("BAD_CLONED_FROM", f"{where}.clonedFrom.sha", f'"{where}.clonedFrom.sha" must be the file\'s sha256 in hex.')
    if cloned["path"] not in PAGE_SOURCES[route]:
        return _error("CLONED_FROM_OUT_OF_SCOPE", f"{where}.clonedFrom.path", (
            f'"{where}.clonedFrom.path" must be one of {", ".join(PAGE_SOURCES[route])}.'
        ))
    return []


def _page_entry_errors(route: str, entry: Any) -> List[ThemeError]:
    where = f"pages.{route}"
    if not isinstance(entry, dict):
        return _error("BAD_PAGE", where, f'"{where}" must be an object with a "mode".')
    errors = _unknown(entry, PAGE_FIELDS, where)
    if entry.get("mode") not in PAGE_MODES:
        errors += _error("BAD_PAGE_MODE", f"{where}.mode", f'"{where}.mode" must be "own" or "clone".')
    if "keepCorePage" in entry and not _is_string_list(entry["keepCorePage"], PARAM_PATTERN, MAX_KEEP):
        errors += _error("BAD_KEEP_CORE_PAGE", f"{where}.keepCorePage", (
            f'"{where}.keepCorePage" must list up to {MAX_KEEP} different link parameter names.'
        ))
    return errors + _cloned_from_errors(route, entry, where)


def _page_errors(pages: Any) -> List[ThemeError]:
    if not isinstance(pages, dict):
        return _error("BAD_PAGE", "pages", '"pages" must be an object keyed by route.')
    errors: List[ThemeError] = []
    for route, entry in pages.items():
        if _is_host_owned(route):
            errors += _error("HOST_OWNED_PAGE", f"pages.{echo(route)}", f'"{echo(route)}" is a host page: a theme can\'t replace it.')
        elif route not in PAGE_SCOPE:
            errors += _error("PAGE_OUT_OF_SCOPE", f"pages.{echo(route)}", (
                f'"{echo(route)}" is not a page a theme may design: {", ".join(PAGE_SCOPE)}.'
            ))
        else:
            errors += _page_entry_errors(route, entry)
    return errors


def _nav_item_errors(item: Any, index: int, routes: List[str]) -> List[ThemeError]:
    where = f"nav[{index}]"
    if not isinstance(item, dict):
        return _error("BAD_NAV", where, f'"{where}" must be {{"id", "label", "path"}}.')
    errors = _unknown(item, NAV_FIELDS, where)
    if not isinstance(item.get("id"), str) or NAV_ID_PATTERN.fullmatch(item["id"]) is None:
        errors += _error("BAD_NAV", f"{where}.id", f'"{where}.id" must be 1 to 32 lower-case letters, digits or dashes.')
    errors += _text_errors(item.get("label"), f"{where}.label", LABEL_MAX)
    path = item.get("path")
    if not isinstance(path, str) or NAV_PATH_PATTERN.fullmatch(path) is None:
        return errors + _error("BAD_NAV", f"{where}.path", f'"{where}.path" must be a path in this app, such as /chat.')
    if path.startswith(X_ROUTE_PREFIX) and path[len(X_ROUTE_PREFIX):] not in routes:
        errors += _error("BAD_NAV", f"{where}.path", f'"{where}.path" goes to a /x/ route the theme doesn\'t declare.')
    return errors


def _route_and_nav_errors(document: Mapping[str, Any]) -> List[ThemeError]:
    routes = document.get("routes", [])
    errors: List[ThemeError] = []
    if not _is_string_list(routes, ROUTE_PATTERN, MAX_ROUTES):
        errors += _error("BAD_ROUTES", "routes", f'"routes" must list up to {MAX_ROUTES} different paths such as "conversations".')
        routes = []
    nav = document.get("nav", [])
    if not isinstance(nav, list) or len(nav) > MAX_NAV:
        return errors + _error("BAD_NAV", "nav", f'"nav" must be a list of up to {MAX_NAV} items.')
    ids = [item.get("id") for item in nav if isinstance(item, dict)]
    if len(set(map(str, ids))) != len(ids):
        errors += _error("BAD_NAV", "nav", 'Each "nav" item needs its own "id".')
    return errors + [error for index, item in enumerate(nav) for error in _nav_item_errors(item, index, routes)]


def _brand_errors(brand: Any) -> List[ThemeError]:
    if not isinstance(brand, dict):
        return _error("BAD_BRAND", "brand", '"brand" must be {"name", "wordmark"}.')
    errors = _unknown(brand, BRAND_FIELDS, "brand")
    if "name" in brand:
        errors += _text_errors(brand["name"], "brand.name", NAME_MAX)
    wordmark = brand.get("wordmark")
    if "wordmark" in brand and (not isinstance(wordmark, str) or WORDMARK_PATTERN.fullmatch(wordmark) is None):
        errors += _error("BAD_ASSET_PATH", "brand.wordmark", '"brand.wordmark" must be a .png or .webp in assets/brand/.')
    return errors


def _mascot_errors(mascot: Any) -> List[ThemeError]:
    if mascot is None:
        return []
    if not isinstance(mascot, dict):
        return _error("BAD_MASCOT", "mascot", '"mascot" must be null or {"states": [...]}.')
    errors = _unknown(mascot, MASCOT_FIELDS, "mascot")
    if not _is_string_list(mascot.get("states"), STATE_PATTERN, MAX_STATES):
        errors += _error("BAD_MASCOT", "mascot.states", f'"mascot.states" must list up to {MAX_STATES} different state names.')
    return errors


def _file_list_errors(document: Mapping[str, Any]) -> List[ThemeError]:
    errors: List[ThemeError] = []
    for field, pattern, shape in (("previews", PREVIEW_PATTERN, "assets/previews/*.png"), ("fonts", FONT_PATTERN, "assets/fonts/*.woff2")):
        if field in document and not _is_string_list(document[field], pattern, MAX_FILES_LISTED):
            errors += _error("BAD_ASSET_PATH", field, f'"{field}" must list up to {MAX_FILES_LISTED} different {shape} files.')
    return errors


def _extra_errors(document: Mapping[str, Any]) -> List[ThemeError]:
    errors = _file_list_errors(document)
    if "uses" in document and not _is_string_list(document["uses"], USES_PATTERN, MAX_USES):
        errors += _error("BAD_USES", "uses", f'"uses" must list up to {MAX_USES} different SDK area names.')
    if "pricing" in document:
        errors += _text_errors(document["pricing"], "pricing", LABEL_MAX)
    look = document.get("look")
    if "look" in document and (not isinstance(look, str) or LOOK_ID_PATTERN.fullmatch(look) is None):
        errors += _error("BAD_LOOK_ID", "look", '"look" must be the id of one of the theme\'s presets.')
    errors += _brand_errors(document["brand"]) if "brand" in document else []
    errors += mobile_errors(document) if "mobile" in document else []
    return errors + _mascot_errors(document.get("mascot"))


def validate_manifest(document: Any) -> ManifestResult:
    """Every reason ``document`` is not a valid theme.json; the manifest itself when there is none."""
    if not isinstance(document, dict):
        return ManifestResult(None, _error("NOT_AN_OBJECT", "", "theme.json must be a JSON object."))
    missing = [field for field in REQUIRED_FIELDS if field not in document]
    checks = _identity_errors(document) + _engines_errors(document.get("engines"))
    checks += _html_class_errors(document) + _layout_errors(document) + _slot_errors(document.get("slots"))
    checks += _page_errors(document["pages"]) if "pages" in document else []
    checks += _route_and_nav_errors(document) + _extra_errors(document)
    # A missing field is said once, as MISSING_FIELD, not again by its own check.
    errors = _unknown(document, REQUIRED_FIELDS + OPTIONAL_FIELDS, "") + [
        theme_error("MISSING_FIELD", field, f'theme.json needs "{field}".') for field in missing
    ] + [error for error in checks if error["path"].split(".")[0] not in missing]
    return ManifestResult(None if errors else dict(document), errors)


def referenced_files(manifest: Mapping[str, Any]) -> List[str]:
    """The package files a checked manifest names: its previews, fonts, wordmark and mobile fonts (M-01)."""
    wordmark = (manifest.get("brand") or {}).get("wordmark")
    named = [*manifest.get("previews", []), *manifest.get("fonts", []), *([wordmark] if wordmark else [])]
    return [*named, *mobile_font_files(manifest)]
