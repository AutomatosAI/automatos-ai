"""PRD-261 A-09a — a member's looks for one theme, kept on the server (Q5).

One ``theme_looks`` row per (workspace, user, theme id): the saved looks and the active one,
exactly as the look editor holds them (``frontend/lib/looks/storage.ts`` ``StoredLooks``).
They are keyed by the theme id (``publisher/slug``), never by a version, so an update or a
rollback keeps them.

* **Which schema.** The workspace's active install of that theme id names a package; its
  ``looks/schema.json`` (checked by ``look_schema.py``) is the one every look is held to.
  Builtin Play (``automatos/play``, PA-8) has no install until A-18: its schema is the
  committed ``data/play-look-schema.json``, mirrored from ``frontend/lib/looks/types.ts`` and
  ``catalogue.ts`` (``tests/test_prd261_play_schema_parity.py``) until A-16c ships it in the
  package. A theme id with neither is "not installed here": 404, also when another workspace
  has it, so a caller learns nothing about other workspaces.
* **Writes are strict.** ``PUT`` refuses any look the schema refuses, naming the field.
* **Reads carry looks over.** A look saved under an earlier version is fitted to the current
  schema on every read: a knob the schema no longer has is dropped, a knob it added or whose
  value it no longer accepts takes the knob's ``default``. A look that still doesn't fit (its
  id or name) is left out and counted in ``dropped``; the active look then falls back to the
  theme's default look (``defaultLook``). The stored row is never rewritten by a read, so a
  rollback finds the looks as they were saved.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import config
from core.models.themes import ThemeInstallation, ThemeLook, ThemePackage
from core.storage import get_s3_client
from services.themes.contract import ThemeError, echo, parse_json, theme_error
from services.themes.digest import sha256_hex
from services.themes.look_schema import (
    PRESET_SUFFIX, PRESETS_DIR, SCHEMA_PATH, LookSchema, preset_paths, validate_look, validate_look_schema,
)
from services.themes.manifest import ID_PATTERN

logger = logging.getLogger(__name__)

# PA-8: builtin Play's package id; its looks are kept under it, so A-18 finds them in place.
BUILTIN_PLAY_THEME_ID = "automatos/play"
# frontend/lib/looks/presets.ts DEFAULT_LOOK (Ember).
BUILTIN_PLAY_DEFAULT_LOOK = "ember"
PLAY_SCHEMA_FILE = Path(__file__).resolve().parent / "data" / "play-look-schema.json"

# A member's saved looks for one theme (the editor's own list has no cap; this keeps a row small).
MAX_SAVED_LOOKS = 100
BODY_FIELDS = ("looks", "active")
# Checked looks/schema.json documents, by the sha256 of the file (content-addressed, so safe
# to share between packages and workspaces).
SCHEMA_CACHE_SIZE = 64
THEMES_STORAGE_ROOT = "themes/"
MARKETPLACE_OWNER = "marketplace"

NOT_INSTALLED = "This theme isn't installed in this workspace."
# Both fields, always: a save replaces the whole list, so a missing "looks" never empties it.
BODY_SHAPE = 'Send {"looks": [...], "active": {...} or null}: both fields, every time.'
SCHEMA_UNAVAILABLE = "This theme's look settings can't be read right now. Try again in a minute."


class LooksError(Exception):
    """A refusal the route turns into ``{detail, errors}`` with ``status``."""

    def __init__(self, status: int, detail: str, errors: Optional[List[ThemeError]] = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.errors = errors or []


class LookSource(NamedTuple):
    """Where a theme's looks are checked: its schema and its default look's id."""

    theme_id: str
    schema: LookSchema
    default_look: Optional[str]


class StoredLooks(NamedTuple):
    """A member's looks as the current schema reads them."""

    looks: List[Dict[str, Any]]
    active: Optional[Dict[str, Any]]
    dropped: int


# ---------------------------------------------------------------------------
# The schema a theme's looks are held to
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def builtin_play_schema() -> LookSchema:
    """The committed schema of builtin Play; a broken file is a release bug, so it raises."""
    document, error = parse_json(PLAY_SCHEMA_FILE.read_bytes(), PLAY_SCHEMA_FILE.name)
    schema, errors = validate_look_schema(document) if error is None else (None, [error])
    if schema is None:
        raise RuntimeError(f"data/play-look-schema.json is not a valid look schema: {errors}")
    return schema


def _object_key(prefix: str, path: str) -> str:
    """The storage key of ``path`` in a package stored under ``prefix`` (a row this server wrote)."""
    parts = prefix.rstrip("/").split("/")
    if not prefix.startswith(THEMES_STORAGE_ROOT) or not prefix.endswith("/") or any(
        part in ("", ".", "..") for part in parts
    ):
        raise ValueError(f"Not a theme storage prefix: {prefix!r}")
    return f"{prefix}{path}"


def read_package_file(package: ThemePackage, path: str) -> bytes:
    """One stored file of ``package`` (blocking: call it from a plain-``def`` route)."""
    key = _object_key(package.storage_prefix, path)
    response = get_s3_client().get_object(Bucket=config.MARKETPLACE_S3_BUCKET, Key=key)
    return response["Body"].read()


class _SchemaCache:
    """A small bounded map of checked schemas by file sha256, safe across threadpool workers."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._items: Dict[str, LookSchema] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[LookSchema]:
        with self._lock:
            return self._items.get(key)

    def put(self, key: str, schema: LookSchema) -> None:
        with self._lock:
            if key not in self._items and len(self._items) >= self._size:
                self._items.pop(next(iter(self._items)))
            self._items[key] = schema


_schema_cache = _SchemaCache(SCHEMA_CACHE_SIZE)


def _stored_schema_bytes(package: ThemePackage, expected_sha: str) -> bytes:
    try:
        raw = read_package_file(package, SCHEMA_PATH)
    except Exception as exc:
        logger.exception("Could not read %s of theme package %s", SCHEMA_PATH, package.id)
        raise LooksError(503, SCHEMA_UNAVAILABLE) from exc
    if sha256_hex(raw) != expected_sha:
        logger.error("Stored %s of theme package %s does not match its file manifest", SCHEMA_PATH, package.id)
        raise LooksError(503, SCHEMA_UNAVAILABLE)
    return raw


def package_look_schema(package: ThemePackage) -> LookSchema:
    """``package``'s looks/schema.json, re-hashed against its file manifest and checked."""
    entry = (package.file_manifest or {}).get(SCHEMA_PATH)
    expected = entry.get("sha256") if isinstance(entry, dict) else None
    if not isinstance(expected, str):
        logger.error("Theme package %s has no %s in its file manifest", package.id, SCHEMA_PATH)
        raise LooksError(503, SCHEMA_UNAVAILABLE)
    cached = _schema_cache.get(expected)
    if cached is not None:
        return cached
    document, error = parse_json(_stored_schema_bytes(package, expected), SCHEMA_PATH)
    schema, errors = validate_look_schema(document) if error is None else (None, [error])
    if schema is None:
        logger.error("Theme package %s has an invalid %s: %s", package.id, SCHEMA_PATH, errors)
        raise LooksError(503, SCHEMA_UNAVAILABLE)
    _schema_cache.put(expected, schema)
    return schema


def package_default_look(package: ThemePackage) -> Optional[str]:
    """The manifest's ``look``, else the first preset by file name, else None."""
    named = (package.manifest or {}).get("look")
    if isinstance(named, str):
        return named
    presets = preset_paths(package.file_manifest or {})
    return presets[0][len(PRESETS_DIR):-len(PRESET_SUFFIX)] if presets else None


def _installed_package(db: Session, workspace_id: UUID, theme_id: str) -> Optional[ThemePackage]:
    """The package this workspace's active install of ``theme_id`` points at, if it may use it."""
    install = (
        db.query(ThemeInstallation)
        .filter(
            ThemeInstallation.workspace_id == workspace_id,
            ThemeInstallation.theme_id == theme_id,
            ThemeInstallation.uninstalled_at.is_(None),
        )
        .one_or_none()
    )
    if install is None:
        return None
    package = db.get(ThemePackage, install.package_id)
    if package is None or package.theme_id != theme_id:
        return None
    # No DB constraint stops an install naming another workspace's private package (W-B2 carry).
    if package.owner_type != MARKETPLACE_OWNER and package.workspace_id != workspace_id:
        return None
    return package


def resolve_look_source(db: Session, workspace_id: UUID, theme_id: str) -> LookSource:
    """The schema ``theme_id``'s looks are held to in this workspace; 404 when it isn't here."""
    if ID_PATTERN.fullmatch(theme_id or "") is None:
        raise LooksError(404, NOT_INSTALLED)
    package = _installed_package(db, workspace_id, theme_id)
    if package is not None:
        return LookSource(theme_id, package_look_schema(package), package_default_look(package))
    if theme_id == BUILTIN_PLAY_THEME_ID:
        return LookSource(theme_id, builtin_play_schema(), BUILTIN_PLAY_DEFAULT_LOOK)
    raise LooksError(404, NOT_INSTALLED)


# ---------------------------------------------------------------------------
# Checking a write, carrying a read over
# ---------------------------------------------------------------------------


def _at(where: str, errors: List[ThemeError]) -> List[ThemeError]:
    """``errors`` with ``where`` in front of each field path (``looks[2].glow``, ``active``)."""
    return [{**error, "path": f"{where}.{error['path']}" if error["path"] else where} for error in errors]


def _saved_look_errors(schema: LookSchema, looks: Any) -> List[ThemeError]:
    if not isinstance(looks, list):
        return [theme_error("LOOKS_NOT_A_LIST", "looks", '"looks" must be a list of looks.')]
    if len(looks) > MAX_SAVED_LOOKS:
        return [theme_error("TOO_MANY_LOOKS", "looks", f"You can keep at most {MAX_SAVED_LOOKS} looks per theme.")]
    errors: List[ThemeError] = []
    seen: set = set()
    for index, look in enumerate(looks):
        errors += _at(f"looks[{index}]", validate_look(schema, look))
        look_id = look.get("id") if isinstance(look, dict) else None
        if isinstance(look_id, str) and look_id in seen:
            errors.append(theme_error(
                "LOOK_DUPLICATE_ID", f"looks[{index}].id", f'Two saved looks share the id "{echo(look_id)}".',
            ))
        if isinstance(look_id, str):
            seen.add(look_id)
    return errors


def check_looks_body(schema: LookSchema, body: Any) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """``(looks, active)`` from a ``PUT`` body, or :class:`LooksError` 400 naming each bad field."""
    if not isinstance(body, dict) or any(field not in body for field in BODY_FIELDS):
        raise LooksError(400, BODY_SHAPE, [theme_error("BAD_BODY", "", BODY_SHAPE)])
    errors = [
        theme_error("UNKNOWN_FIELD", echo(str(key)), f'"{echo(str(key))}" isn\'t part of a looks update. Remove it.')
        for key in body if key not in BODY_FIELDS
    ]
    errors += _saved_look_errors(schema, body["looks"])
    active = body["active"]
    if active is not None:
        errors += _at("active", validate_look(schema, active))
    if errors:
        raise LooksError(400, errors[0]["message"], errors)
    return list(body["looks"]), active


def carry_look(schema: LookSchema, stored: Any) -> Optional[Dict[str, Any]]:
    """``stored`` fitted to ``schema``: dropped knobs left out, new or refused knobs at their default.

    None when it still doesn't fit (its id, name or sub), so the caller leaves it out.
    """
    if not isinstance(stored, dict):
        return None
    header = {"schema": schema.look_schema, **({"interface": schema.interface} if schema.interface else {})}
    text = {field: stored.get(field) for field in ("id", "name", "sub")}
    knobs = {name: stored.get(name, knob["default"]) for name, knob in schema.knobs.items()}
    candidate = {**header, **text, **knobs}
    refused = {error["path"] for error in validate_look(schema, candidate) if error["code"] == "LOOK_BAD_VALUE"}
    carried = {**candidate, **{name: schema.knobs[name]["default"] for name in refused if name in schema.knobs}}
    return carried if not validate_look(schema, carried) else None


def carry_looks(schema: LookSchema, looks: Any, active: Any) -> StoredLooks:
    """A stored row's looks as ``schema`` reads them; ``dropped`` counts what couldn't be carried."""
    saved = looks if isinstance(looks, list) else []
    kept: List[Dict[str, Any]] = []
    for look in saved:
        carried = carry_look(schema, look)
        if carried is not None and all(carried["id"] != other["id"] for other in kept):
            kept.append(carried)
    carried_active = carry_look(schema, active) if active is not None else None
    dropped = len(saved) - len(kept) + (1 if active is not None and carried_active is None else 0)
    return StoredLooks(kept, carried_active, dropped)


# ---------------------------------------------------------------------------
# The row
# ---------------------------------------------------------------------------


def _row(db: Session, workspace_id: UUID, user_id: int, theme_id: str) -> Optional[ThemeLook]:
    return (
        db.query(ThemeLook)
        .filter(ThemeLook.workspace_id == workspace_id, ThemeLook.user_id == user_id, ThemeLook.theme_id == theme_id)
        .one_or_none()
    )


def looks_response(source: LookSource, stored: StoredLooks) -> Dict[str, Any]:
    """The JSON both routes answer with."""
    return {
        "themeId": source.theme_id,
        "looks": stored.looks,
        "active": stored.active,
        "defaultLook": source.default_look,
        "dropped": stored.dropped,
    }


def read_looks(db: Session, workspace_id: UUID, user_id: int, source: LookSource) -> Dict[str, Any]:
    """This member's looks for the theme in this workspace (none saved: an empty list)."""
    row = _row(db, workspace_id, user_id, source.theme_id)
    if row is None:
        return looks_response(source, StoredLooks([], None, 0))
    return looks_response(source, carry_looks(source.schema, row.looks, row.active))


def _write(db: Session, workspace_id: UUID, user_id: int, theme_id: str, values: Mapping[str, Any]) -> None:
    row = _row(db, workspace_id, user_id, theme_id)
    if row is None:
        db.add(ThemeLook(workspace_id=workspace_id, user_id=user_id, theme_id=theme_id, **values))
    else:
        row.looks, row.active = values["looks"], values["active"]
    db.commit()


def save_looks(db: Session, workspace_id: UUID, user_id: int, source: LookSource, body: Any) -> Dict[str, Any]:
    """Check ``body`` against the theme's schema and store it for this member; answers as a read."""
    looks, active = check_looks_body(source.schema, body)
    values = {"looks": looks, "active": active}
    try:
        _write(db, workspace_id, user_id, source.theme_id, values)
    except IntegrityError:
        # Two saves raced to create the row: the other one won, so update it.
        db.rollback()
        _write(db, workspace_id, user_id, source.theme_id, values)
    return looks_response(source, StoredLooks(looks, active, 0))
