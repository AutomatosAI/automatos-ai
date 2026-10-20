"""PRD-261 M-04 (host 1.1.0, plan §2.2.11) — an install's theme as data for the phone.

``GET /api/themes/mobile/{install_id}`` (``api/themes_mobile.py``) answers what a phone needs to
draw the theme a member chose without loading any of its code::

    {"schema": 1, "installId", "themeId", "version",
     "mobile": null | {"source": "tokens" | "fromLooks", "tone": "light" | "dark" | null,
                       "tokens": {"light": {...} | null, "dark": {...} | null},
                       "fonts": {"sans" | "serif" | "mono": "<files URL>"},
                       "mascot": {"mode": "clips" | "stills" | "none",
                                  "states": [{"name", "clip": "<files URL>" | null, "still": "<files URL>" | null}]}}}

* **Who.** The install and the caller are matched as the boot route matches them (``serve.boot_target``):
  membership of the install's workspace, never ``ctx.workspace_id``.
* **When ``mobile`` is null.** The checked ``theme.json`` has no ``mobile`` block, the caller may not
  run the package (private code without their opt-in, ``serve.may_run``), it is revoked
  (``revocations.is_disabled``), or a ``fromLooks`` block's ``boot/prepaint.json`` can't be read
  (missing, tampered or refused on read; logged), so nothing can be painted.
* **``tokens``.** The block's own schemes, exactly as theme.json writes them (``H S% L%`` or
  ``#rrggbb``; the phone converts them with look-kit's ``hsl.ts``), and ``tone`` null: the phone
  follows the device. A scheme the block leaves out is null.
* **``fromLooks``.** The member's active look for the theme in the install's workspace
  (``looks.read_looks``, carried to the current schema), else the prepaint table's ``defaults`` (the
  web boot script's fallback), projected through the table (``mobile_tokens.project_look``: Studio
  token names only, values as the table writes them) and placed under the look's tone
  (``mobile_tokens.look_tone``); the other scheme is null. A look store that can't be read paints
  the defaults and is logged, as the boot paints the default look.
* **Files.** Font and mascot URLs are ``serve.module_url(digest, path)``, served by
  ``GET /api/themes/files/...`` under the same membership. A URL is given only for a path the
  package's file manifest lists, never for a JS or CSS file. A mascot clip is the state's ``.mp4``
  (``.webm`` is the web's), a still its ``.png``; ``stills`` gives no clip and ``none`` no states.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from core.models.themes import ThemePackage
from services.themes.contract import MOBILE_FONT_ROLES
from services.themes.css_value import PREPAINT_PATH
from services.themes.look_schema import SCHEMA_PATH as LOOK_SCHEMA_PATH
from services.themes.looks import LooksError, read_looks, resolve_look_source
from services.themes.membership import Membership
from services.themes.mobile_manifest import MASCOT_NONE, MOBILE_FIELD, MOBILE_SCHEMES
from services.themes.mobile_tokens import look_tone, project_look
from services.themes.revocations import is_disabled
from services.themes.serve import BootTarget, FileReader, ThemeFileMissing, may_run, module_url, theme_prepaint

logger = logging.getLogger(__name__)

MOBILE_ANSWER_SCHEMA = 1
SOURCE_TOKENS = "tokens"
SOURCE_LOOKS = "fromLooks"
MASCOT_CLIPS = "clips"
MASCOT_STILLS = "stills"
MASCOT_DIR = "assets/mascot/"
CLIP_SUFFIX = ".mp4"
STILL_SUFFIX = ".png"
# The phone never receives a code URL (App Store guideline 2.5.2): no module, no style sheet.
CODE_SUFFIXES = (".js", ".mjs", ".css")
CODE_TYPES = frozenset({"text/javascript", "application/javascript", "text/css"})


def _object(value: Any) -> Mapping[str, Any]:
    """``value`` when it is a JSON object, else an empty one."""
    return value if isinstance(value, dict) else {}


def _is_code(path: str, row: Mapping[str, Any]) -> bool:
    listed_type = str(row.get("type", "")).split(";")[0].strip().lower()
    return path.lower().endswith(CODE_SUFFIXES) or listed_type in CODE_TYPES


def _file_url(package: ThemePackage, path: str) -> Optional[str]:
    """The files-route URL of ``path`` when the package lists it and it is not code, else None (logged)."""
    row = _object(_object(package.file_manifest).get(path))
    if not row or _is_code(path, row):
        logger.error("Package %s's mobile block names %r, which it doesn't list as a data file: no URL", package.id, path)
        return None
    return module_url(package.digest, path)


def _fonts(package: ThemePackage, block: Mapping[str, Any]) -> Dict[str, str]:
    """``{role: URL}`` for every font role the block declares whose file the package lists."""
    declared = _object(block.get("fonts"))
    urls = {role: _file_url(package, path) for role, path in declared.items() if role in MOBILE_FONT_ROLES and isinstance(path, str)}
    return {role: url for role, url in urls.items() if url is not None}


def _state(package: ThemePackage, mode: str, name: str) -> Dict[str, Any]:
    clip = _file_url(package, f"{MASCOT_DIR}{name}{CLIP_SUFFIX}") if mode == MASCOT_CLIPS else None
    return {"name": name, "clip": clip, "still": _file_url(package, f"{MASCOT_DIR}{name}{STILL_SUFFIX}")}


def _mascot(package: ThemePackage, block: Mapping[str, Any]) -> Dict[str, Any]:
    """The mode, and each top-level ``mascot.states`` entry's clip and still URLs (none for ``none``)."""
    mode = block.get("mascot", MASCOT_NONE)
    if mode not in (MASCOT_CLIPS, MASCOT_STILLS):
        return {"mode": MASCOT_NONE, "states": []}
    states = _object((package.manifest or {}).get("mascot")).get("states")
    names = [name for name in states if isinstance(name, str)] if isinstance(states, list) else []
    return {"mode": mode, "states": [_state(package, mode, name) for name in names]}


def _block_colours(block: Mapping[str, Any]) -> Dict[str, Any]:
    """A ``tokens`` block's schemes as written; ``tone`` null, so the phone follows the device."""
    schemes = _object(block.get(SOURCE_TOKENS))
    tokens = {scheme: dict(schemes[scheme]) if isinstance(schemes.get(scheme), dict) else None for scheme in MOBILE_SCHEMES}
    return {"source": SOURCE_TOKENS, "tone": None, "tokens": tokens}


def _stored_active_look(db: Any, membership: Membership, target: BootTarget) -> Optional[Dict[str, Any]]:
    """The member's active look for the theme in the install's workspace, carried to its schema; None for none."""
    install, package = target.install, target.package
    if membership.user_pk is None or LOOK_SCHEMA_PATH not in _object(package.file_manifest):
        return None
    try:
        source = resolve_look_source(db, install.workspace_id, install.theme_id)
    except LooksError as error:
        logger.error("Install %s's looks can't be read (%s): the phone gets the default look", install.id, error.detail)
        return None
    active = read_looks(db, install.workspace_id, membership.user_pk, source)["active"]
    return active if isinstance(active, dict) else None


def _look_colours(db: Any, membership: Membership, target: BootTarget, reader: FileReader) -> Optional[Dict[str, Any]]:
    """A ``fromLooks`` block's colours: the active look (else the defaults) projected under its tone."""
    package = target.package
    try:
        table = theme_prepaint(package, reader)
    except ThemeFileMissing:
        return None  # verified_file or theme_prepaint logged why
    if table is None:
        logger.error("Package %s's mobile block paints from looks, but it has no %s", package.id, PREPAINT_PATH)
        return None
    look = _stored_active_look(db, membership, target) or _object(table.get("defaults"))
    tone = look_tone(table, look, (package.manifest or {}).get("tone"))
    tokens = {scheme: (project_look(table, look) if scheme == tone else None) for scheme in MOBILE_SCHEMES}
    return {"source": SOURCE_LOOKS, "tone": tone, "tokens": tokens}


def _colours(
    db: Any, membership: Membership, target: BootTarget, block: Mapping[str, Any], reader: FileReader,
) -> Optional[Dict[str, Any]]:
    """The block's source, tone and schemes; None when it can't paint (the check on import allows one source)."""
    if SOURCE_TOKENS in block:
        return _block_colours(block)
    if block.get(SOURCE_LOOKS) is True:
        return _look_colours(db, membership, target, reader)
    logger.error("Package %s's mobile block has no source of colours", target.package.id)
    return None


def _mobile_block(db: Any, membership: Membership, target: BootTarget, reader: FileReader) -> Optional[Dict[str, Any]]:
    """The answer's ``mobile``, or None when the theme has no block, may not run, is revoked or can't paint."""
    package = target.package
    block = (package.manifest or {}).get(MOBILE_FIELD)
    if not isinstance(block, dict):
        return None
    if not may_run(db, membership, target.install, package) or is_disabled(db, package):
        return None
    colours = _colours(db, membership, target, block, reader)
    if colours is None:
        return None
    return {**colours, "fonts": _fonts(package, block), "mascot": _mascot(package, block)}


def mobile_theme(db: Any, membership: Membership, target: BootTarget, reader: FileReader) -> Dict[str, Any]:
    """The phone's answer for an install the caller may see (module docstring).

    Raises ``serve.ThemeStorageUnavailable`` when object storage fails (the route's 503).
    """
    install, package = target.install, target.package
    return {
        "schema": MOBILE_ANSWER_SCHEMA,
        "installId": str(install.id),
        "themeId": install.theme_id,
        "version": package.version,
        "mobile": _mobile_block(db, membership, target, reader),
    }


__all__ = ["MOBILE_ANSWER_SCHEMA", "mobile_theme"]
