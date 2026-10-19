"""PRD-261 C-02a — a theme's declared ``uses`` against the SDK names its bundle imports.

``theme.json``'s ``uses`` lists the SDK areas a theme reads (the Marketplace detail shows it,
C-04a). This check reads the code the browser runs (``dist/entry.js``, ``dist/chunks/*.js``)
and finds what it imports from ``@automatos/theme-sdk``:

* a named import or re-export (``import{useChatController as e}from"@automatos/theme-sdk"``)
  gives exact names; a name this host's SDK doesn't export is an **error**: the module can't
  link, so the theme can never load;
* a namespace import (``import * as sdk``) or a dynamic ``import()`` gives no list, so every SDK
  name found anywhere in the bundle text counts as used, and the check is a **warning** that
  names were not checked one by one;
* an area the code uses that ``uses`` doesn't declare, or a ``uses`` entry that is no SDK area,
  is a **warning**; a declared area the code never touches is reported, not warned.

An area is written in ``uses`` as its name; a page area (``pages/agents``) as ``pages-agents``
(its surface file's name) or ``pages.agents``, since ``uses`` entries hold no ``/``.

The SDK's names come from ``services/themes/data/sdk-surface.json`` (PA-18), generated from
``frontend/lib/theme-sdk/surface/*.json``: the running image has no ``frontend/`` folder.
``tests/test_prd261_sdk_surface_drift.py`` holds it to the checkout and regenerates it.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SDK_SURFACE_FILE = Path(__file__).resolve().parent / "data" / "sdk-surface.json"
BUNDLE_PREFIXES: Tuple[str, ...] = ("dist/entry.js", "dist/chunks/")
LEVEL_PASS, LEVEL_WARNING, LEVEL_ERROR = "pass", "warning", "error"

_SDK = r"""(?P<quote>["'])@automatos/theme-sdk(?P=quote)"""
NAMED_IMPORT = re.compile(r"\b(?:import|export)\s*(?:[A-Za-z_$][\w$]*\s*,\s*)?\{([^{}]*)\}\s*from\s*" + _SDK)
NAMESPACE_IMPORT = re.compile(r"\bimport\s*(?:[A-Za-z_$][\w$]*\s*,\s*)?\*\s*as\s*[A-Za-z_$][\w$]*\s*from\s*" + _SDK)
DYNAMIC_IMPORT = re.compile(r"\bimport\s*\(\s*" + _SDK + r"\s*\)")
STAR_EXPORT = re.compile(r"\bexport\s*\*\s*(?:as\s*[A-Za-z_$][\w$]*\s*)?from\s*" + _SDK)
SPECIFIER_NAME = re.compile(r"""\s*(?:type\s+)?(?:([A-Za-z_$][\w$]*)|(["'])([^"']*)\2)""")

SurfaceAreas = Mapping[str, Iterable[str]]

SURFACE_MISSING = (
    "This build has no readable SDK name list (services/themes/data/sdk-surface.json), "
    "so the SDK names the theme imports were not checked."
)
UNKNOWN_NAMES = "The theme imports names this host's SDK doesn't export, so its code can't load: {names}."
UNDECLARED = 'The code uses SDK areas theme.json\'s "uses" doesn\'t declare: {areas}.'
NOT_AREAS = '"uses" names entries that are not SDK areas: {entries}.'
WIDE_IMPORT = (
    "The code imports the SDK as a whole (a namespace or a dynamic import), so the names it uses "
    "were found by searching the bundle text, not checked one by one."
)


class SdkImports(NamedTuple):
    """What the bundle imports from the SDK: exact names, and whether it also imports it whole."""

    named: Set[str]
    wide: bool


@lru_cache(maxsize=1)
def sdk_surface() -> Optional[Dict[str, Tuple[str, ...]]]:
    """``{area: names}`` of this host's SDK, read once per process; None (logged) when unreadable."""
    try:
        document = json.loads(SDK_SURFACE_FILE.read_text(encoding="utf-8"))
        return {str(area): tuple(str(name) for name in names) for area, names in document["areas"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        logger.exception("[Themes] The SDK name list %s can't be read", SDK_SURFACE_FILE)
        return None


def uses_name(area: str) -> str:
    """How ``uses`` writes ``area``: ``pages/agents`` as ``pages-agents``."""
    return area.replace("/", "-")


def _aliases(surface: SurfaceAreas) -> Dict[str, str]:
    """Every accepted ``uses`` spelling → its area."""
    return {spelling: area for area in surface for spelling in (area, uses_name(area), area.replace("/", "."))}


def _specifier_names(block: str) -> Set[str]:
    """The exported names of ``a, b as c, "d" as e`` (the name before ``as``)."""
    names: Set[str] = set()
    for part in block.split(","):
        match = SPECIFIER_NAME.match(part)
        if match is not None and part.strip():
            names.add(match.group(1) or match.group(3))
    return names


def bundle_texts(files: Mapping[str, bytes]) -> Dict[str, str]:
    """The browser's code, ``{path: text}``, in path order."""
    return {
        path: files[path].decode("utf-8", errors="replace")
        for path in sorted(files) if path.startswith(BUNDLE_PREFIXES) and path.endswith(".js")
    }


def find_sdk_imports(texts: Iterable[str]) -> SdkImports:
    """The names imported from the SDK across ``texts``, and whether any imports it whole."""
    named: Set[str] = set()
    wide = False
    for text in texts:
        for match in NAMED_IMPORT.finditer(text):
            named |= _specifier_names(match.group(1))
        wide = wide or any(pattern.search(text) for pattern in (NAMESPACE_IMPORT, DYNAMIC_IMPORT, STAR_EXPORT))
    return SdkImports(named, wide)


def _names_in_text(texts: Iterable[str], names: Iterable[str]) -> Set[str]:
    """Which of ``names`` appear as whole identifiers anywhere in ``texts``."""
    wanted = sorted(set(names), key=len, reverse=True)
    if not wanted:
        return set()
    pattern = re.compile(r"(?<![\w$])(" + "|".join(re.escape(name) for name in wanted) + r")(?![\w$])")
    return {match.group(1) for text in texts for match in pattern.finditer(text)}


def _used_names(texts: List[str], imports: SdkImports, owner: Mapping[str, str]) -> Set[str]:
    known = {name for name in imports.named if name in owner}
    return known | _names_in_text(texts, owner) if imports.wide else known


def _declared(manifest: Mapping[str, Any], aliases: Mapping[str, str]) -> Tuple[Set[str], List[str]]:
    """The areas ``uses`` declares, and the entries that name no area."""
    entries = [entry for entry in manifest.get("uses") or [] if isinstance(entry, str)]
    return {aliases[entry] for entry in entries if entry in aliases}, [entry for entry in entries if entry not in aliases]


def _problems(unknown: List[str], undeclared: List[str], not_areas: List[str], wide: bool) -> List[str]:
    problems = [UNKNOWN_NAMES.format(names=", ".join(unknown))] if unknown else []
    problems += [UNDECLARED.format(areas=", ".join(undeclared))] if undeclared else []
    problems += [NOT_AREAS.format(entries=", ".join(not_areas))] if not_areas else []
    return problems + ([WIDE_IMPORT] if wide else [])


def uses_check(manifest: Mapping[str, Any], files: Mapping[str, bytes],
               surface: Optional[SurfaceAreas] = None) -> Dict[str, Any]:
    """The ``uses`` entry of the scan report; ``surface`` defaults to this host's SDK names."""
    surface = sdk_surface() if surface is None else surface
    if surface is None:
        return {"level": LEVEL_WARNING, "status": "not_checked", "reason": SURFACE_MISSING}
    owner = {name: area for area, names in surface.items() for name in names}
    texts = list(bundle_texts(files).values())
    imports = find_sdk_imports(texts)
    used = _used_names(texts, imports, owner)
    used_areas = {owner[name] for name in used}
    declared, not_areas = _declared(manifest, _aliases(surface))
    unknown = sorted(name for name in imports.named if name not in owner)
    undeclared = sorted(uses_name(area) for area in used_areas - declared)
    problems = _problems(unknown, undeclared, not_areas, imports.wide)
    level = LEVEL_ERROR if unknown else (LEVEL_WARNING if problems else LEVEL_PASS)
    return {
        "level": level, "status": "checked", "problems": problems, "unknownNames": unknown,
        "usedAreas": sorted(uses_name(area) for area in used_areas), "usedNames": sorted(used),
        "undeclaredAreas": undeclared, "notAreas": not_areas,
        "declaredUnused": sorted(uses_name(area) for area in declared - used_areas), "wideImport": imports.wide,
    }
