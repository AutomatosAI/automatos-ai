"""PRD-221 S1 — page manifest loader.

The checked-in contract at ``orchestrator/contracts/page-manifest.json`` tells
Auto what each frontend page is for, which entities it shows, which platform
actions apply there, and which quick prompts the UI may offer. It is validated
in CI (every action name must exist in the ActionRegistry; the frontend
contract test asserts routes exist in the app router and the generated TS
mirror is in sync). Consumers: the page-context preamble renderer, the tool
router's page-prior exposure, and the generated frontend mirror.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "contracts" / "page-manifest.json"


@dataclass(frozen=True)
class QuickPrompt:
    """One tappable prompt the UI may offer on a page."""

    text: str
    admin_only: bool = False


@dataclass(frozen=True)
class PageEntry:
    """One page's contract: identity, purpose, and what applies there."""

    key: str
    route: str
    title: str
    purpose: str
    entities: Tuple[str, ...] = ()
    tabs: Tuple[str, ...] = ()
    actions: Tuple[str, ...] = ()
    quick_prompts: Tuple[QuickPrompt, ...] = ()


def _parse_prompt(raw: Any) -> QuickPrompt:
    if isinstance(raw, str):
        return QuickPrompt(text=raw)
    return QuickPrompt(
        text=str(raw.get("text", "")),
        admin_only=bool(raw.get("admin_only", False)),
    )


@lru_cache(maxsize=1)
def _load() -> Tuple[PageEntry, ...]:
    with open(_MANIFEST_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    pages = []
    for raw in data.get("pages", []):
        pages.append(
            PageEntry(
                key=raw["key"],
                route=raw["route"],
                title=raw["title"],
                purpose=raw["purpose"],
                entities=tuple(raw.get("entities", ())),
                tabs=tuple(raw.get("tabs", ())),
                actions=tuple(raw.get("actions", ())),
                quick_prompts=tuple(_parse_prompt(q) for q in raw.get("quick_prompts", ())),
            )
        )
    return tuple(pages)


def all_pages() -> Tuple[PageEntry, ...]:
    """Every page entry, in manifest order."""
    return _load()


def get_page(key: Optional[str]) -> Optional[PageEntry]:
    """Entry for a manifest key, or None — unknown keys never raise."""
    if not key:
        return None
    for page in _load():
        if page.key == key:
            return page
    return None


def resolve_route(route: Optional[str]) -> Optional[str]:
    """Resolve a concrete route (including subpaths like ``/missions/123``) to a page key.

    Longest matching manifest route wins; unknown routes return None.
    """
    if not route or not route.startswith("/"):
        return None
    best: Optional[PageEntry] = None
    for page in _load():
        if route == page.route or route.startswith(page.route.rstrip("/") + "/"):
            if best is None or len(page.route) > len(best.route):
                best = page
    return best.key if best else None


def list_actions(key: str) -> Tuple[str, ...]:
    """Manifest action names for a page key; empty tuple when unknown."""
    page = get_page(key)
    return page.actions if page else ()
