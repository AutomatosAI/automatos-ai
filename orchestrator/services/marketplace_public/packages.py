"""PRD-270 US-270-07a (K10) — the public read of the Marketplace's packages.

``services.marketplace_packages.list_packages``, less what a brand-new workspace would not be shown, in
the member Packages tab's order (showcased first, then by name; the slug as the last tie). The one rule
the member route applies per workspace is Socials (``api/marketplace.py`` ``_visible_packages``,
PRD-251B US-B106): the Socials package is listed only while Socials is on. A brand-new workspace has
no ``socials`` setting, so it takes ``SOCIALS_ENABLED_DEFAULT``; the visitor's answer is therefore the
platform's master switch AND that default (``modules.socials.settings``), and a master switch that can't
be read hides it (``socials_master_enabled`` fails closed).

The switch is read before the package query, so a request never holds the session's connection while
``read_system_setting`` asks the pool for a second one.

A package has no creator: its publisher is ``Automatos``. Its required apps are its setup manifest's
``required_connects`` app names, as slugs, and its members only their type and name (``None`` for a
member that has none; its ``ref`` is never answered): never the setup manifest's questions, guide steps
or report templates, nor the matching signals or the row's id.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from core.models.marketplace_packages import MEMBER_TYPES
from modules.socials.settings import SOCIALS_PACKAGE_SLUG, parse_workspace_socials, socials_master_enabled
from services.marketplace_packages import list_packages
from services.marketplace_public.fields import (
    PACKAGE_FIELDS, PACKAGE_MEMBER_FIELDS, allowed_item, display_item, text_or_none,
)

REQUIRED_CONNECTS = "required_connects"
APP_NAME = "app_name"


def socials_hidden_for_a_new_workspace() -> bool:
    """Whether a brand-new workspace (no ``socials`` setting) would be shown no Socials package."""
    return not (socials_master_enabled() and parse_workspace_socials({}).enabled)


def _shown(packages: Iterable[Any], socials_hidden: bool) -> List[Any]:
    if not socials_hidden:
        return list(packages)
    return [package for package in packages if package.slug != SOCIALS_PACKAGE_SLUG]


def _required_apps(manifest: Any) -> List[Any]:
    connects = manifest.get(REQUIRED_CONNECTS) if isinstance(manifest, dict) else None
    if not isinstance(connects, list):
        return []
    return [connect.get(APP_NAME) for connect in connects if isinstance(connect, dict)]


def _member(member: Any) -> Optional[Dict[str, Any]]:
    """A member as ``{type, name}`` (``name`` ``None`` when it has none), or ``None`` for one of no known type."""
    if not isinstance(member, dict) or member.get("type") not in MEMBER_TYPES:
        return None
    return allowed_item(PACKAGE_MEMBER_FIELDS, type=member["type"], name=text_or_none(member.get("name")))


def _members(members: Any) -> List[Dict[str, Any]]:
    if not isinstance(members, list):
        return []
    return [shown for shown in (_member(member) for member in members) if shown is not None]


def _item(package: Any) -> Dict[str, Any]:
    return display_item(
        PACKAGE_FIELDS, key=package.slug, name=package.name, description=package.description,
        tags=package.vertical_tags, required_apps=_required_apps(package.setup_manifest),
        members=_members(package.members),
    )


def read_packages(db: Session, limit: int, offset: int) -> List[Dict[str, Any]]:
    """Up to ``limit`` of the packages a brand-new workspace is shown, from ``offset``, each as its public item."""
    socials_hidden = socials_hidden_for_a_new_workspace()
    shown = sorted(
        _shown(list_packages(db), socials_hidden),
        key=lambda package: (not bool(package.showcase), (package.name or "").lower(), package.slug),
    )
    return [_item(package) for package in shown[offset:offset + limit]]


__all__ = ["read_packages", "socials_hidden_for_a_new_workspace"]
