"""PRD-261 A-07b — a member's opt-in to run one private theme version (Q6).

Unreviewed code runs in the member's browser with their session, so it runs only after
that member says yes to that exact package version: ``theme_opt_ins`` is keyed by
``(workspace, user, package)``. A new version is a new package row, so an update needs a
fresh opt-in without anything being cleared. A reviewed Marketplace package needs none.

``POST /api/themes/installs/{id}/opt-in`` takes ``{packageId}`` and answers 409 when that
is not the install's current package, so a dialog left open across an update can't opt
the member in to code they never saw.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.models.themes import ThemeOptIn, ThemePackage
from services.themes.choices import (
    CONFLICT,
    FORBIDDEN,
    ThemeInstallError,
    active_install,
    is_private,
    visible_package,
)

logger = logging.getLogger(__name__)

# PRD §6 / Q6: the words the opt-in dialog shows (A-07d reads them from here, never its own copy).
OPT_IN_WORDING = "This theme was not reviewed. Its code runs with your access to all your workspaces."
SUPERSEDED = "That version is no longer this theme's current one. Reload to see the version installed now."
NEEDS_MEMBER = "Opting in to a theme needs a signed-in member of this workspace."


def has_opted_in(db: Session, workspace_id: UUID, user_pk: Optional[int], package_id: UUID) -> bool:
    """True when this member opted in to this exact package in this workspace."""
    if user_pk is None:
        return False
    row = (
        db.query(ThemeOptIn.id)
        .filter_by(workspace_id=workspace_id, user_id=user_pk, package_id=package_id)
        .first()
    )
    return row is not None


def opt_in_state(db: Session, workspace_id: UUID, user_pk: Optional[int], package: ThemePackage) -> Dict[str, Any]:
    """``{required, wording, packageId}``: required until the member opts in to this package."""
    required = is_private(package) and not has_opted_in(db, workspace_id, user_pk, package.id)
    return {"required": required, "wording": OPT_IN_WORDING, "packageId": str(package.id)}


def _insert_opt_in(db: Session, workspace_id: UUID, user_pk: int, package_id: UUID) -> None:
    db.add(ThemeOptIn(workspace_id=workspace_id, user_id=user_pk, package_id=package_id))
    try:
        db.flush()
    except IntegrityError:
        # Another request recorded the same opt-in first: the unique index holds one row.
        # This request wrote nothing else, so rolling back loses nothing.
        db.rollback()
        logger.info("theme opt-in for package %s already recorded", package_id)


def record_opt_in(
    db: Session, workspace_id: UUID, user_pk: Optional[int], install_id: UUID, package_id: UUID,
) -> Dict[str, Any]:
    """Opt the member in to the install's current package; 409 for any other package."""
    if user_pk is None:
        raise ThemeInstallError(FORBIDDEN, NEEDS_MEMBER)
    install = active_install(db, workspace_id, install_id)
    if package_id != install.package_id:
        visible_package(db, workspace_id, package_id)  # another workspace's package: 404, not 409
        raise ThemeInstallError(CONFLICT, SUPERSEDED)
    package = visible_package(db, workspace_id, package_id)
    if is_private(package) and not has_opted_in(db, workspace_id, user_pk, package_id):
        _insert_opt_in(db, workspace_id, user_pk, package_id)
    return opt_in_state(db, workspace_id, user_pk, package)
