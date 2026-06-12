"""Variable resolution service (PRD-167 S3).

Resolves ``{{user.*}} / {{company.*}} / {{brand.*}} / {{date.*}}`` against the
requesting user's profile, the workspace business profile, the workspace brand kit and
the render-time clock.

Unresolved policy (PRD-167 S3): a *known* path that resolves empty is reported as
``unresolved`` (the caller surfaces a render-time error list — never a silent blank);
an *unknown* path (not in the catalog) is reported separately as an authoring error.

The context-building and path-resolution logic is pure (``build_context`` /
``resolve_paths``) and unit-testable without a database; :class:`VariableResolver` is
the thin DB-backed wrapper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from ..brand_kit import get_brand_kit
from .catalog import is_dynamic_path, is_known_path, DYNAMIC_PREFIX

logger = logging.getLogger(__name__)


@dataclass
class ResolvedVariables:
    values: Dict[str, str] = field(default_factory=dict)
    unresolved: List[str] = field(default_factory=list)  # known paths, empty value
    unknown: List[str] = field(default_factory=list)      # paths not in the catalog


def _long_date(now: datetime) -> str:
    # Avoid %-d (not portable to Windows); build the long form manually.
    return f"{now.strftime('%B')} {now.day}, {now.year}"


def _walk(data: Any, dotted_key: str) -> Any:
    """Walk a dotted key through nested dicts (for the ``data.*`` namespace)."""
    cur = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def build_context(
    user: Any,
    business_profile: Any,
    brand_kit: Dict[str, Any],
    now: datetime,
    extra_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the nested resolution context from already-fetched objects. Pure.

    ``extra_data`` populates the dynamic ``data.*`` namespace (caller-supplied
    per-generation values, e.g. from an agent's ``generate_document`` call).
    """
    name = ((getattr(user, "name", None) or "") if user else "").strip()
    first, _, last = name.partition(" ")
    user_ctx = {
        "name": name,
        "first_name": first,
        "last_name": last,
        "email": (getattr(user, "email", "") or "") if user else "",
        "username": (getattr(user, "username", "") or "") if user else "",
    }

    company_contact = brand_kit.get("company", {}) if isinstance(brand_kit, dict) else {}
    company_name = (
        (getattr(business_profile, "company_name", None) if business_profile else None)
        or company_contact.get("name")
        or brand_kit.get("name")
        or ""
    )
    domain = (getattr(business_profile, "domain", "") or "") if business_profile else ""
    company_ctx = {
        "name": company_name,
        "website": company_contact.get("website") or domain or "",
        "address": company_contact.get("address", ""),
        "email": company_contact.get("email", ""),
        "phone": company_contact.get("phone", ""),
    }

    brand_ctx = {
        "name": brand_kit.get("name") or company_name or "",
        "tagline": brand_kit.get("tagline", ""),
        "logo_url": brand_kit.get("logo_url", ""),
        "primary_color": brand_kit.get("primary_color", ""),
        "secondary_color": brand_kit.get("secondary_color", ""),
        "accent_color": brand_kit.get("accent_color", ""),
        "font_family": brand_kit.get("font_family", ""),
    }

    date_ctx = {
        "today": now.strftime("%Y-%m-%d"),
        "long": _long_date(now),
        "year": now.strftime("%Y"),
        "month": now.strftime("%m"),
        "day": now.strftime("%d"),
        "iso": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    return {
        "user": user_ctx,
        "company": company_ctx,
        "brand": brand_ctx,
        "date": date_ctx,
        "data": extra_data or {},
    }


def resolve_paths(context: Dict[str, Any], paths: Iterable[str]) -> ResolvedVariables:
    """Resolve a set of paths against a pre-built context. Pure."""
    out = ResolvedVariables()
    for path in sorted(set(paths)):
        if is_dynamic_path(path):
            value = _walk(context.get("data", {}), path[len(DYNAMIC_PREFIX):])
            if value is None or value == "":
                out.unresolved.append(path)
            else:
                out.values[path] = str(value)
            continue
        if not is_known_path(path):
            out.unknown.append(path)
            continue
        category, _, key = path.partition(".")
        value = context.get(category, {}).get(key)
        if value is None or value == "":
            out.unresolved.append(path)
        else:
            out.values[path] = str(value)
    return out


class VariableResolver:
    """DB-backed resolver. Fetches the user, business profile and brand kit, then
    delegates to the pure helpers above."""

    def __init__(self, db: Session):
        self.db = db

    def resolve(
        self,
        workspace_id: UUID,
        user_id: Optional[int],
        paths: Iterable[str],
        now: Optional[datetime] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> ResolvedVariables:
        now = now or datetime.utcnow()
        context = self.build_context_for(workspace_id, user_id, now, extra_data)
        return resolve_paths(context, paths)

    def build_context_for(
        self,
        workspace_id: UUID,
        user_id: Optional[int],
        now: Optional[datetime] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch DB objects for the workspace/user and build the resolution context."""
        now = now or datetime.utcnow()
        # Imported here to keep the pure helpers import-light and avoid a circular
        # import with the model layer at module load.
        from core.models.business_profiles import BusinessProfile
        from core.models.core import User
        from core.models.workspaces import Workspace

        user = self.db.query(User).filter(User.id == user_id).first() if user_id else None
        workspace = self.db.query(Workspace).filter(Workspace.id == workspace_id).first()
        business_profile = (
            self.db.query(BusinessProfile)
            .filter(BusinessProfile.workspace_id == workspace_id)
            .order_by(BusinessProfile.created_at.desc())
            .first()
        )
        brand_kit = get_brand_kit(getattr(workspace, "settings", None))
        return build_context(user, business_profile, brand_kit, now, extra_data)


__all__ = ["ResolvedVariables", "build_context", "resolve_paths", "VariableResolver"]
