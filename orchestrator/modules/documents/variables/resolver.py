"""Variable resolution service (PRD-167 S3).

Resolves ``{{user.*}} / {{company.*}} / {{brand.*}} / {{date.*}}`` against the
requesting user's profile (the workspace owner's when no person asks, F344), the
workspace business profile, the workspace brand kit and the render-time clock.

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
from ..brand_logo import BRAND_LOGO_ROUTE
from ..locale_text import currency_of, date_style_of, long_date
from .catalog import is_blank, is_dynamic_path, is_known_path, signer_of, walk_dynamic
from .chip_text import chip_text
from .document_user import document_user

logger = logging.getLogger(__name__)

# The brand context's currency: what a ``data.*`` amount prints in (not a chip of its own).
CURRENCY_KEY = "currency"


@dataclass
class ResolvedVariables:
    values: Dict[str, str] = field(default_factory=dict)
    unresolved: List[str] = field(default_factory=list)  # known paths, empty value
    unknown: List[str] = field(default_factory=list)      # paths not in the catalog


def _user_context(user: Any) -> Dict[str, str]:
    name = ((getattr(user, "name", None) or "") if user else "").strip()
    first, _, last = name.partition(" ")
    return {
        "name": name,
        "first_name": first,
        "last_name": last,
        "email": (getattr(user, "email", "") or "") if user else "",
        "username": (getattr(user, "username", "") or "") if user else "",
    }


def _company_context(business_profile: Any, brand_kit: Dict[str, Any]) -> Dict[str, str]:
    company_contact = brand_kit.get("company", {}) if isinstance(brand_kit, dict) else {}
    company_name = (
        (getattr(business_profile, "company_name", None) if business_profile else None)
        or company_contact.get("name")
        or brand_kit.get("name")
        or ""
    )
    domain = (getattr(business_profile, "domain", "") or "") if business_profile else ""
    return {
        "name": company_name,
        "website": company_contact.get("website") or domain or "",
        "address": company_contact.get("address", ""),
        "email": company_contact.get("email", ""),
        "phone": company_contact.get("phone", ""),
    }


def _sign_off(brand_kit: Dict[str, Any], person: str, signer: str = "") -> str:
    """Who signs (PRD-255 ``brand.sign_off``): the signer the document's data names (F364: "sign it
    from me, Gerard"), else the kit voice's sign-off, else the person signing (F344: an agent's
    letter is signed with the owner's name); empty when none is known."""
    if signer:
        return signer
    voice = brand_kit.get("voice") if isinstance(brand_kit.get("voice"), dict) else {}
    sign_off = voice.get("sign_off")
    return sign_off.strip() if isinstance(sign_off, str) and sign_off.strip() else person


def _brand_context(brand_kit: Dict[str, Any], company_name: str, person: str, signer: str = "") -> Dict[str, str]:
    """``brand.*``; ``currency`` is not a chip: it is what ``data.*`` amounts print in (PRD-255 FR-7)."""
    # PRD-242 S3: an uploaded logo has no public URL; the chip resolves to the
    # platform route that streams it (the renderers inline the bytes instead).
    logo_url = brand_kit.get("logo_url", "") or (
        BRAND_LOGO_ROUTE if brand_kit.get("logo_path") else ""
    )
    return {
        "name": brand_kit.get("name") or company_name or "",
        "tagline": brand_kit.get("tagline", ""),
        "logo_url": logo_url,
        "primary_color": brand_kit.get("primary_color", ""),
        "secondary_color": brand_kit.get("secondary_color", ""),
        "accent_color": brand_kit.get("accent_color", ""),
        "font_family": brand_kit.get("font_family", ""),
        "sign_off": _sign_off(brand_kit, person, signer),
        CURRENCY_KEY: currency_of(brand_kit),
    }


def _date_context(now: datetime, brand_kit: Dict[str, Any]) -> Dict[str, str]:
    return {
        "today": now.strftime("%Y-%m-%d"),
        # PRD-255 FR-8: in the kit's date style ("5 October 2026" by default; F350).
        "long": long_date(now, date_style_of(brand_kit)),
        "year": now.strftime("%Y"),
        "month": now.strftime("%m"),
        "day": now.strftime("%d"),
        "iso": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


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
    kit = brand_kit if isinstance(brand_kit, dict) else {}
    company_ctx = _company_context(business_profile, kit)
    user_ctx = _user_context(user)
    return {
        "user": user_ctx,
        "company": company_ctx,
        "brand": _brand_context(kit, company_ctx["name"], user_ctx["name"], signer_of(extra_data)),
        "date": _date_context(now, kit),
        "data": extra_data or {},
    }


def resolve_paths(context: Dict[str, Any], paths: Iterable[str]) -> ResolvedVariables:
    """Resolve a set of paths against a pre-built context. Pure."""
    out = ResolvedVariables()
    currency = context.get("brand", {}).get(CURRENCY_KEY, "")
    for path in sorted(set(paths)):
        if is_dynamic_path(path):
            value = walk_dynamic(context.get("data", {}), path)
            if is_blank(value):  # F345: whitespace fills nothing
                out.unresolved.append(path)
            else:
                out.values[path] = chip_text(path, value, currency)  # F347: "311.00"; F356: a list of names as bullets
            continue
        if not is_known_path(path):
            out.unknown.append(path)
            continue
        category, _, key = path.partition(".")
        value = context.get(category, {}).get(key)
        if is_blank(value):  # F345: whitespace fills nothing
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
        from core.models.workspaces import Workspace

        workspace = self.db.query(Workspace).filter(Workspace.id == workspace_id).first()
        business_profile = (
            self.db.query(BusinessProfile)
            .filter(BusinessProfile.workspace_id == workspace_id)
            .order_by(BusinessProfile.created_at.desc())
            .first()
        )
        brand_kit = get_brand_kit(getattr(workspace, "settings", None))
        # F344: with no person asking (an agent, Auto), user.* is this workspace's owner.
        user = document_user(self.db, workspace, user_id, brand_kit)
        return build_context(user, business_profile, brand_kit, now, extra_data)


__all__ = ["ResolvedVariables", "build_context", "resolve_paths", "VariableResolver"]
