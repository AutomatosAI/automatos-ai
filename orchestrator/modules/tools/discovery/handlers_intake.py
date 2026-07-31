"""Business-intake handlers for PlatformActionExecutor (PRD-222 W1S8).

``platform_scan_business_site`` reuses ``api.wizard.start_business_scan`` (the same
Firecrawl map + archetype select + background pipeline the wizard uses) and never
lets the wizard's 503 escape — an unconfigured Firecrawl is an HONEST tool result,
not an error. ``platform_get_intake_status`` reads business_profiles scoped to the
caller's workspace; a cross-workspace profile_id is refused, never leaked.
"""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

from config import config

logger = logging.getLogger(__name__)


def _summarize_profile(profile: Any) -> Dict[str, Any]:
    """Client-safe intake summary — stage + shape counts, no scraped content dumped."""
    return {
        "profile_id": str(profile.id),
        "stage": profile.status,
        "domain": profile.domain,
        "archetype": profile.archetype,
        "company_name": profile.company_name,
        "pages_found": len(profile.raw_map_urls or []),
        "pages_selected": len(profile.selected_urls or []),
        "quality_findings": profile.quality_findings or {},
    }


async def scan_business_site(
    db: Session, workspace_id: Any, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Start the intake pipeline for a domain (US-008).

    Firecrawl unset → honest ``{configured:false}`` (never a 503 through the tool).
    """
    domain = (params.get("domain") or "").strip()
    if not domain:
        return {"success": False, "error": "domain is required"}

    if not config.FIRECRAWL_API_KEY:
        # Honest degrade — the deployment can't scan the web; offer alternatives.
        return {
            "success": True,
            "data": {
                "configured": False,
                "alternatives": "doc upload / conversation",
            },
        }

    try:
        from api.wizard import start_business_scan

        data = await start_business_scan(db, workspace_id, domain)
        return {"success": True, "data": data}
    except Exception as exc:  # noqa: BLE001 - surface a clean tool error, never crash/503
        logger.error("[scan_business_site] failed: %s", exc, exc_info=True)
        try:
            db.rollback()
        except Exception:
            pass
        return {"success": False, "error": str(exc)}


async def get_intake_status(
    db: Session, workspace_id: Any, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return stage + summary for a workspace-owned intake profile (US-008).

    A profile_id belonging to another workspace (or unknown) is refused with the
    same not-found message — never confirming another tenant's profile exists.
    """
    raw = params.get("profile_id")
    if not raw:
        return {"success": False, "error": "profile_id is required"}
    try:
        pid = UUID(str(raw))
    except (ValueError, TypeError, AttributeError):
        return {"success": False, "error": "invalid profile_id"}

    from core.models.business_profiles import BusinessProfile

    profile = (
        db.query(BusinessProfile)
        .filter(
            BusinessProfile.id == pid,
            BusinessProfile.workspace_id == workspace_id,
        )
        .first()
    )
    if not profile:
        return {"success": False, "error": "intake profile not found"}

    return {"success": True, "data": _summarize_profile(profile)}
