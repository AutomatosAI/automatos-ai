"""PRD-277 US-704 (D-12) — which of Automatos's hosted services this install calls, for the web app.

Route (registered through ``RouterSpec('api.hosted_services')`` in ``router_manifest.py`` and listed in
``reports/route-manifest.json``):

==============================  ===================================================================
``GET /api/hosted-services``    ``{themeRevocations, cloudSync, cloudItems}``: each true while this
                                install calls that service (``editions.current()``'s
                                ``calls_theme_revocations``, ``calls_cloud_sync``, ``calls_cloud_items``)
==============================  ===================================================================

The enterprise web app reads it to show what offers or describes each service (the Sync button, Settings ›
Automatos Cloud and what a Sync brings, the revocation sync), or to say the service is off, because there
each one is the operator's own setting. What the install already holds is shown whatever it answers. The local and hosted web apps know the answers
when they are built (``frontend/lib/hosted-services.ts``) and never ask. Any signed-in caller: it names
the install's settings, no workspace's data. It reads no setting itself and calls nothing. A plain
``def`` (AGENTS.md).
"""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, Depends

from core import editions
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid

router = APIRouter(prefix="/api/hosted-services", tags=["Editions"])


def hosted_services() -> Dict[str, bool]:
    """Each hosted service by the web app's name for it, and whether this install calls it."""
    capabilities = editions.current()
    return {
        "themeRevocations": capabilities.calls_theme_revocations,
        "cloudSync": capabilities.calls_cloud_sync,
        "cloudItems": capabilities.calls_cloud_items,
    }


@router.get("")
def get_hosted_services(_ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, bool]:
    """Which of Automatos's hosted services this install calls."""
    return hosted_services()


__all__ = ["hosted_services", "router"]
