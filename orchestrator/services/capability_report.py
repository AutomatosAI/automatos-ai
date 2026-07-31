"""PRD-222 W1 (US-007) — the onboarding capability report (honest-degrade signal).

A booleans-only snapshot of what the platform can actually do RIGHT NOW for a
workspace, so Auto degrades honestly (e.g. offers doc-upload instead of a scan it
can't run) rather than promising a capability the deployment lacks. Consumed by
the OnboardingSection context (US-009) and surfaced on the admin system-health
endpoint (``GET /api/system/health``).

This is a NEW module rather than an extension of ``services/substrate_health.py``
— that surface aggregates ``substrate_metric_events`` into per-seam retrieval
health (documents/memory/field), a different concern from a config-presence +
workspace-key capability read.

Secrets discipline: every value is a plain ``bool`` derived from presence only —
an API key's VALUE never leaves ``config.py`` through this report.
"""
from __future__ import annotations

from typing import Any

from config import config


def _workspace_has_validated_llm_key(db: Any, workspace_id: Any) -> bool:
    """True when the workspace holds an ACTIVE (validated) BYOK LLM key.

    US-006 made ``UserApiKey.is_active`` the validated-on-save truth (a key that
    fails its live provider test is stored inactive), and ``_resolve_api_key``
    only resolves ``is_active`` keys — so an active key is the workspace-scoped
    "you have a working key of your own" signal. ``db``/``workspace_id`` missing
    → ``False`` (can't check, don't claim; never raises).
    """
    if db is None or workspace_id is None:
        return False
    try:
        from core.models.core import UserApiKey

        row = (
            db.query(UserApiKey)
            .filter(
                UserApiKey.workspace_id == workspace_id,
                UserApiKey.is_active == True,  # noqa: E712 - SQLAlchemy column truthiness
            )
            .first()
        )
        return row is not None
    except Exception:
        return False


def onboarding_capabilities(db: Any = None, *, workspace_id: Any = None) -> dict[str, bool]:
    """Return the four onboarding capability booleans (every value is a ``bool``).

    * ``llm_key_valid`` — workspace has a validated BYOK LLM key (US-006 truth).
    * ``firecrawl_configured`` / ``composio_configured`` / ``redis_configured`` —
      the platform integration is configured (its key/URL is present in config.py).

    No secret value is ever surfaced — presence booleans only. An unset key reads
    as ``False`` without raising.
    """
    return {
        "llm_key_valid": _workspace_has_validated_llm_key(db, workspace_id),
        "firecrawl_configured": bool(config.FIRECRAWL_API_KEY),
        "composio_configured": bool(config.COMPOSIO_API_KEY),
        "redis_configured": bool(config.REDIS_URL),
    }
