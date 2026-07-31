"""
Model policy — which models may occupy the orchestrator (Auto) seat.

PRD-223 Wave 0: the 2026-07-31 incident put an unvetted model in Auto's chair
through a route with zero validation. This module is the interim policy gate:
a quarantine denylist plus an optional strict allowlist, both runtime-tunable
via ``system_settings(category='model_policy')``. Wave 1 replaces these lists
with per-model approval rows on ``WorkspaceModel``; the call sites stay.

Semantics (decided PRD-223 §2 / D2):
- Quarantined model for the orchestrator role → BLOCKED (fail-closed).
- Allowlist empty → any non-quarantined model passes (quarantine-only mode).
- Allowlist non-empty → strict mode: only listed models pass.
- Policy storage unreadable → ALLOW and log an error (fail-open on infra
  failure only — Auto degrades to a trusted brain elsewhere, never to a dead
  chat because the settings table hiccuped).
"""

import json
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

MODEL_POLICY_CATEGORY = "model_policy"
ORCHESTRATOR_QUARANTINE_KEY = "orchestrator_quarantine"
ORCHESTRATOR_ALLOWLIST_KEY = "orchestrator_allowlist"


def _load_model_list(key: str) -> List[str]:
    """Read a JSON-list setting from model_policy; [] on missing or malformed."""
    from core.llm.manager import get_system_setting

    raw = get_system_setting(MODEL_POLICY_CATEGORY, key, "[]")
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        logger.error(
            "model_policy.%s is not a valid JSON list (%r) — treating as empty",
            key, raw,
        )
        return []
    if not isinstance(value, list):
        logger.error(
            "model_policy.%s must be a JSON list, got %s — treating as empty",
            key, type(value).__name__,
        )
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def check_orchestrator_model(model_id: str) -> Tuple[bool, str]:
    """Return (allowed, reason) for a model occupying the orchestrator seat."""
    candidate = (model_id or "").strip()
    if not candidate:
        return False, "no model id supplied"

    quarantined = _load_model_list(ORCHESTRATOR_QUARANTINE_KEY)
    if candidate in quarantined:
        return False, (
            f"model '{candidate}' is quarantined for the orchestrator role "
            "(model_policy.orchestrator_quarantine)"
        )

    allowlist = _load_model_list(ORCHESTRATOR_ALLOWLIST_KEY)
    if allowlist and candidate not in allowlist:
        return False, (
            f"model '{candidate}' is not on the orchestrator allowlist "
            "(model_policy.orchestrator_allowlist is in strict mode)"
        )

    return True, "allowed"
