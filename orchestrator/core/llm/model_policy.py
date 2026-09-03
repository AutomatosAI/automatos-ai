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

# PRD-223 D4: the fixed role taxonomy (v1). A model is granted roles by the
# promotion gate; 'orchestrator' is the highest-trust seat (Auto).
MODEL_ROLES = (
    "orchestrator",   # primary Auto — highest trust requirement
    "research",       # can explore, must cite
    "drafting",       # can write, no tools
    "coding",         # task-scoped, repo-aware
    "background",     # low-risk summaries / classification
    "experimental",   # quarantined sandbox
)


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


def check_model_for_agent(
    db,
    workspace_id,
    model_id: str,
    *,
    orchestrator_seat: bool,
    provider: str = None,
) -> Tuple[bool, str]:
    """Full W1 predicate: workspace approval row + platform policy.

    Layering (PRD-223 Q5 semantics — a workspace may further RESTRICT, never
    loosen):
    - A workspace row with ``approval_status='quarantined'`` blocks the model
      for every agent in that workspace.
    - For the orchestrator seat, a workspace row with a non-empty
      ``approved_roles`` that omits 'orchestrator' blocks; an absent row or
      empty list defers to platform policy.
    - The platform quarantine/allowlist (``check_orchestrator_model``) always
      runs for the orchestrator seat — workspace grants cannot override it.
    - Approval-row lookup errors fail OPEN to platform policy (infra failure
      must not dead-chat Auto); policy matches themselves fail CLOSED.
    """
    candidate = (model_id or "").strip()
    if not candidate:
        return False, "no model id supplied"

    try:
        from core.models.core import LLMModel, WorkspaceModel
        from core.llm.providers import normalize_slug

        # PRD-236 W1: the same vendor id may be installed once per serving
        # provider; when the caller knows the route, judge THAT row.
        q = (
            db.query(WorkspaceModel)
            .join(LLMModel, WorkspaceModel.model_id == LLMModel.id)
            .filter(
                WorkspaceModel.workspace_id == workspace_id,
                LLMModel.model_id == candidate,
            )
        )
        route = normalize_slug(provider) if provider else None
        if route:
            q = q.filter(LLMModel.serving_provider == route)
        row = q.first()
        if row is not None:
            if getattr(row, "approval_status", None) == "quarantined":
                return False, (
                    f"model '{candidate}' is quarantined in this workspace "
                    "(workspace_models.approval_status)"
                )
            if orchestrator_seat:
                roles = list(getattr(row, "approved_roles", None) or [])
                if roles and "orchestrator" not in roles:
                    return False, (
                        f"model '{candidate}' is not approved for the "
                        "orchestrator role in this workspace "
                        f"(approved roles: {roles})"
                    )
    except Exception as exc:
        logger.error(
            "model approval lookup failed for '%s' in workspace %s: %s",
            candidate, workspace_id, exc,
        )

    if orchestrator_seat:
        return check_orchestrator_model(candidate)
    return True, "allowed"
