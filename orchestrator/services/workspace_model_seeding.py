"""Workspace base-model seeding — every workspace starts with working models.

Gerard's 2026-08-29 design call (the Harbourline test): a new workspace must
never start with ZERO rows in ``workspace_models`` — that empty mapping is why
Settings → Orchestrator failed with "Failed to load LLM settings" and why no
sane default existed for chat. Provisioning now seeds a small set of
OpenRouter-served base models (the platform key everyone has), marks one
primary in ``workspaces.settings.orchestrator``, and the marketplace remains
the "add more" path (PRD-230 D1 registers those to the workspace too).

Selection: active ``llm_models`` rows served via OpenRouter — ``is_default``
rows first, then featured OpenRouter-tier rows — capped at ``_SEED_CAP``.
Rows are inserted with ``source='default'`` (the enum value the schema always
anticipated) and ``approval_status='approved'`` (they are the platform's own
vetted picks; PRD-223 governance still quarantines them like any model later).

Idempotent: the (workspace_id, model_id) unique constraint + existence checks
make re-runs no-ops. Never raises into provisioning — a workspace without
seeded models is degraded, not broken, and the failure is logged loudly.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from core.models.core import LLMModel, WorkspaceModel
from core.models.workspaces import Workspace

logger = logging.getLogger(__name__)

_SEED_CAP = 4


def _openrouter_served(q):
    """Filter llm_models to rows the platform OpenRouter key can serve."""
    return q.filter(
        or_(
            LLMModel.provider == "openrouter",
            LLMModel.tier == "openrouter",
            LLMModel.model_id.contains("/"),
        )
    )


def pick_base_models(db: Session) -> list[LLMModel]:
    """The platform's base set: defaults first, then featured, OpenRouter-served."""
    base = _openrouter_served(
        db.query(LLMModel).filter(LLMModel.status == "active")
    )
    defaults = base.filter(LLMModel.is_default.is_(True)).all()
    picked = list(defaults[:_SEED_CAP])
    if len(picked) < _SEED_CAP:
        seen = {m.id for m in picked}
        featured = (
            base.filter(LLMModel.is_featured.is_(True))
            .order_by(LLMModel.popularity_score.desc())
            .limit(_SEED_CAP * 2)
            .all()
        )
        for m in featured:
            if m.id not in seen:
                picked.append(m)
                seen.add(m.id)
            if len(picked) >= _SEED_CAP:
                break
    return picked


def seed_workspace_models(db: Session, ws_id: Any) -> Optional[str]:
    """Seed base models into ``workspace_models`` + set the primary.

    Takes the workspace id (the provisioning site's currency). Returns the
    primary model id string when seeding produced/confirmed one, else None.
    Caller owns the commit. Never raises.
    """
    try:
        workspace = db.query(Workspace).get(ws_id)
        if workspace is None:
            logger.error("[ModelSeed] workspace %s not found", ws_id)
            return None
        existing = {
            wm.model_id
            for wm in db.query(WorkspaceModel.model_id)
            .filter(WorkspaceModel.workspace_id == ws_id)
            .all()
        }
        base = pick_base_models(db)
        if not base and not existing:
            logger.error(
                "[ModelSeed] no OpenRouter-served base models in llm_models — "
                "workspace %s starts with an empty model mapping", ws_id
            )
            return None
        for m in base:
            if m.id in existing:
                continue
            db.add(
                WorkspaceModel(
                    workspace_id=ws_id,
                    model_id=m.id,
                    is_active=True,
                    source="default",
                    approval_status="approved",
                )
            )
        # Primary: first default pick (or keep an already-set choice).
        # PRD-220: rebuild the settings doc, never mutate in place.
        settings = dict(workspace.settings or {})
        orch = dict(settings.get("orchestrator", {}))
        primary = orch.get("model")
        if not primary and base:
            primary = base[0].model_id
            orch["model"] = primary
            settings["orchestrator"] = orch
            workspace.settings = settings
            db.add(workspace)
        return str(primary) if primary else None
    except Exception:  # pragma: no cover — degraded, not broken
        logger.exception("[ModelSeed] seeding failed for workspace %s", getattr(workspace, "id", "?"))
        return None
