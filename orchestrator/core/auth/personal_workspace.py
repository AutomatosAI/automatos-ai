"""A new user's personal workspace: its row, then its starter content.

Split out of ``core/auth/hybrid.py`` (already over the 800-line limit) when the
Clerk production cutover made user provisioning idempotent. ``hybrid.py``'s
``_provision_new_user_workspace`` still owns the order: the user row, the advisory
lock, the owner membership and the commits around these two steps.
"""
from __future__ import annotations

import logging
import secrets
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)


def insert_personal_workspace(
    db, uid: int, clerk_user_id: str, email: Optional[str], name: Optional[str],
) -> tuple[UUID, str]:
    """Insert the user's personal ``workspaces`` row, retrying slug collisions; its id and name."""
    ws_id = uuid4()
    ws_name = f"{name or email or 'My'}'s Workspace"
    base_slug = (email or clerk_user_id).split("@")[0].lower().replace(" ", "-")[:50]

    webhook_key = uuid4().hex

    for attempt in range(5):
        slug = base_slug if attempt == 0 else f"{base_slug}-{secrets.token_hex(3)}"
        try:
            db.execute(
                text(
                    "INSERT INTO workspaces (id, name, slug, owner_id, is_personal, is_active, plan, plan_limits, webhook_key) "
                    "VALUES (:id, :name, :slug, :owner_id, true, true, 'basic', :plan_limits, :webhook_key)"  # PRD-222 W2·S1: entry tier (renamed from 'starter')
                ),
                {
                    "id": str(ws_id),
                    "name": ws_name,
                    "slug": slug,
                    "owner_id": uid,
                    "plan_limits": '{"max_agents": 10, "max_workflows": 10, "max_documents": 100, "max_members": 5}',
                    "webhook_key": webhook_key,
                },
            )
            db.flush()
            break
        except IntegrityError:
            db.rollback()
            if attempt == 4:
                raise
    return ws_id, ws_name


def seed_new_workspace(db, ws_id: UUID, uid: int) -> None:
    """Seed a just-provisioned workspace: Auto, starter templates, base models, the trial.

    Every step is non-fatal: a workspace must provision even if one seed hiccups.
    """
    # 4) Seed the Auto agent for this workspace (orchestrator default)
    try:
        from core.seeds.seed_auto_agent import seed_auto_agent
        seed_auto_agent(db, ws_id)
        db.commit()
    except Exception:
        logger.exception("Failed to seed Auto agent for workspace %s — non-fatal", ws_id)

    # 5) Seed starter document templates for this workspace
    try:
        from modules.documents.seed_templates import seed_starter_templates
        seed_starter_templates(db, ws_id)
        db.commit()
    except Exception:
        logger.exception("Failed to seed document templates for workspace %s — non-fatal", ws_id)

    # 5.5) 2026-08-29 (the Harbourline failure): every workspace starts with
    # WORKING models — base OpenRouter set in workspace_models + the primary in
    # settings.orchestrator. Zero-model workspaces broke Settings → Orchestrator
    # ("Failed to load LLM settings") and left chat with no sane default.
    try:
        from services.workspace_model_seeding import seed_workspace_models
        if seed_workspace_models(db, ws_id) is not None:
            db.commit()
    except Exception:
        logger.exception("Base-model seed failed for workspace %s — non-fatal", ws_id)

    # 6) PRD-222 US-004 (W1·S9): grant the one-time $5 onboarding trial credit so
    # the value moment lands before the BYOK ask. One per Clerk user; paused by
    # the global daily cap or the TRIAL_ENABLED kill switch (all config-driven).
    # Non-fatal — a workspace must provision even if the grant hiccups.
    try:
        from services.trial_ledger import grant_trial_at_provisioning
        if grant_trial_at_provisioning(db, ws_id, owner_id=uid) is not None:
            db.commit()
    except Exception:
        logger.exception("Trial grant failed for workspace %s — non-fatal", ws_id)
        try:
            db.rollback()
        except Exception:
            logger.exception("Rollback after the failed trial grant failed for workspace %s", ws_id)
