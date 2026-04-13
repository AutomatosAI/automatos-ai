"""
Shopify Integration API Endpoints

Handles the install-time provisioning flow when a Shopify merchant installs the
Automatos app from the Shopify App Store:

1. POST /api/shopify/provision  — Create workspace, seed agents, return API key
2. POST /api/shopify/connect    — Store Shopify access token for Composio
3. POST /api/shopify/events     — Forward Shopify webhook events
4. POST /api/shopify/deactivate — Deactivate workspace on app uninstall
5. POST /api/shopify/sync       — Sync shop data changes
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.core import Agent, Skill
from core.models.workspaces import Workspace
from core.services.api_key_service import ApiKeyService
from config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/shopify", tags=["Shopify Integration"])

# Internal API key for Shopify app → Automatos server calls
SHOPIFY_INTERNAL_KEY = config.SHOPIFY_INTERNAL_API_KEY if hasattr(config, "SHOPIFY_INTERNAL_API_KEY") else None


# ── Auth helper ──────────────────────────────────────────────────────

def _verify_internal_key(authorization: str = Header(...)) -> None:
    """Verify the internal API key from the Shopify app server."""
    if not SHOPIFY_INTERNAL_KEY:
        # No key configured — accept all (dev mode)
        return
    token = authorization.replace("Bearer ", "")
    if token != SHOPIFY_INTERNAL_KEY:
        raise HTTPException(status_code=401, detail="Invalid internal API key")


# ── Pydantic models ─────────────────────────────────────────────────

class ProvisionRequest(BaseModel):
    source: str = "shopify"
    external_id: str = Field(..., description="Shop domain, e.g. store.myshopify.com")
    name: str = Field(..., description="Shop display name")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProvisionResponse(BaseModel):
    id: str
    public_id: str
    name: str
    api_key: str = Field(..., description="Public widget API key — shown once")
    agents_installed: int
    is_new: bool


class ConnectRequest(BaseModel):
    workspace_id: str
    shop_domain: str
    access_token: str


class EventRequest(BaseModel):
    shop: str
    event: str
    data: Any = None


class DeactivateRequest(BaseModel):
    external_id: str
    source: str = "shopify"


class SyncRequest(BaseModel):
    shop: str
    data: Any = None


# ── Shopify marketplace agent slugs ─────────────────────────────────

SHOPIFY_AGENT_SLUGS = [
    "shopify-ops",
    "shopify-support",
    "shopify-product-expert",
    "shopify-merchandiser",
    "shopify-review-analyst",
    "shopify-gift-concierge",
    "shopify-seo-content",
    "shopify-business-analyst",
    "shopify-inventory-watchdog",
]


# ===================================================================
# POST /api/shopify/provision
# ===================================================================

@router.post("/provision", response_model=ProvisionResponse)
async def provision_workspace(
    request: ProvisionRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
):
    """
    Provision an Automatos workspace for a Shopify store.

    Idempotent: if a workspace with this external_id already exists, returns it
    without re-seeding agents. The API key is re-generated on re-provision.

    Flow:
    1. Find or create workspace
    2. Clone marketplace agents into workspace
    3. Create a public API key for widget usage
    4. Return workspace + key
    """
    shop = request.external_id
    is_new = False

    # 1. Find existing workspace by shop domain in settings
    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.settings["shopify_domain"].astext == shop,
            Workspace.is_active.is_(True),
        )
        .first()
    )

    if not workspace:
        # Create new workspace
        workspace_id = uuid4()
        workspace = Workspace(
            id=workspace_id,
            name=request.name,
            slug=shop.replace(".myshopify.com", ""),
            plan="starter",
            is_personal=False,
            is_active=True,
            webhook_key=uuid4().hex,
            settings={
                "source": request.source,
                "shopify_domain": shop,
                "shopify_metadata": request.metadata,
            },
        )
        db.add(workspace)
        db.flush()
        is_new = True
        logger.info("Created workspace %s for shop %s", workspace.id, shop)

    # 2. Clone marketplace agents (only if new or no agents exist)
    existing_agent_count = (
        db.query(Agent)
        .filter(
            Agent.workspace_id == workspace.id,
            Agent.owner_type == "workspace",
        )
        .count()
    )

    agents_installed = 0
    if existing_agent_count == 0:
        agents_installed = _seed_shopify_agents(db, workspace.id)

    # 3. Create public API key for widget usage
    key_result = ApiKeyService.create_api_key(
        db=db,
        workspace_id=workspace.id,
        name=f"Shopify Widget Key ({shop})",
        key_type="public",
        permissions=["chat", "documents:read", "agents:read", "agents:execute"],
        allowed_domains=[f"https://{shop}", f"https://*.{shop}", "https://*.myshopify.com"],
    )

    db.commit()

    logger.info(
        "Provisioned workspace %s for %s: %d agents, key=%s",
        workspace.id, shop, agents_installed, key_result["key_prefix"],
    )

    return ProvisionResponse(
        id=str(workspace.id),
        public_id=str(workspace.id),
        name=workspace.name,
        api_key=key_result["key"],
        agents_installed=agents_installed,
        is_new=is_new,
    )


def _seed_shopify_agents(db: Session, workspace_id: UUID) -> int:
    """Clone Shopify marketplace agents into a workspace."""
    marketplace_agents = (
        db.query(Agent)
        .filter(
            Agent.owner_type == "marketplace",
            Agent.is_approved.is_(True),
            Agent.slug.in_(SHOPIFY_AGENT_SLUGS),
        )
        .all()
    )

    cloned_count = 0
    ops_manager_clone_id = None

    for marketplace_agent in marketplace_agents:
        # Check if already cloned (idempotent)
        existing = (
            db.query(Agent)
            .filter(
                Agent.workspace_id == workspace_id,
                Agent.cloned_from_id == marketplace_agent.id,
            )
            .first()
        )
        if existing:
            if marketplace_agent.slug == "shopify-ops":
                ops_manager_clone_id = existing.id
            continue

        cloned = Agent(
            name=marketplace_agent.name,
            slug=marketplace_agent.slug,
            description=marketplace_agent.description,
            agent_type=marketplace_agent.agent_type,
            status=marketplace_agent.status,
            configuration=marketplace_agent.configuration,
            model_config=marketplace_agent.model_config,
            tags=marketplace_agent.tags,
            marketplace_category=marketplace_agent.marketplace_category,
            marketplace_icon=marketplace_agent.marketplace_icon,
            team=marketplace_agent.team,
            job_title=marketplace_agent.job_title,
            custom_persona_prompt=marketplace_agent.custom_persona_prompt,
            use_custom_persona=marketplace_agent.use_custom_persona,
            # Ownership
            owner_type="workspace",
            owner_id=str(workspace_id),
            workspace_id=workspace_id,
            cloned_from_id=marketplace_agent.id,
            original_creator_id=marketplace_agent.original_creator_id,
            is_approved=True,
            version=marketplace_agent.version,
        )
        db.add(cloned)
        db.flush()

        # Copy skills
        if marketplace_agent.skills:
            cloned.skills = list(marketplace_agent.skills)

        # Copy tool assignments
        tool_rows = db.execute(
            text("SELECT tool_id, enabled FROM agent_tool_assignments WHERE agent_id = :aid"),
            {"aid": marketplace_agent.id},
        ).fetchall()
        for tool_id, enabled in tool_rows:
            db.execute(
                text(
                    "INSERT INTO agent_tool_assignments (agent_id, tool_id, enabled, created_at, updated_at) "
                    "VALUES (:aid, :tid, :en, NOW(), NOW())"
                ),
                {"aid": cloned.id, "tid": tool_id, "en": enabled},
            )

        # Increment marketplace install count
        marketplace_agent.install_count = (marketplace_agent.install_count or 0) + 1

        if marketplace_agent.slug == "shopify-ops":
            ops_manager_clone_id = cloned.id

        cloned_count += 1

    # Wire reports_to for widget agents → ops manager
    if ops_manager_clone_id:
        db.query(Agent).filter(
            Agent.workspace_id == workspace_id,
            Agent.slug.in_(SHOPIFY_AGENT_SLUGS),
            Agent.slug != "shopify-ops",
        ).update(
            {"reports_to_id": ops_manager_clone_id},
            synchronize_session="fetch",
        )

    return cloned_count


# ===================================================================
# POST /api/shopify/connect
# ===================================================================

@router.post("/connect")
async def connect_shopify_store(
    request: ConnectRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
):
    """
    Store the Shopify access token for a workspace.

    Saved in workspace.settings.shopify_access_token (encrypted at rest
    via database-level encryption). Used by Composio for API calls.
    """
    workspace = db.query(Workspace).get(request.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    settings = dict(workspace.settings or {})
    settings["shopify_domain"] = request.shop_domain
    settings["shopify_access_token"] = request.access_token
    workspace.settings = settings

    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(workspace, "settings")
    db.commit()

    logger.info("Stored Shopify credentials for workspace %s", workspace.id)

    return {"status": "connected", "shop": request.shop_domain}


# ===================================================================
# POST /api/shopify/deactivate
# ===================================================================

@router.post("/deactivate")
async def deactivate_workspace(
    request: DeactivateRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
):
    """
    Deactivate a workspace when a merchant uninstalls.
    Soft-delete — data preserved for potential re-install.
    """
    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.settings["shopify_domain"].astext == request.external_id,
            Workspace.is_active.is_(True),
        )
        .first()
    )

    if not workspace:
        # Already deactivated or never provisioned — safe to ignore
        return {"status": "not_found"}

    workspace.is_active = False
    db.commit()

    logger.info("Deactivated workspace %s for shop %s", workspace.id, request.external_id)

    return {"status": "deactivated", "workspace_id": str(workspace.id)}


# ===================================================================
# POST /api/shopify/sync
# ===================================================================

@router.post("/sync")
async def sync_shop_data(
    request: SyncRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
):
    """
    Sync shop data changes (shop/update webhook).
    Updates workspace metadata.
    """
    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.settings["shopify_domain"].astext == request.shop,
            Workspace.is_active.is_(True),
        )
        .first()
    )

    if not workspace:
        return {"status": "not_found"}

    settings = dict(workspace.settings or {})
    settings["shopify_metadata"] = request.data
    workspace.settings = settings

    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(workspace, "settings")
    db.commit()

    return {"status": "synced"}


# ===================================================================
# POST /api/shopify/events
# ===================================================================

@router.post("/events")
async def forward_event(
    request: EventRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
):
    """
    Forward Shopify webhook events for agent context enrichment.
    Currently logs events; future: queue for agent processing.
    """
    logger.info(
        "Shopify event received: shop=%s event=%s",
        request.shop, request.event,
    )

    # TODO: Queue event for agent processing (orders/create → inventory agent, etc.)

    return {"status": "received", "event": request.event}
