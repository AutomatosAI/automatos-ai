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


# PRD-007: default proactive widget config seeded into workspace.settings
# at provision time. Merchant flips `enabled: true` from the dashboard
# (or via PATCH /api/workspaces/:id/settings) to activate.
DEFAULT_WIDGET_PROACTIVE_CONFIG: dict = {
    "enabled": False,
    "page_types": ["product"],
    "triggers": [
        {"type": "time_on_page", "seconds": 20},
    ],
    "frequency_cap": {"scope": "session", "max_pops": 1},
    "greeting_source": "agent_with_canned_fallback",
    "canned_fallback": "Need a hand finding the right product?",
    "agent_timeout_ms": 1500,
    "popup_style": "corner_bubble",
    "respect_consent": True,
    "dismissal_persistence": "session",
}


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
                "widget_proactive": dict(DEFAULT_WIDGET_PROACTIVE_CONFIG),
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

    # PRD-009 Layer 2 — incremental graph updates on catalog mutations.
    # We treat product/collection/inventory events as triggers to re-sync
    # ONLY the touched entities. For the POC we use a coarse signal: any
    # catalog-shaped event schedules an incremental rebuild via the existing
    # GraphifyService.schedule_incremental_update mechanism. Fine-grained
    # per-product re-embed is a follow-up — graph diff machinery in
    # graph_service already deduplicates so this is safe.
    CATALOG_EVENTS = {
        "products/create", "products/update", "products/delete",
        "inventory_levels/update",
        "collections/create", "collections/update", "collections/delete",
    }
    if request.event in CATALOG_EVENTS:
        # Resolve workspace by shop domain (already an indexed lookup pattern
        # used by /sync and /deactivate above).
        ws = (
            db.query(Workspace)
            .filter(
                Workspace.settings["shopify_domain"].astext == request.shop,
                Workspace.is_active.is_(True),
            )
            .first()
        )
        if ws:
            try:
                from modules.knowledge.graph_service import GraphifyService
                gs = GraphifyService()
                gs.schedule_incremental_update(
                    workspace_id=str(ws.id),
                    changed_sources=[{
                        "source": "shopify",
                        "event": request.event,
                        "shop": request.shop,
                    }],
                )
                logger.info(
                    "[PRD-009] Scheduled incremental graph update for workspace=%s reason=%s",
                    ws.id, request.event,
                )
            except Exception as e:
                logger.warning(
                    "[PRD-009] Could not schedule incremental update: %s", e
                )

    return {"status": "received", "event": request.event}


# ===================================================================
# PRD-009 Layer 2 — Product Knowledge Graph sync
# ===================================================================
# Composio's SHOPIFY_BULK_QUERY_OPERATION returns a signed GCS URL once the
# Bulk Op completes (synchronous from our side — Composio polls Shopify).
# We download the JSONL, map it to graph nodes/edges via the existing
# graph_extraction.map_shopify_catalog helper, and hand it to the existing
# GraphifyService.import_graph for clustering + persistence.

# Shopify Admin GraphQL bulk-op query for the catalog.
_SHOPIFY_BULK_CATALOG_QUERY = """{
  products {
    edges {
      node {
        id
        title
        handle
        productType
        vendor
        status
        tags
        descriptionHtml
        createdAt
        updatedAt
        publishedAt
        priceRangeV2 { minVariantPrice { amount currencyCode } maxVariantPrice { amount currencyCode } }
        featuredImage { url altText }
        totalInventory
        tracksInventory
        variants { edges { node { id sku title price compareAtPrice inventoryQuantity availableForSale selectedOptions { name value } barcode } } }
        metafields { edges { node { id namespace key value type } } }
        collections { edges { node { id title handle } } }
      }
    }
  }
}"""


class SyncStartResponse(BaseModel):
    status: str
    workspace_id: str
    bulk_operation_id: Optional[str] = None
    object_count: Optional[int] = None
    file_size: Optional[int] = None
    download_seconds: Optional[float] = None
    node_count: Optional[int] = None
    edge_count: Optional[int] = None
    community_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


@router.post("/sync/products/start", response_model=SyncStartResponse)
async def start_product_sync(
    workspace_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Run a full Shopify catalog sync → knowledge graph for a workspace.

    Reuses:
      - composio_client.composio.tools.execute  (SHOPIFY_BULK_QUERY_OPERATION)
      - modules.knowledge.graph_extraction.map_shopify_catalog
      - modules.knowledge.graph_service.GraphifyService.import_graph
      - core.composio.entity_manager.EntityManager

    Called manually (admin) or auto-triggered when a workspace's SHOPIFY
    Composio connection flips pending → active (see consumer of this).
    """
    import time
    import httpx

    from core.composio.client import get_composio_client
    from core.composio.entity_manager import EntityManager
    from modules.knowledge.graph_extraction import map_shopify_catalog
    from modules.knowledge.graph_service import GraphifyService

    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id required")

    workspace = db.query(Workspace).get(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    em = EntityManager(db)
    entity = em.get_or_create_entity(workspace_id)
    entity_id = entity.get("composio_entity_id")
    if not entity_id:
        raise HTTPException(status_code=400, detail="No Composio entity for workspace")

    client = get_composio_client()
    if not client.composio:
        raise HTTPException(status_code=503, detail="Composio not configured")

    t0 = time.time()
    settings = dict(workspace.settings or {})
    settings["product_sync"] = {"status": "running", "started_at": time.time()}
    workspace.settings = settings
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(workspace, "settings")
    db.commit()

    try:
        # 1. Bulk Op via Composio (synchronous — returns the signed URL).
        # Python SDK 0.12.0 returns a dict with keys: data, error, successful,
        # logId. NOT an attribute-access object — keep .get() everywhere.
        bulk = client.composio.tools.execute(
            "SHOPIFY_BULK_QUERY_OPERATION",
            user_id=entity_id,
            arguments={"query": _SHOPIFY_BULK_CATALOG_QUERY},
        )
        # Be liberal in what we accept — some SDK variants return objects with
        # attribute access. Normalise to dict-style.
        bulk_dict = bulk if isinstance(bulk, dict) else {
            "successful": getattr(bulk, "successful", None),
            "data": getattr(bulk, "data", None),
            "error": getattr(bulk, "error", None),
        }
        if not bulk_dict.get("successful"):
            raise HTTPException(
                status_code=502,
                detail=f"Bulk Op failed: {bulk_dict.get('error') or bulk_dict}",
            )
        bulk_data = bulk_dict.get("data") or {}
        download_url = bulk_data.get("url")
        bulk_op_id = bulk_data.get("bulk_operation_id")
        object_count = int(bulk_data.get("object_count") or 0)
        file_size = int(bulk_data.get("file_size") or 0)
        if not download_url:
            raise HTTPException(
                status_code=502,
                detail=f"Bulk Op did not return a download URL — data={bulk_data}",
            )

        # 2. Download JSONL
        dl_t0 = time.time()
        async with httpx.AsyncClient(timeout=120.0) as http:
            resp = await http.get(download_url)
            resp.raise_for_status()
            jsonl_text = resp.text
        dl_secs = time.time() - dl_t0

        # 3. Map to graph (deterministic, in-memory)
        graph = map_shopify_catalog(jsonl_text.splitlines(), bulk_op_id=bulk_op_id)

        # 4. Import via existing GraphifyService — clusters + persists + exports
        gs = GraphifyService()
        meta = await gs.import_graph(workspace_id, graph, merge=False)

        duration = time.time() - t0
        settings["product_sync"] = {
            "status": "complete",
            "bulk_operation_id": bulk_op_id,
            "object_count": object_count,
            "file_size": file_size,
            "node_count": meta.get("node_count"),
            "edge_count": meta.get("edge_count"),
            "community_count": meta.get("community_count"),
            "duration_seconds": duration,
            "completed_at": time.time(),
        }
        workspace.settings = settings
        flag_modified(workspace, "settings")
        db.commit()

        logger.info(
            "PRD-009 sync complete: workspace=%s nodes=%s edges=%s duration=%.1fs",
            workspace_id, meta.get("node_count"), meta.get("edge_count"), duration,
        )

        return SyncStartResponse(
            status="complete",
            workspace_id=str(workspace_id),
            bulk_operation_id=bulk_op_id,
            object_count=object_count,
            file_size=file_size,
            download_seconds=dl_secs,
            node_count=meta.get("node_count"),
            edge_count=meta.get("edge_count"),
            community_count=meta.get("community_count"),
            duration_seconds=duration,
        )
    except HTTPException as he:
        # Persist the failure so the UI doesn't stay stuck on "running" — the
        # previous version's `except HTTPException: raise` skipped this and
        # left workspaces wedged in running-forever state.
        settings["product_sync"] = {
            "status": "error",
            "error": str(he.detail) if isinstance(he.detail, str) else f"HTTP {he.status_code}",
            "errored_at": time.time(),
        }
        workspace.settings = settings
        flag_modified(workspace, "settings")
        db.commit()
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("PRD-009 sync failed for workspace %s", workspace_id)
        settings["product_sync"] = {"status": "error", "error": str(e), "errored_at": time.time()}
        workspace.settings = settings
        flag_modified(workspace, "settings")
        db.commit()
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")


@router.get("/sync/status")
async def get_product_sync_status(
    workspace_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Return the current product_sync state stored on the workspace."""
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id required")
    workspace = db.query(Workspace).get(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return (workspace.settings or {}).get("product_sync") or {"status": "never_synced"}


# ===================================================================
# PRD-009 Phase 2 — Orders → FREQUENTLY_BOUGHT_WITH edges
# ===================================================================
# Privacy by design: GraphQL query requests only order id, createdAt,
# cancelledAt, currencyCode, and lineItems[].variant.product.id. NO
# customer / email / phone / address / note fields are requested, so they
# can never leak into the graph regardless of mapper bugs. The mapper
# produces ONLY Product↔Product co-occurrence edges; no Customer or Order
# nodes ever land in workspace_graphs.


def _orders_bulk_query(days: int = 90) -> str:
    """Build the orders bulk-op GraphQL string. Window in days from now."""
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    # Note: Shopify's Bulk Op GraphQL doesn't support variables, so we inline
    # the cutoff date. Safe — it's a controlled date string, not user input.
    return (
        "{ orders(query: \"created_at:>=" + cutoff + "\") "
        "{ edges { node { id createdAt currencyCode cancelledAt "
        "lineItems { edges { node { id quantity "
        "variant { id product { id } } } } } } } } }"
    )


class OrdersSyncResponse(BaseModel):
    status: str
    workspace_id: str
    bulk_operation_id: Optional[str] = None
    object_count: Optional[int] = None
    file_size: Optional[int] = None
    fbt_edges_added: Optional[int] = None
    total_orders_analysed: Optional[int] = None
    days_window: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


@router.post("/sync/orders/start", response_model=OrdersSyncResponse)
async def start_orders_sync(
    workspace_id: Optional[str] = None,
    days: int = 90,
    min_support: int = 2,
    db: Session = Depends(get_db),
):
    """
    Run a Shopify orders sync → FBT edges merged into the workspace graph.

    Args:
        workspace_id: target workspace.
        days: time window for orders (default 90). Older orders less
              relevant for current customer co-purchase patterns.
        min_support: minimum number of orders a (Product A, Product B) pair
                     must co-occur in before we emit an edge (default 2).
    """
    import time
    import httpx

    from core.composio.client import get_composio_client
    from core.composio.entity_manager import EntityManager
    from modules.knowledge.graph_extraction import map_shopify_orders
    from modules.knowledge.graph_service import GraphifyService

    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id required")

    workspace = db.query(Workspace).get(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    em = EntityManager(db)
    entity = em.get_or_create_entity(workspace_id)
    entity_id = entity.get("composio_entity_id")
    if not entity_id:
        raise HTTPException(status_code=400, detail="No Composio entity for workspace")

    client = get_composio_client()
    if not client.composio:
        raise HTTPException(status_code=503, detail="Composio not configured")

    t0 = time.time()
    from sqlalchemy.orm.attributes import flag_modified
    settings = dict(workspace.settings or {})
    settings["orders_sync"] = {
        "status": "running",
        "started_at": time.time(),
        "days": days,
        "min_support": min_support,
    }
    workspace.settings = settings
    flag_modified(workspace, "settings")
    db.commit()

    try:
        bulk = client.composio.tools.execute(
            "SHOPIFY_BULK_QUERY_OPERATION",
            user_id=entity_id,
            arguments={"query": _orders_bulk_query(days=days)},
        )
        bulk_dict = bulk if isinstance(bulk, dict) else {
            "successful": getattr(bulk, "successful", None),
            "data": getattr(bulk, "data", None),
            "error": getattr(bulk, "error", None),
        }
        if not bulk_dict.get("successful"):
            raise HTTPException(
                status_code=502,
                detail=f"Bulk Op failed: {bulk_dict.get('error') or bulk_dict}",
            )
        bulk_data = bulk_dict.get("data") or {}
        download_url = bulk_data.get("url")
        bulk_op_id = bulk_data.get("bulk_operation_id")
        object_count = int(bulk_data.get("object_count") or 0)
        file_size = int(bulk_data.get("file_size") or 0)
        if not download_url:
            raise HTTPException(
                status_code=502,
                detail=f"Bulk Op did not return a download URL — data={bulk_data}",
            )

        async with httpx.AsyncClient(timeout=180.0) as http:
            resp = await http.get(download_url)
            resp.raise_for_status()
            jsonl_text = resp.text

        # Pure-function map: yields only edges, never customer/order nodes.
        graph_delta = map_shopify_orders(
            jsonl_text.splitlines(),
            bulk_op_id=bulk_op_id,
            min_support=min_support,
        )
        fbt_edges = len(graph_delta.get("edges", []))

        # Merge into the existing workspace graph (catalog nodes already
        # there from the products sync; we only add FBT edges).
        gs = GraphifyService()
        meta = await gs.import_graph(workspace_id, graph_delta, merge=True)

        # Count total orders analysed from the first edge's attrs (mapper
        # writes the same value into every edge).
        total_orders = 0
        if graph_delta["edges"]:
            total_orders = int(graph_delta["edges"][0].get("attrs", {}).get("total_orders", 0))

        duration = time.time() - t0
        settings["orders_sync"] = {
            "status": "complete",
            "bulk_operation_id": bulk_op_id,
            "object_count": object_count,
            "file_size": file_size,
            "days": days,
            "min_support": min_support,
            "fbt_edges_added": fbt_edges,
            "total_orders_analysed": total_orders,
            "duration_seconds": duration,
            "completed_at": time.time(),
        }
        workspace.settings = settings
        flag_modified(workspace, "settings")
        db.commit()

        logger.info(
            "PRD-009 orders sync complete: workspace=%s edges=%s orders=%s duration=%.1fs",
            workspace_id, fbt_edges, total_orders, duration,
        )

        return OrdersSyncResponse(
            status="complete",
            workspace_id=str(workspace_id),
            bulk_operation_id=bulk_op_id,
            object_count=object_count,
            file_size=file_size,
            fbt_edges_added=fbt_edges,
            total_orders_analysed=total_orders,
            days_window=days,
            duration_seconds=duration,
        )
    except HTTPException as he:
        settings["orders_sync"] = {
            "status": "error",
            "error": str(he.detail) if isinstance(he.detail, str) else f"HTTP {he.status_code}",
            "errored_at": time.time(),
        }
        workspace.settings = settings
        flag_modified(workspace, "settings")
        db.commit()
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("PRD-009 orders sync failed for workspace %s", workspace_id)
        settings["orders_sync"] = {"status": "error", "error": str(e), "errored_at": time.time()}
        workspace.settings = settings
        flag_modified(workspace, "settings")
        db.commit()
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")


@router.get("/sync/orders/status")
async def get_orders_sync_status(
    workspace_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Return the current orders_sync state stored on the workspace."""
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id required")
    workspace = db.query(Workspace).get(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return (workspace.settings or {}).get("orders_sync") or {"status": "never_synced"}
