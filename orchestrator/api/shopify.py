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

import hmac
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import Agent, Skill
from core.models.workspaces import Workspace
from core.services.api_key_service import ApiKeyService
from config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/shopify", tags=["Shopify Integration"])


# ── At-rest secret encryption (F058) ─────────────────────────────────
#
# The Shopify Admin access token is a 147-scope full-write credential. It is
# encrypted at rest through the platform's canonical Fernet path
# (``core.credentials.encryption`` — the same mechanism used for stored n8n
# credentials and LLM API keys) before being written to
# ``workspace.settings.shopify_access_token``, and decrypted at the point of
# use. The key comes from ``config.CREDENTIAL_ENCRYPTION_KEY``; no key material
# lives in this module.

def _encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret for at-rest storage via the canonical Fernet service."""
    from core.credentials.encryption import get_encryption_service

    return get_encryption_service().encrypt(plaintext)


def _decrypt_secret(ciphertext: str) -> str:
    """Decrypt an at-rest secret written by :func:`_encrypt_secret`."""
    from core.credentials.encryption import get_encryption_service

    return get_encryption_service().decrypt(ciphertext)


# ── Auth helper ──────────────────────────────────────────────────────

def _resolve_sync_workspace(ctx: RequestContext, requested: Optional[str]) -> str:
    """PRD-172 F003: resolve the workspace a Shopify sync route may act on.

    The sync routes previously trusted a caller-supplied ``workspace_id`` query
    param and ran with only ``db=Depends(get_db)`` — a guessed UUID triggered
    costly Composio bulk-ops and overwrote another tenant's knowledge graph via
    ``import_graph(merge=False)``. Now every sync route carries the shared
    workspace-scoped auth dependency and derives its target from ``ctx``:

    - A cross-workspace admin (``admin_all_workspaces``) may target any explicit
      ``requested`` workspace (ops/backfill), else its own.
    - Every other caller is pinned to ``ctx.workspace_id``; a mismatched
      ``requested`` param is rejected 403 rather than silently honoured.
    """
    if getattr(ctx, "admin_all_workspaces", False):
        return str(requested) if requested else str(ctx.workspace_id)
    if requested and str(requested) != str(ctx.workspace_id):
        raise HTTPException(
            status_code=403,
            detail="workspace_id does not match the authenticated workspace",
        )
    return str(ctx.workspace_id)


def _verify_internal_key(authorization: str = Header(...)) -> None:
    """Verify the internal API key from the Shopify app server.

    PRD-172 F004: fail-closed. There is NO "no key configured → accept all"
    branch — an unset key is caught at boot by ``config.validate_security()``,
    so by the time a request lands the key is guaranteed present and this only
    ever compares against a real secret. A falsy configured key can no longer
    wave through an arbitrary ``Authorization: Bearer x``.
    """
    expected = (config.SHOPIFY_INTERNAL_API_KEY or "").strip()
    if not expected:
        # Defence in depth: boot should already have failed. Refuse rather than
        # fall open if this is somehow reached (e.g. key blanked at runtime).
        raise HTTPException(status_code=503, detail="Shopify provisioning not configured")
    token = authorization.replace("Bearer ", "").strip()
    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid internal API key")


def _build_allowed_domains(shop: str, metadata: Dict[str, Any]) -> List[str]:
    """Origins permitted to use the minted widget key.

    Always allows the shop's ``*.myshopify.com`` domains. When the shop has a
    custom primary domain (e.g. ``www.inbuilduk.com``) the storefront serves
    the widget from there, so the browser ``Origin`` is the custom domain —
    not ``*.myshopify.com``. Allow that host plus its apex and sibling
    subdomains, otherwise the blog/chat widgets 403 on custom-domain stores.
    """
    domains = [f"https://{shop}", f"https://*.{shop}", "https://*.myshopify.com"]

    primary = metadata.get("domain")
    if primary:
        host = primary.split("://", 1)[-1].strip("/").split("/", 1)[0]
        if host and "myshopify.com" not in host:
            apex = host[4:] if host.startswith("www.") else host
            domains += [f"https://{host}", f"https://{apex}", f"https://*.{apex}"]

    seen: set[str] = set()
    return [d for d in domains if not (d in seen or seen.add(d))]


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
    # product_session keys the frequency-cap slot by product handle, so the
    # opener fires once PER product per session rather than once for the whole
    # session across every product page (the old "session" behaviour).
    "frequency_cap": {"scope": "product_session", "max_pops": 1},
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
        allowed_domains=_build_allowed_domains(shop, request.metadata),
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

    The 147-scope Admin access token is encrypted at rest before storage
    (application-level Fernet encryption via ``core.credentials.encryption``,
    keyed by ``CREDENTIAL_ENCRYPTION_KEY``) and saved as ciphertext in
    ``workspace.settings.shopify_access_token``. Read it back through
    :func:`_decrypt_secret`. Used by Composio for API calls.
    """
    workspace = db.query(Workspace).get(request.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    settings = dict(workspace.settings or {})
    settings["shopify_domain"] = request.shop_domain
    # Encrypt at rest — never persist the raw Admin token (F058).
    settings["shopify_access_token"] = _encrypt_secret(request.access_token)
    workspace.settings = settings

    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(workspace, "settings")
    db.commit()

    logger.info("Stored Shopify credentials for workspace %s", workspace.id)

    # Ensure the workspace has a Site of type=shopify. Without this, the
    # dashboard's cart-aware panels (cart-idle, callback) never light up
    # even though the workspace IS connected. Idempotent — no-op when the
    # Site already exists with the right type.
    try:
        from services.sites import ensure_shopify_site_for_workspace
        ensure_shopify_site_for_workspace(db, workspace)
    except Exception as e:  # noqa: BLE001 — never fail connect on heal error
        logger.warning(
            "ensure_shopify_site_for_workspace failed for workspace %s: %s",
            workspace.id, e,
        )

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

# Catalog-mutation webhook topics that must refresh the commerce graph.
# This is the set the Shopify Remix app (Part B) registers and POSTs to
# ``/events``. A catalog change re-runs the full catalog sync (products →
# variants → collections → vendors) — the ONLY path that rebuilds the
# commerce graph, since ``map_shopify_catalog`` produces those nodes and
# ``partition_pending_sources`` (document-only) never can.
CATALOG_EVENTS = frozenset({
    "products/create", "products/update", "products/delete",
    "inventory_levels/update",
    "collections/create", "collections/update", "collections/delete",
})


async def _sync_catalog_for_workspace(workspace_id: str, event: str) -> None:
    """Re-sync the Shopify catalog → commerce graph for a workspace.

    Runs on its own DB session (``SessionLocal``) so it is safe to fire as a
    detached background task from the request handler — the request-scoped
    session is torn down when ``/events`` returns and must never be used here
    (this is the F033 teardown class of bug, avoided by construction).

    F032: the old handler called ``GraphifyService.schedule_incremental_update``
    with a Shopify-shaped pending dict that ``partition_pending_sources`` drops,
    so the commerce graph never changed. The catalog graph is built by
    ``_product_sync_impl`` (``map_shopify_catalog`` → ``import_graph``); a
    catalog webhook must therefore trigger a catalog re-sync, not a document
    re-extraction.
    """
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        await _product_sync_impl(workspace_id, db)
        logger.info(
            "[PRD-183 S1] Catalog graph re-synced for workspace=%s reason=%s",
            workspace_id, event,
        )
    except Exception as e:  # noqa: BLE001 — a webhook must never raise back
        logger.warning(
            "[PRD-183 S1] Catalog re-sync failed for workspace=%s reason=%s: %s",
            workspace_id, event, e,
        )
    finally:
        db.close()


@router.post("/events")
async def forward_event(
    request: EventRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_verify_internal_key),
):
    """
    Forward Shopify webhook events for agent context enrichment.

    Catalog-mutation events (products/*, collections/*, inventory_levels/update)
    trigger an incremental commerce-graph refresh (PRD-009 sub-60s freshness,
    F032). The heavy re-sync runs as a detached background task with its own
    session, so the webhook returns immediately.
    """
    logger.info(
        "Shopify event received: shop=%s event=%s",
        request.shop, request.event,
    )

    if request.event in CATALOG_EVENTS:
        # Resolve workspace by shop domain (the indexed lookup pattern used by
        # /sync and /deactivate above).
        ws = (
            db.query(Workspace)
            .filter(
                Workspace.settings["shopify_domain"].astext == request.shop,
                Workspace.is_active.is_(True),
            )
            .first()
        )
        if ws:
            import asyncio as _asyncio

            _asyncio.create_task(
                _sync_catalog_for_workspace(str(ws.id), request.event)
            )
            logger.info(
                "[PRD-183 S1] Scheduled catalog re-sync for workspace=%s reason=%s",
                ws.id, request.event,
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
        onlineStoreUrl
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
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """HTTP entrypoint for the Shopify catalog sync (PRD-172 F003 authed).

    The workspace-scoped auth dependency proves the tenant; the effective
    workspace is derived from ``ctx`` (never a guessed query param). The heavy
    lifting lives in ``_product_sync_impl`` so the internal auto-trigger
    (``api/tools.py`` on SHOPIFY going active) can call it directly with an
    already-trusted workspace_id.
    """
    effective_ws = _resolve_sync_workspace(ctx, workspace_id)
    return await _product_sync_impl(effective_ws, db)


async def _product_sync_impl(workspace_id: str, db: Session) -> "SyncStartResponse":
    """
    Run a full Shopify catalog sync → knowledge graph for a workspace.

    Reuses:
      - composio_client.composio.tools.execute  (SHOPIFY_BULK_QUERY_OPERATION)
      - modules.knowledge.graph_extraction.map_shopify_catalog
      - modules.knowledge.graph_service.GraphifyService.import_graph
      - core.composio.entity_manager.EntityManager

    Callers MUST pass an already-authorised ``workspace_id`` (the HTTP route
    resolves it from ``ctx``; the internal auto-trigger passes ``ctx.workspace_id``).
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
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return the current product_sync state stored on the workspace.

    PRD-172 F003: scoped to the authenticated workspace — a guessed UUID can no
    longer read another tenant's sync state.
    """
    effective_ws = _resolve_sync_workspace(ctx, workspace_id)
    workspace = db.query(Workspace).get(effective_ws)
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
    """Build the orders bulk-op GraphQL string.

    Args:
        days: rolling window in days. 0 (or negative) = all-time, no filter.
              90/180/365 are the common selectable presets in the UI.
    """
    from datetime import datetime, timedelta, timezone

    inner = (
        "edges { node { id createdAt currencyCode cancelledAt "
        "lineItems { edges { node { id quantity "
        "variant { id product { id } } } } } } }"
    )

    if days and days > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        # Note: Shopify's Bulk Op GraphQL doesn't support variables, so we inline
        # the cutoff date. Safe — controlled date string, not user input.
        return "{ orders(query: \"created_at:>=" + cutoff + "\") { " + inner + " } }"

    # All-time: omit the filter. Shopify's bulk-op streams everything.
    # For very large merchants this may approach file-size limits; we'll
    # handle chunking in a follow-up if anyone actually hits it.
    return "{ orders { " + inner + " } }"


class OrdersSyncResponse(BaseModel):
    status: str
    workspace_id: str
    bulk_operation_id: Optional[str] = None
    object_count: Optional[int] = None
    file_size: Optional[int] = None
    fbt_edges_added: Optional[int] = None
    stale_fbt_removed: Optional[int] = None
    total_orders_analysed: Optional[int] = None
    days_window: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


@router.post("/sync/orders/start", response_model=OrdersSyncResponse)
async def start_orders_sync(
    workspace_id: Optional[str] = None,
    days: int = 90,
    min_support: int = 2,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """HTTP entrypoint for the Shopify orders sync (PRD-172 F003 authed).

    Proves the tenant via the shared auth dependency and derives the workspace
    from ``ctx`` before running the (costly, graph-mutating) sync.
    """
    effective_ws = _resolve_sync_workspace(ctx, workspace_id)
    return await _orders_sync_impl(effective_ws, days, min_support, db)


async def _orders_sync_impl(
    workspace_id: str, days: int, min_support: int, db: Session
) -> "OrdersSyncResponse":
    """
    Run a Shopify orders sync → FBT edges merged into the workspace graph.

    Args:
        workspace_id: target workspace (already authorised by the caller).
        days: time window for orders (default 90). Older orders less
              relevant for current customer co-purchase patterns.
        min_support: minimum number of orders a (Product A, Product B) pair
                     must co-occur in before we emit an edge (default 2).
    """
    import time
    import httpx
    import networkx as nx

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

        # Before merging, strip stale `frequently_bought_with` edges from
        # the existing graph. Otherwise each sync silently keeps the FIRST
        # version's counts forever and never refreshes — the opposite of
        # 'smarter over time'. Catalog edges (variant_of, by_vendor,
        # in_collection, has_metafield) are preserved.
        gs = GraphifyService()
        existing_graph = await gs.load_graph(workspace_id)
        stale_fbt_removed = 0
        if existing_graph is not None:
            stale_fbt = [
                (u, v) for u, v, attrs in existing_graph.edges(data=True)
                if (attrs.get("relation") or "").lower() == "frequently_bought_with"
            ]
            for u, v in stale_fbt:
                existing_graph.remove_edge(u, v)
            stale_fbt_removed = len(stale_fbt)
            if stale_fbt:
                # Persist the cleaned graph back before the import_graph merge,
                # so the merge starts from a clean FBT slate. import_graph(merge=True)
                # will then ADD the fresh FBT edges from this sync run.
                fresh_data = nx.node_link_data(existing_graph)
                await gs._import_graph_unlocked(workspace_id, fresh_data, merge=False)

        # Pull total_orders BEFORE import_graph — that call mutates the dict
        # in place (renames "edges" -> "links" for NetworkX 3.x compat),
        # so reading graph_delta["edges"] AFTER would KeyError. Defensive
        # .get() either way.
        total_orders = 0
        delta_edges = graph_delta.get("edges") or graph_delta.get("links") or []
        if delta_edges:
            total_orders = int(delta_edges[0].get("attrs", {}).get("total_orders", 0))

        # Merge into the existing workspace graph (catalog nodes already
        # there from the products sync; we only add FBT edges).
        gs = GraphifyService()
        meta = await gs.import_graph(workspace_id, graph_delta, merge=True)

        duration = time.time() - t0
        settings["orders_sync"] = {
            "status": "complete",
            "bulk_operation_id": bulk_op_id,
            "object_count": object_count,
            "file_size": file_size,
            "days": days,
            "min_support": min_support,
            "fbt_edges_added": fbt_edges,
            "stale_fbt_removed": stale_fbt_removed,
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
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return the current orders_sync state stored on the workspace.

    PRD-172 F003: scoped to the authenticated workspace.
    """
    effective_ws = _resolve_sync_workspace(ctx, workspace_id)
    workspace = db.query(Workspace).get(effective_ws)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return (workspace.settings or {}).get("orders_sync") or {"status": "never_synced"}
