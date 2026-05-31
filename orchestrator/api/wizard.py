"""
PRD-130: Business Intake Wizard API
====================================

6 endpoints driving the wizard flow:

  POST   /api/wizard/start              → create business_profiles row
  POST   /api/wizard/scan/{profile_id}  → Firecrawl map + archetype detect
  POST   /api/wizard/scrape/{profile_id}→ 202 + background pipeline
  GET    /api/wizard/progress/{profile_id} → SSE live progress feed
  PATCH  /api/wizard/profile/{profile_id}→ user edits to the profile
  POST   /api/wizard/plan/{profile_id}  → generate Mission Zero draft plan

The scrape endpoint returns 202 immediately and runs the pipeline in the
background so Railway's edge proxy cannot kill long-running intake jobs
(~15min on a medium site). Clients open an EventSource against
``/progress/{profile_id}`` to watch every stage in real time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any
from uuid import UUID
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db, get_db_session
from core.models.business_profiles import BusinessProfile
from core.models.core import Agent
from core.utils.exception_telemetry import record_error

from modules.intake.archetypes import (
    ARCHETYPES,
    detect_archetype,
    select_target_urls,
)
from modules.intake.firecrawl_client import FirecrawlClient, FirecrawlError
from modules.intake.plan_generator import build_mission_goal
from modules.intake.profile_builder import build_profile
from modules.intake.progress import (
    STAGE_COMPLETE,
    STAGE_FAILED,
    STAGE_GRAPHIFY,
    STAGE_INGEST,
    STAGE_PLAN,
    STAGE_PROFILE,
    STAGE_SCAN,
    STAGE_SCRAPE,
    clear as progress_clear,
    emit as progress_emit,
    stream as progress_stream,
)
from modules.intake.schemas import pick_schema_for_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wizard", tags=["Business Intake Wizard"])


# ===========================================================================
# Pydantic models
# ===========================================================================

class StartBody(BaseModel):
    domain: str = Field(..., min_length=3, max_length=255)
    goals: list[str] = Field(default_factory=list)


class StartResponse(BaseModel):
    profile_id: str
    domain: str
    status: str
    domain_verified: bool


class ScanResponse(BaseModel):
    profile_id: str
    archetype: str | None
    confidence: float
    matched_signals: list[str]
    total_urls: int
    must_have_urls: list[str]
    recommended_urls: list[str]
    sample_urls: list[str]


class ScrapeBody(BaseModel):
    selected_urls: list[str] = Field(..., min_length=1)


class ScrapeAcceptedResponse(BaseModel):
    profile_id: str
    status: str
    message: str


class ProfilePatch(BaseModel):
    company_name: str | None = None
    sectors: list[str] | None = None
    brands: list[dict[str, Any]] | None = None
    standards: list[str] | None = None
    voice_notes: str | None = None
    goals: list[str] | None = None


class PlanResponse(BaseModel):
    profile_id: str
    mission_id: str
    goal: str


# ===========================================================================
# Helpers
# ===========================================================================

def _get_profile_or_404(
    db: Session, profile_id: str, workspace_id: Any
) -> BusinessProfile:
    try:
        pid = UUID(profile_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid profile_id")
    profile = (
        db.query(BusinessProfile)
        .filter(
            BusinessProfile.id == pid,
            BusinessProfile.workspace_id == workspace_id,
        )
        .first()
    )
    if profile is None:
        raise HTTPException(status_code=404, detail="Business profile not found")
    return profile


def _verify_domain_match(domain: str, ctx: RequestContext) -> bool:
    """Email-domain match check. Skipped when WIZARD_REQUIRE_DOMAIN_VERIFY=False."""
    if not config.WIZARD_REQUIRE_DOMAIN_VERIFY:
        return True
    user_email = (getattr(ctx, "user_email", None) or "").lower().strip()
    if not user_email or "@" not in user_email:
        return False
    user_domain = user_email.split("@", 1)[1].removeprefix("www.")
    target = domain.lower().strip().removeprefix("https://").removeprefix("http://")
    target = target.split("/", 1)[0].removeprefix("www.")
    return user_domain == target or user_domain.endswith(f".{target}")


def _firecrawl_client() -> FirecrawlClient:
    if not config.FIRECRAWL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Firecrawl is not configured (FIRECRAWL_API_KEY missing)",
        )
    return FirecrawlClient(
        api_key=config.FIRECRAWL_API_KEY,
        base_url=config.FIRECRAWL_BASE_URL,
        max_pages=config.FIRECRAWL_MAX_PAGES_PER_SCAN,
    )


def _slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = (parsed.path or "/").strip("/").replace("/", "_") or "home"
    return f"{parsed.netloc}_{path}"[:120]


# ===========================================================================
# Endpoints
# ===========================================================================

@router.post("/start", response_model=StartResponse, status_code=201)
async def start_wizard(
    body: StartBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> StartResponse:
    """Create a business profile for the workspace and return its id."""
    if not config.WIZARD_ENABLED:
        raise HTTPException(status_code=503, detail="Wizard disabled")

    domain = body.domain.strip().lower()
    domain = domain.removeprefix("https://").removeprefix("http://")
    domain = domain.split("/", 1)[0].removeprefix("www.")

    verified = _verify_domain_match(domain, ctx)
    if config.WIZARD_REQUIRE_DOMAIN_VERIFY and not verified:
        raise HTTPException(
            status_code=403,
            detail="Domain verification failed: your email domain does not match",
        )

    profile = BusinessProfile(
        workspace_id=ctx.workspace_id,
        domain=domain,
        goals=body.goals,
        status="started",
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)

    logger.info(
        "wizard.start workspace=%s profile=%s domain=%s goals=%s",
        ctx.workspace_id, profile.id, domain, body.goals,
    )

    return StartResponse(
        profile_id=str(profile.id),
        domain=profile.domain,
        status=profile.status,
        domain_verified=verified,
    )


@router.post("/scan/{profile_id}", response_model=ScanResponse)
async def scan_domain(
    profile_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> ScanResponse:
    """Run Firecrawl map + archetype detection. No scraping yet."""
    profile = _get_profile_or_404(db, profile_id, ctx.workspace_id)
    profile.status = "scanning"
    db.commit()

    await progress_clear(profile_id)
    await progress_emit(
        profile_id, STAGE_SCAN, f"Mapping {profile.domain} via Firecrawl…"
    )

    client = _firecrawl_client()
    try:
        urls = await client.map(profile.domain)
    except FirecrawlError as exc:
        logger.error("wizard.scan firecrawl_error: %s", exc, exc_info=True)
        profile.status = "failed"
        profile.quality_findings = {"errors": [f"Firecrawl map failed: {exc}"]}
        db.commit()
        await progress_emit(
            profile_id, STAGE_FAILED, f"Firecrawl map failed: {exc}", level="error"
        )
        raise HTTPException(status_code=502, detail=f"Firecrawl map failed: {exc}")

    detection = detect_archetype(urls)
    archetype = detection.archetype
    archetype_slug = archetype.slug if archetype else None

    bucketed = (
        select_target_urls(archetype, urls)
        if archetype
        else {"must": [], "recommended": []}
    )

    profile.raw_map_urls = urls
    profile.archetype = archetype_slug
    profile.status = "scanned"
    db.commit()

    await progress_emit(
        profile_id,
        STAGE_SCAN,
        f"Scan complete — found {len(urls)} URLs, archetype={archetype_slug or 'unknown'}",
        meta={
            "total_urls": len(urls),
            "archetype": archetype_slug,
            "confidence": detection.confidence,
        },
    )

    logger.info(
        "wizard.scan profile=%s urls=%d archetype=%s confidence=%.2f",
        profile.id, len(urls), archetype_slug, detection.confidence,
    )

    return ScanResponse(
        profile_id=str(profile.id),
        archetype=archetype_slug,
        confidence=detection.confidence,
        matched_signals=detection.matched_signals,
        total_urls=len(urls),
        must_have_urls=bucketed["must"],
        recommended_urls=bucketed["recommended"],
        sample_urls=urls[:50],
    )


@router.post(
    "/scrape/{profile_id}",
    response_model=ScrapeAcceptedResponse,
    status_code=202,
)
async def scrape_selected(
    profile_id: str,
    body: ScrapeBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> ScrapeAcceptedResponse:
    """Kick off the scrape→ingest→graphify→profile pipeline in the background.

    Returns 202 immediately so the HTTP connection doesn't have to stay
    open for the 10-20min graphify phase. The frontend watches
    ``GET /progress/{profile_id}`` for live updates and the final
    ``stage=complete`` event.
    """
    profile = _get_profile_or_404(db, profile_id, ctx.workspace_id)

    selected = body.selected_urls[: config.FIRECRAWL_MAX_PAGES_PER_SCAN]
    profile.selected_urls = selected
    profile.status = "scraping"
    db.commit()

    # Snapshot values needed by the background task so we don't carry the
    # request-scoped DB session into it.
    workspace_id = str(ctx.workspace_id)
    domain = profile.domain
    archetype = profile.archetype
    user_goals = list(profile.goals or [])

    await progress_clear(profile_id)
    await progress_emit(
        profile_id,
        STAGE_SCRAPE,
        f"Starting intake — {len(selected)} pages selected",
        meta={"total": len(selected)},
    )

    # Fire-and-forget; the task handles all its own errors and progress emits
    asyncio.create_task(
        _run_scrape_pipeline(
            profile_id=profile_id,
            workspace_id=workspace_id,
            domain=domain,
            archetype_slug=archetype,
            selected_urls=selected,
            user_goals=user_goals,
        )
    )

    logger.info(
        "wizard.scrape.accepted profile=%s workspace=%s urls=%d",
        profile_id, workspace_id, len(selected),
    )

    return ScrapeAcceptedResponse(
        profile_id=profile_id,
        status="scraping",
        message=f"Intake pipeline started for {len(selected)} pages",
    )


@router.get("/progress/{profile_id}")
async def progress_feed(
    profile_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> StreamingResponse:
    """Server-Sent Events stream of wizard pipeline progress.

    Replays buffered events then subscribes live. Returns when the
    pipeline emits a terminal stage (``complete`` or ``failed``).
    """
    # Verify the profile belongs to the caller's workspace before streaming
    with get_db_session() as db:
        _get_profile_or_404(db, profile_id, ctx.workspace_id)

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",  # disable nginx/Railway edge buffering
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        progress_stream(profile_id),
        media_type="text/event-stream",
        headers=headers,
    )


@router.patch("/profile/{profile_id}")
async def patch_profile(
    profile_id: str,
    body: ProfilePatch,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """User edits to the extracted profile (Step 6 of the wizard)."""
    profile = _get_profile_or_404(db, profile_id, ctx.workspace_id)

    updates = body.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(profile, field, value)
    db.commit()
    db.refresh(profile)

    return {
        "profile_id": str(profile.id),
        "updated_fields": list(updates.keys()),
        "status": profile.status,
    }


@router.get("/profile/{profile_id}")
async def get_profile(
    profile_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Fetch the current profile — used by the frontend after the SSE
    ``stage=complete`` event lands to pull the finished profile state."""
    profile = _get_profile_or_404(db, profile_id, ctx.workspace_id)
    return {
        "profile_id": str(profile.id),
        "domain": profile.domain,
        "archetype": profile.archetype,
        "company_name": profile.company_name,
        "sectors": profile.sectors,
        "brands": profile.brands,
        "standards": profile.standards,
        "voice_notes": profile.voice_notes,
        "goals": profile.goals,
        "quality_findings": profile.quality_findings,
        "status": profile.status,
        "draft_plan": profile.draft_plan,
    }


@router.post("/plan/{profile_id}", response_model=PlanResponse)
async def generate_plan(
    profile_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> PlanResponse:
    """Launch Mission Zero as a real mission in the Coordinator.

    Converts the scraped business profile into a rich natural-language
    goal and hands it to ``CoordinatorService.create_mission()``. The
    coordinator owns planning, task decomposition, agent dispatch and
    output summarization — this endpoint just kicks it off and returns
    the ``mission_id`` so the wizard can redirect to the mission page.
    """
    from services.coordinator_service import get_coordinator_service
    from modules.coordination.planner import PlanValidationError

    profile = _get_profile_or_404(db, profile_id, ctx.workspace_id)

    await progress_emit(
        profile_id, STAGE_PLAN, "Launching Mission Zero…"
    )

    archetype = ARCHETYPES.get(profile.archetype or "")
    default_team = list(archetype.default_team) if archetype else []

    # Mission Zero uses global onboarding agents (VOYAGER, BLUEPRINT,
    # SCRIBE, FORGE) seeded at startup.  Verify they exist; if the seed
    # didn't run for some reason, seed them now.
    _ensure_onboarding_agents(db)

    profile_dict = {
        "domain": profile.domain,
        "archetype": profile.archetype,
        "company_name": profile.company_name,
        "sectors": profile.sectors,
        "brands": profile.brands,
        "standards": profile.standards,
        "voice_notes": profile.voice_notes,
        "goals": profile.goals or [],
    }

    goal = build_mission_goal(profile_dict, archetype_default_team=default_team)

    mission_config = {
        "source": "mission_zero",
        "auto_approve": True,
        "skip_verification": True,  # Onboarding tasks don't need LLM judge — saves ~50% cost
        "profile_id": str(profile.id),
        "domain": profile.domain,
        "archetype": profile.archetype,
        "default_team": default_team,
    }

    coordinator = get_coordinator_service()

    try:
        run = await coordinator.create_mission(
            db=db,
            workspace_id=ctx.workspace_id,
            goal=goal,
            created_by=ctx.user.id or "unknown",
            config=mission_config,
        )
        profile.draft_plan = {
            "mission_run_id": str(run.id),
            "goal": goal,
        }
        profile.status = "planned"
        db.commit()
    except PlanValidationError as exc:
        db.rollback()
        logger.exception("wizard.plan coordinator rejected plan: %s", exc)
        await progress_emit(
            profile_id, STAGE_FAILED,
            f"Mission planning failed: {exc}", level="error",
        )
        raise HTTPException(
            status_code=422,
            detail=f"Mission planning failed: {exc}",
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("wizard.plan mission creation failed: %s", exc)
        await progress_emit(
            profile_id, STAGE_FAILED,
            f"Mission launch failed: {exc}", level="error",
        )
        raise HTTPException(status_code=500, detail=f"Mission launch failed: {exc}")

    await progress_emit(
        profile_id, STAGE_PLAN, "Mission Zero launched",
        meta={"mission_id": str(run.id)},
    )

    return PlanResponse(
        profile_id=str(profile.id),
        mission_id=str(run.id),
        goal=goal,
    )


# ===========================================================================
# Mission Zero onboarding agents — verification helper
# ===========================================================================


def _ensure_onboarding_agents(db: Session) -> None:
    """Verify the global onboarding agents exist; lazy-seed if missing.

    The 4 onboarding agents (VOYAGER, BLUEPRINT, SCRIBE, FORGE) are seeded
    at startup by ``seed_onboarding_agents``. This is a safety net for the
    rare case where the seed didn't run (e.g. first deploy before restart).
    """
    count = (
        db.query(Agent)
        .filter(
            Agent.is_system_agent.is_(True),
            Agent.required_role == "onboarding",
            Agent.status == "active",
        )
        .count()
    )
    if count >= 4:
        return

    logger.warning("Only %d onboarding agents found — lazy-seeding", count)
    from core.seeds.seed_onboarding_agents import seed_onboarding_agents
    seed_onboarding_agents(db)


# ===========================================================================
# Background pipeline
# ===========================================================================

async def _run_scrape_pipeline(
    *,
    profile_id: str,
    workspace_id: str,
    domain: str,
    archetype_slug: str | None,
    selected_urls: list[str],
    user_goals: list[str],
) -> None:
    """Full scrape → ingest → graphify → profile pipeline.

    Runs in an asyncio task spawned from ``scrape_selected``. Owns its
    own DB session and emits a progress event at every meaningful step.
    Any exception is caught and turned into a ``stage=failed`` event so
    the frontend always gets a terminal signal.
    """
    # TEMP: WIZARD_SKIP_GRAPHIFY=1 bypasses the slow knowledge-graph build
    # so we can iterate on Step 6 / Mission Zero without waiting for the
    # graph on every test run. The existing graph from a previous run is
    # reused. Remove before E2E testing.
    skip_graphify = os.getenv("WIZARD_SKIP_GRAPHIFY", "").lower() in ("1", "true", "yes")

    try:
        client = _firecrawl_client()

        scrape_results: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        total = len(selected_urls)

        # --- Scrape loop ------------------------------------------------
        for i, url in enumerate(selected_urls, start=1):
            await progress_emit(
                profile_id,
                STAGE_SCRAPE,
                f"[{i}/{total}] Scraping {url}",
                meta={"index": i, "total": total, "url": url},
            )
            schema, page_type = pick_schema_for_url(url)
            try:
                result = await client.scrape(url, schema=schema)
            except FirecrawlError as exc:
                logger.warning("wizard.scrape url=%s failed: %s", url, exc)
                failures.append({"url": url, "error": str(exc)})
                await progress_emit(
                    profile_id, STAGE_SCRAPE,
                    f"[{i}/{total}] FAILED {url}: {exc}",
                    level="warn",
                    meta={"index": i, "total": total, "url": url, "error": str(exc)},
                )
                continue
            except Exception as exc:  # noqa: BLE001 — last-resort, one bad URL must not kill the pipeline
                logger.exception("wizard.scrape url=%s unexpected error", url)
                failures.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
                await progress_emit(
                    profile_id, STAGE_SCRAPE,
                    f"[{i}/{total}] ERROR {url}: {type(exc).__name__}",
                    level="warn",
                    meta={"index": i, "total": total, "url": url, "error": str(exc)},
                )
                continue
            result["page_type"] = page_type
            scrape_results.append(result)
            await progress_emit(
                profile_id,
                STAGE_SCRAPE,
                f"[{i}/{total}] OK {url} ({page_type})",
                meta={
                    "index": i, "total": total, "url": url,
                    "page_type": page_type,
                    "chars": len((result.get("markdown") or "")),
                },
            )

        await progress_emit(
            profile_id,
            STAGE_SCRAPE,
            f"Scrape complete — {len(scrape_results)} ok, {len(failures)} failed",
            meta={"scraped": len(scrape_results), "failed": len(failures)},
        )

        # --- Ingest into RAG --------------------------------------------
        await progress_emit(
            profile_id, STAGE_INGEST,
            f"Ingesting {len(scrape_results)} pages into RAG pipeline…",
            meta={"total": len(scrape_results)},
        )
        documents_ingested = await _ingest_scrape_results_to_rag(
            scrape_results,
            workspace_id=workspace_id,
            domain=domain,
            profile_id=profile_id,
        )
        await progress_emit(
            profile_id, STAGE_INGEST,
            f"Ingest complete — {documents_ingested} documents persisted",
            meta={"ingested": documents_ingested},
        )

        # --- Graphify ---------------------------------------------------
        if skip_graphify:
            await progress_emit(
                profile_id, STAGE_GRAPHIFY,
                "Skipping graphify (WIZARD_SKIP_GRAPHIFY=1) — reusing existing graph",
                level="warn",
                meta={"skipped": True},
            )
        else:
            await progress_emit(
                profile_id, STAGE_GRAPHIFY,
                "Building knowledge graph (entity extraction)…",
            )
            try:
                from modules.knowledge.graph_service import GraphifyService
                graphify = GraphifyService()
                meta = await graphify.build_graph(workspace_id)
                await progress_emit(
                    profile_id, STAGE_GRAPHIFY,
                    f"Graph built — {meta.get('node_count', 0)} nodes, "
                    f"{meta.get('edge_count', 0)} edges, "
                    f"{meta.get('community_count', 0)} communities",
                    meta=meta,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "wizard pipeline graphify failed: %s", exc, exc_info=True
                )
                await progress_emit(
                    profile_id, STAGE_GRAPHIFY,
                    f"Graph build failed (non-fatal): {exc}",
                    level="warn",
                )

        # --- Profile build ----------------------------------------------
        await progress_emit(
            profile_id, STAGE_PROFILE,
            "Synthesising business profile from scraped content…",
        )
        profile_dict = build_profile(
            domain=domain,
            archetype_slug=archetype_slug or "unknown",
            scrape_results=scrape_results,
            user_goals=user_goals,
        )

        quality = profile_dict.get("quality_findings") or {"errors": [], "notes": []}
        if failures:
            quality.setdefault("errors", []).extend(
                f"{f['url']}: {f['error']}" for f in failures
            )

        # Persist profile fields in a fresh DB session
        with get_db_session() as db:
            profile = (
                db.query(BusinessProfile)
                .filter(BusinessProfile.id == UUID(profile_id))
                .first()
            )
            if profile is None:
                logger.error(
                    "wizard pipeline: profile %s vanished mid-run", profile_id
                )
                await progress_emit(
                    profile_id, STAGE_FAILED,
                    "Profile row disappeared mid-run",
                    level="error",
                )
                return

            profile.company_name = (
                profile_dict.get("company_name") or profile.company_name
            )
            profile.sectors = profile_dict.get("sectors")
            profile.brands = profile_dict.get("brands")
            profile.standards = profile_dict.get("standards")
            profile.voice_notes = profile_dict.get("voice_notes")
            profile.quality_findings = quality
            profile.status = "profiled"

        await progress_emit(
            profile_id, STAGE_PROFILE,
            f"Profile ready — {profile_dict.get('company_name') or domain} "
            f"({len(profile_dict.get('sectors') or [])} sectors, "
            f"{len(profile_dict.get('brands') or [])} brands)",
            meta={
                "company_name": profile_dict.get("company_name"),
                "sectors": len(profile_dict.get("sectors") or []),
                "brands": len(profile_dict.get("brands") or []),
            },
        )

        # --- Done -------------------------------------------------------
        await progress_emit(
            profile_id, STAGE_COMPLETE,
            "Intake complete — ready for review",
            meta={
                "scraped": len(scrape_results),
                "failed": len(failures),
                "ingested": documents_ingested,
            },
        )
        logger.info(
            "wizard.pipeline.complete profile=%s scraped=%d failed=%d ingested=%d",
            profile_id, len(scrape_results), len(failures), documents_ingested,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("wizard pipeline failed profile=%s: %s", profile_id, exc)
        record_error(
            subsystem="wizard",
            operation="scrape_pipeline",
            error=exc,
            workspace_id=workspace_id,
            extra={"profile_id": profile_id, "domain": domain},
        )
        # Mark the profile row failed so the UI can reflect it on reload
        try:
            with get_db_session() as db:
                profile = (
                    db.query(BusinessProfile)
                    .filter(BusinessProfile.id == UUID(profile_id))
                    .first()
                )
                if profile is not None:
                    profile.status = "failed"
                    profile.quality_findings = {
                        "errors": [f"Pipeline failed: {exc}"]
                    }
        except Exception:  # noqa: BLE001
            logger.exception("wizard pipeline: failed to mark profile failed")

        await progress_emit(
            profile_id, STAGE_FAILED,
            f"Pipeline failed: {exc}", level="error",
        )


# ===========================================================================
# RAG ingestion helper
# ===========================================================================

async def _ingest_scrape_results_to_rag(
    scrape_results: list[dict[str, Any]],
    workspace_id: str,
    domain: str,
    profile_id: str,
) -> int:
    """Write each scraped page to a temp .md file and upload via DocumentManager.

    Emits per-document progress events so the feed stays live through
    the ingest phase. Returns the count of successfully ingested documents.
    """
    if not scrape_results:
        return 0

    try:
        from api.documents import get_document_manager
    except Exception as exc:  # noqa: BLE001
        logger.error("wizard ingest: cannot import get_document_manager: %s", exc, exc_info=True)
        return 0

    try:
        doc_manager = get_document_manager(workspace_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("wizard ingest: DocumentManager init failed: %s", exc, exc_info=True)
        return 0

    ingested = 0
    total = len(scrape_results)
    with tempfile.TemporaryDirectory(prefix="wizard_intake_") as tmpdir:
        for i, result in enumerate(scrape_results, start=1):
            url = result.get("url", "")
            markdown = result.get("markdown") or ""
            if not markdown.strip():
                await progress_emit(
                    profile_id, STAGE_INGEST,
                    f"[{i}/{total}] SKIP empty page {url}",
                    level="warn",
                    meta={"index": i, "total": total, "url": url},
                )
                continue

            slug = _slug_from_url(url)
            filename = f"{slug}.md"
            file_path = os.path.join(tmpdir, filename)

            header = (
                f"# Source: {url}\n"
                f"# Domain: {domain}\n"
                f"# Page type: {result.get('page_type', 'generic')}\n\n"
            )
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(header + markdown)

            await progress_emit(
                profile_id, STAGE_INGEST,
                f"[{i}/{total}] Embedding {filename}",
                meta={
                    "index": i, "total": total,
                    "filename": filename, "url": url,
                },
            )
            try:
                await doc_manager.upload_document(
                    file_path=file_path,
                    filename=filename,
                    tags=["wizard", "intake", domain, result.get("page_type", "generic")],
                    description=f"Wizard intake from {url}",
                    created_by="wizard",
                )
                ingested += 1
                await progress_emit(
                    profile_id, STAGE_INGEST,
                    f"[{i}/{total}] INGESTED {filename}",
                    meta={
                        "index": i, "total": total,
                        "filename": filename, "url": url,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("wizard ingest: failed for %s: %s", url, exc, exc_info=True)
                await progress_emit(
                    profile_id, STAGE_INGEST,
                    f"[{i}/{total}] FAILED {filename}: {exc}",
                    level="warn",
                    meta={
                        "index": i, "total": total,
                        "filename": filename, "url": url, "error": str(exc),
                    },
                )

    return ingested
