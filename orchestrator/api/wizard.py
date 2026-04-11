"""
PRD-130: Business Intake Wizard API
====================================

5 endpoints driving the wizard flow:

  POST   /api/wizard/start              → create business_profiles row
  POST   /api/wizard/scan/{profile_id}  → Firecrawl map + archetype detect
  POST   /api/wizard/scrape/{profile_id}→ Firecrawl scrape → DocumentManager → graphify
  PATCH  /api/wizard/profile/{profile_id}→ user edits to the profile
  POST   /api/wizard/plan/{profile_id}  → generate Mission Zero draft plan

Phase 1 = blocking endpoints (no background queue). Good enough for <60s scrapes.
The /api/wizard/approve endpoint is intentionally NOT included — Mission 1 team
provisioning is parked as Phase 2 / TODO per PRD-130 review.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any
from uuid import UUID
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.business_profiles import BusinessProfile

from modules.intake.archetypes import (
    ARCHETYPES,
    detect_archetype,
    select_target_urls,
)
from modules.intake.firecrawl_client import FirecrawlClient, FirecrawlError
from modules.intake.plan_generator import generate_draft_plan
from modules.intake.profile_builder import build_profile
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


class ScrapeResponse(BaseModel):
    profile_id: str
    pages_scraped: int
    pages_failed: int
    documents_ingested: int
    profile: dict[str, Any]


class ProfilePatch(BaseModel):
    company_name: str | None = None
    sectors: list[str] | None = None
    brands: list[dict[str, Any]] | None = None
    standards: list[str] | None = None
    voice_notes: str | None = None
    goals: list[str] | None = None


class PlanResponse(BaseModel):
    profile_id: str
    draft_plan: dict[str, Any]


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

    client = _firecrawl_client()
    try:
        urls = await client.map(profile.domain)
    except FirecrawlError as exc:
        logger.error("wizard.scan firecrawl_error: %s", exc, exc_info=True)
        profile.status = "failed"
        profile.quality_findings = {"errors": [f"Firecrawl map failed: {exc}"]}
        db.commit()
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


@router.post("/scrape/{profile_id}", response_model=ScrapeResponse)
async def scrape_selected(
    profile_id: str,
    body: ScrapeBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> ScrapeResponse:
    """Scrape selected URLs, push to RAG, trigger graphify, build profile."""
    profile = _get_profile_or_404(db, profile_id, ctx.workspace_id)

    selected = body.selected_urls[: config.FIRECRAWL_MAX_PAGES_PER_SCAN]
    profile.selected_urls = selected
    profile.status = "scraping"
    db.commit()

    client = _firecrawl_client()

    scrape_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for url in selected:
        schema, page_type = pick_schema_for_url(url)
        try:
            result = await client.scrape(url, schema=schema)
        except FirecrawlError as exc:
            logger.warning("wizard.scrape url=%s failed: %s", url, exc)
            failures.append({"url": url, "error": str(exc)})
            continue
        result["page_type"] = page_type
        scrape_results.append(result)

    # Push to RAG via DocumentManager
    documents_ingested = await _ingest_scrape_results_to_rag(
        scrape_results, workspace_id=str(ctx.workspace_id), domain=profile.domain
    )

    # Trigger graphify build (best-effort, non-blocking on failure)
    try:
        from modules.knowledge.graph_service import GraphifyService

        graphify = GraphifyService()
        await graphify.build_graph(str(ctx.workspace_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("wizard.scrape graphify build failed: %s", exc, exc_info=True)

    # Build the profile dict
    profile_dict = build_profile(
        domain=profile.domain,
        archetype_slug=profile.archetype or "unknown",
        scrape_results=scrape_results,
        user_goals=profile.goals or [],
    )

    # Persist enriched fields
    profile.company_name = profile_dict.get("company_name") or profile.company_name
    profile.sectors = profile_dict.get("sectors")
    profile.brands = profile_dict.get("brands")
    profile.standards = profile_dict.get("standards")
    profile.voice_notes = profile_dict.get("voice_notes")

    quality = profile_dict.get("quality_findings") or {"errors": [], "notes": []}
    if failures:
        quality.setdefault("errors", []).extend(
            f"{f['url']}: {f['error']}" for f in failures
        )
    profile.quality_findings = quality
    profile.status = "profiled"
    db.commit()
    db.refresh(profile)

    logger.info(
        "wizard.scrape profile=%s scraped=%d failed=%d ingested=%d",
        profile.id, len(scrape_results), len(failures), documents_ingested,
    )

    return ScrapeResponse(
        profile_id=str(profile.id),
        pages_scraped=len(scrape_results),
        pages_failed=len(failures),
        documents_ingested=documents_ingested,
        profile={
            "domain": profile.domain,
            "archetype": profile.archetype,
            "company_name": profile.company_name,
            "sectors": profile.sectors,
            "brands": profile.brands,
            "standards": profile.standards,
            "voice_notes": profile.voice_notes,
            "goals": profile.goals,
            "quality_findings": profile.quality_findings,
        },
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


@router.post("/plan/{profile_id}", response_model=PlanResponse)
async def generate_plan(
    profile_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> PlanResponse:
    """Generate Mission Zero draft plan with graph-cited evidence."""
    profile = _get_profile_or_404(db, profile_id, ctx.workspace_id)

    archetype = ARCHETYPES.get(profile.archetype or "")
    default_team = list(archetype.default_team) if archetype else []

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

    graphify_service = None
    try:
        from modules.knowledge.graph_service import GraphifyService
        graphify_service = GraphifyService()
    except Exception as exc:  # noqa: BLE001
        logger.warning("wizard.plan graphify import failed: %s", exc)

    draft_plan = await generate_draft_plan(
        profile=profile_dict,
        archetype_default_team=default_team,
        workspace_id=str(ctx.workspace_id),
        graphify_service=graphify_service,
    )

    profile.draft_plan = draft_plan
    profile.status = "planned"
    db.commit()

    return PlanResponse(profile_id=str(profile.id), draft_plan=draft_plan)


# ===========================================================================
# RAG ingestion helper
# ===========================================================================

async def _ingest_scrape_results_to_rag(
    scrape_results: list[dict[str, Any]],
    workspace_id: str,
    domain: str,
) -> int:
    """Write each scraped page to a temp .md file and upload via DocumentManager.

    Returns the count of successfully ingested documents.
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
    with tempfile.TemporaryDirectory(prefix="wizard_intake_") as tmpdir:
        for result in scrape_results:
            url = result.get("url", "")
            markdown = result.get("markdown") or ""
            if not markdown.strip():
                continue

            slug = _slug_from_url(url)
            filename = f"{slug}.md"
            file_path = os.path.join(tmpdir, filename)

            # Prepend a small front-matter so RAG retrieval has provenance
            header = (
                f"# Source: {url}\n"
                f"# Domain: {domain}\n"
                f"# Page type: {result.get('page_type', 'generic')}\n\n"
            )
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(header + markdown)

            try:
                await doc_manager.upload_document(
                    file_path=file_path,
                    filename=filename,
                    tags=["wizard", "intake", domain, result.get("page_type", "generic")],
                    description=f"Wizard intake from {url}",
                    created_by="wizard",
                )
                ingested += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("wizard ingest: failed for %s: %s", url, exc, exc_info=True)

    return ingested
