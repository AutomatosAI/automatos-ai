"""PRD-203 O·S2 — the intake surface's FIRST behavioural tests.

The ~2,500-LOC business-intake path had zero dedicated coverage. This is its
first. Every boundary is mocked — Firecrawl, DocumentManager, GraphifyService,
Redis, the coordinator, and the DB session — so nothing live is touched.

Covers:
  (a) ``_run_scrape_pipeline`` happy-path: scrape → RAG-ingest → profile-build →
      terminal ``complete`` (Firecrawl + DocumentManager mocked);
  (b) one-bad-URL resilience: a single scrape failure does not kill the run;
  (c) the no-Redis degrade path (progress.py fallback emitter + stream);
  (d) the boot reaper sweeping a stranded ``scraping`` profile to ``failed``;
  (e) the payoff contract: ``/plan`` renders ``build_mission_goal()`` and calls
      ``coordinator.create_mission()`` with an honest, verified config (W2·S5).
"""
from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Fakes shared across the pipeline tests
# ---------------------------------------------------------------------------


class _FakeFirecrawl:
    """Fake FirecrawlClient: ``scrape`` returns markdown+extract, or raises for
    URLs listed in ``fail_urls`` (one-bad-URL resilience)."""

    def __init__(self, fail_urls: set[str] | None = None):
        self.fail_urls = fail_urls or set()
        self.scraped: list[str] = []

    async def scrape(self, url: str, schema=None):
        self.scraped.append(url)
        if url in self.fail_urls:
            from modules.intake.firecrawl_client import FirecrawlError

            raise FirecrawlError(f"boom {url}")
        return {
            "url": url,
            "markdown": "# About Acme\n" + ("content " * 60),
            "extract": {"company_name": "Acme", "industries_served": ["retail"]},
            "metadata": {},
        }


def _fake_db_session_ctx(profile):
    """A get_db_session() replacement yielding a db whose query→filter→first
    returns ``profile`` (the pipeline persists the built profile onto it)."""

    class _Ctx:
        def __enter__(self):
            db = MagicMock()
            db.query.return_value.filter.return_value.first.return_value = profile
            return db

        def __exit__(self, *a):
            return False

    return lambda: _Ctx()


def _install_pipeline_fakes(monkeypatch, wiz_mod, firecrawl, profile, emits):
    """Wire every external boundary of ``_run_scrape_pipeline`` to a fake."""
    monkeypatch.setattr(wiz_mod, "_firecrawl_client", lambda: firecrawl)

    async def fake_emit(profile_id, stage, message, *, level="info", meta=None):
        emits.append((stage, level))

    monkeypatch.setattr(wiz_mod, "progress_emit", fake_emit)
    monkeypatch.setattr(wiz_mod, "get_db_session", _fake_db_session_ctx(profile))

    # DocumentManager — the ingest seam (api.documents.get_document_manager).
    doc_mgr = SimpleNamespace(upload_document=AsyncMock())
    fake_documents = types.ModuleType("api.documents")
    fake_documents.get_document_manager = lambda ws: doc_mgr
    monkeypatch.setitem(sys.modules, "api.documents", fake_documents)

    # GraphifyService — a fast, successful stub (local import in the pipeline).
    fake_graph = types.ModuleType("modules.knowledge.graph_service")

    class _FakeGraphify:
        async def build_graph(self, ws):
            return {"node_count": 3, "edge_count": 2, "community_count": 1}

    fake_graph.GraphifyService = _FakeGraphify
    monkeypatch.setitem(sys.modules, "modules.knowledge.graph_service", fake_graph)

    return doc_mgr


def _blank_profile():
    return SimpleNamespace(
        company_name=None,
        sectors=None,
        brands=None,
        standards=None,
        voice_notes=None,
        quality_findings=None,
        status="scraping",
    )


# ---------------------------------------------------------------------------
# (a) happy-path — scrape → ingest → profile → complete
# ---------------------------------------------------------------------------


def test_run_scrape_pipeline_happy_path_reaches_complete(monkeypatch):
    from api import wizard as wiz_mod
    from modules.intake.progress import STAGE_COMPLETE, STAGE_FAILED

    firecrawl = _FakeFirecrawl()
    profile = _blank_profile()
    emits: list[tuple[str, str]] = []
    doc_mgr = _install_pipeline_fakes(monkeypatch, wiz_mod, firecrawl, profile, emits)

    urls = ["https://acme.io/about", "https://acme.io/pricing"]
    asyncio.run(
        wiz_mod._run_scrape_pipeline(
            profile_id=str(uuid4()),
            workspace_id=str(uuid4()),
            domain="acme.io",
            archetype_slug="saas_app",
            selected_urls=urls,
            user_goals=["grow revenue"],
        )
    )

    stages = [s for s, _ in emits]
    assert STAGE_COMPLETE in stages, f"pipeline never reached complete: {stages}"
    assert STAGE_FAILED not in stages
    assert firecrawl.scraped == urls
    assert doc_mgr.upload_document.await_count == 2  # both pages ingested
    # profile-build ran and persisted onto the row
    assert profile.company_name == "Acme"
    assert profile.status == "profiled"


# ---------------------------------------------------------------------------
# (b) one bad URL does not kill the run
# ---------------------------------------------------------------------------


def test_run_scrape_pipeline_survives_one_bad_url(monkeypatch):
    from api import wizard as wiz_mod
    from modules.intake.progress import STAGE_COMPLETE, STAGE_FAILED

    bad = "https://acme.io/broken"
    firecrawl = _FakeFirecrawl(fail_urls={bad})
    profile = _blank_profile()
    emits: list[tuple[str, str]] = []
    doc_mgr = _install_pipeline_fakes(monkeypatch, wiz_mod, firecrawl, profile, emits)

    urls = ["https://acme.io/about", bad, "https://acme.io/pricing"]
    asyncio.run(
        wiz_mod._run_scrape_pipeline(
            profile_id=str(uuid4()),
            workspace_id=str(uuid4()),
            domain="acme.io",
            archetype_slug="saas_app",
            selected_urls=urls,
            user_goals=[],
        )
    )

    stages = [s for s, _ in emits]
    # Terminal state is still complete, NOT failed — one bad URL is tolerated.
    assert STAGE_COMPLETE in stages
    assert STAGE_FAILED not in stages
    # Only the two good pages made it to ingest.
    assert doc_mgr.upload_document.await_count == 2
    # The failure is recorded in the profile's quality findings, not fatal.
    errors = (profile.quality_findings or {}).get("errors", [])
    assert any(bad in e for e in errors)


# ---------------------------------------------------------------------------
# (c) no-Redis degrade — emit is a safe no-op, stream yields one failure frame
# ---------------------------------------------------------------------------


def test_progress_emit_without_redis_is_noop(monkeypatch):
    from modules.intake import progress as progress_mod

    async def scenario():
        # Simulate Redis unavailable at the seam.
        monkeypatch.setattr(progress_mod, "_get_async_redis", AsyncMock(return_value=None))
        # Must not raise — progress events can never kill the pipeline.
        result = await progress_mod.emit("pid", progress_mod.STAGE_SCRAPE, "hi")
        assert result is None

    asyncio.run(scenario())


def test_progress_stream_without_redis_yields_failure_frame(monkeypatch):
    from modules.intake import progress as progress_mod

    async def scenario():
        monkeypatch.setattr(progress_mod, "_get_async_redis", AsyncMock(return_value=None))
        frames = [frame async for frame in progress_mod.stream("pid")]
        assert len(frames) == 1
        assert '"stage": "failed"' in frames[0]
        assert "unavailable" in frames[0].lower()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# (d) boot reaper sweeps a stranded scraping profile → failed
# ---------------------------------------------------------------------------


def test_boot_reaper_sweeps_stranded_scraping_profile(monkeypatch):
    from core.boot import reaper as reaper_mod

    monkeypatch.setattr(reaper_mod, "record_error", MagicMock())

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=30)
    stranded = SimpleNamespace(
        id=uuid4(),
        status="scraping",
        updated_at=now - timedelta(hours=2),  # older than cutoff → stale
        quality_findings=None,
    )
    fresh = SimpleNamespace(
        id=uuid4(),
        status="scanning",
        updated_at=now,  # newer than cutoff → left alone
        quality_findings=None,
    )

    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [stranded, fresh]

    reaped = reaper_mod._reap_business_profiles(db, cutoff, now)

    assert reaped == 1
    assert stranded.status == "failed"
    assert "orphaned_on_restart" in stranded.quality_findings["errors"][0]
    # The fresh in-flight row is untouched.
    assert fresh.status == "scanning"


# ---------------------------------------------------------------------------
# (e) /plan payoff contract — build_mission_goal + create_mission (honest config)
# PRD-222 W2·S5 retired the Mission Zero source-tag + auto_approve/skip_verification;
# /plan now launches a NORMAL verified mission that defaults to awaiting_approval.
# ---------------------------------------------------------------------------


def test_generate_plan_launches_an_honest_verified_mission(monkeypatch):
    from api import wizard as wiz_mod

    # Fake coordinator + planner (local imports inside generate_plan).
    run = SimpleNamespace(id="mission-123")
    coordinator = SimpleNamespace(create_mission=AsyncMock(return_value=run))

    fake_coord_mod = types.ModuleType("services.coordinator_service")
    fake_coord_mod.get_coordinator_service = lambda: coordinator
    monkeypatch.setitem(sys.modules, "services.coordinator_service", fake_coord_mod)

    fake_planner_mod = types.ModuleType("modules.coordination.planner")

    class _PlanValidationError(Exception):
        pass

    fake_planner_mod.PlanValidationError = _PlanValidationError
    monkeypatch.setitem(sys.modules, "modules.coordination.planner", fake_planner_mod)

    profile = SimpleNamespace(
        id=uuid4(),
        domain="acme.io",
        archetype="saas_app",
        company_name="Acme",
        sectors=["retail"],
        brands=[],
        standards=[],
        voice_notes=None,
        goals=["grow"],
        draft_plan=None,
        status="profiled",
    )
    monkeypatch.setattr(wiz_mod, "_get_profile_or_404", lambda db, pid, ws: profile)
    monkeypatch.setattr(wiz_mod, "progress_emit", AsyncMock())

    ctx = SimpleNamespace(workspace_id=uuid4(), user=SimpleNamespace(id="user-1"))
    db = MagicMock()

    result = asyncio.run(wiz_mod.generate_plan(profile_id=str(profile.id), ctx=ctx, db=db))

    coordinator.create_mission.assert_awaited_once()
    kwargs = coordinator.create_mission.await_args.kwargs
    cfg = kwargs["config"]
    # Honest config (D1/D7): the retired source-tag + trust-bypass flags are gone,
    # so the coordinator defaults auto_approve→False and verifies the build.
    assert "source" not in cfg
    assert "auto_approve" not in cfg
    assert "skip_verification" not in cfg
    assert cfg["default_team"], "the build got an empty default_team"
    assert "Acme" in kwargs["goal"]
    assert result.mission_id == "mission-123"
    assert profile.status == "planned"
