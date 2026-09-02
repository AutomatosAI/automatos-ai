"""PRD-142 Wave 2 · WS-H — golden-journey backbone (J1–J10).

The ten journeys in ``docs/architecture/TEST-PLAN.md`` §5 are the platform's
load-bearing user flows. This module is their *backbone*: one ``golden``-marked
test per journey, selectable with ``-m golden``.

Honesty over theatre. There is no in-process FastAPI/TestClient harness in this
repo, and several journeys cross live external boundaries (Clerk auth, Shopify,
an LLM, S3, the scheduler). Faking a platform layer would test the fakes, not
the platform. So each journey is implemented at the highest layer that can be
proven *honestly today*:

* **Implemented now** (no external service, deterministic or real local PG):
  J2 reasoning-entry decision, J3 widget → vertical plugin → grounded response,
  J5 RAG ingest→select→assemble pipeline, J9 Shopify sync → Knowledge Graph →
  FBT proactive opener, J10 cross-workspace isolation (real Postgres, the P0
  security property). J3/J9 (PRD-189 S5) mock Shopify at the BOUNDARY only —
  recorded Bulk-Op JSONL fixtures stand in for Composio + the signed-URL
  download — while the mappers, the import/merge pipeline, and the widget
  plugin dispatch are the real production code.
* **Tracked gaps** — journeys that need infrastructure this suite can't stand up
  offline ``pytest.skip`` with a precise reason naming exactly what they need.
  They are not silent: ``-m golden`` lists them every run, so the backbone shows
  the whole map and the holes in it. Filling them is Wave 2.3 / Wave 3 work
  (recorded fixtures for the LLM legs, an app-level client fixture).

The implemented journeys compose primitives that W2-S9 (reasoning) and W2-S10
(RAG, verification) unit-test in isolation; here they are wired end-to-end so a
break *between* the units is caught, which the unit tests cannot see.
"""

from __future__ import annotations

import os
import sys
import types
from uuid import uuid4

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test_db")

# consumers/RAG import chains pull camelot (optional PDF dep, absent in test env).
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402

pytestmark = pytest.mark.golden


# ===========================================================================
# J1 — signup → onboarding (Mission Zero wizard)
# ===========================================================================


def test_j1_signup_to_onboarding():
    pytest.skip(
        "needs an app-level client + Clerk auth: the journey is "
        "POST /auth → workspace provisioning → Mission Zero wizard "
        "(VOYAGER/BLUEPRINT/SCRIBE/FORGE). No in-process FastAPI harness exists "
        "yet and Clerk can't be exercised offline. Fill in Wave 2.3 with an app "
        "client fixture + a stubbed auth principal."
    )


# ===========================================================================
# J2 — chat → reasoning entry → action routing
# The decision every chat turn makes, before any LLM/tool runs. Deterministic.
# ===========================================================================


@pytest.mark.asyncio
async def test_j2_chat_reasoning_entry_routes_message():
    """A chat turn enters AutoBrain.assess and emerges with the (complexity,
    action, tool) decision that downstream routing depends on. Two ends of the
    table: free chitchat resolves with no tools; a platform query routes to the
    very ``platform_list_agents`` tool whose isolation J10 then proves."""
    from consumers.chatbot.auto import Action, AutoBrain, Complexity

    brain = AutoBrain(db=MagicMock(), workspace_id="ws-golden")
    brain._redis = None
    brain._cache_lookup = lambda *a, **k: None
    brain._cache_store = lambda *a, **k: None

    greeting = await brain.assess("hey there")
    assert greeting.complexity is Complexity.ATOM
    assert greeting.action is Action.RESPOND
    assert greeting.tool_hints == []  # free turn, no tools, no spend

    platform = await brain.assess("list my agents")
    assert platform.complexity is Complexity.MOLECULE
    assert platform.action is Action.RESPOND
    # PRD-232 US-008: the phrase map is a booster, not a gate — it classifies
    # MOLECULE + tool_hints=["platform"] so the platform surface loads; the
    # specific tool is chosen by the ranker (dead matched_tools removed).
    assert platform.tool_hints == ["platform"]


# ===========================================================================
# J3 — widget → vertical plugin → response (generic AND shopify)
# ===========================================================================


@pytest.mark.asyncio
async def test_j3_widget_plugin_response(monkeypatch):
    """A widget message reaches the per-vertical plugin and comes back grounded
    (PRD-189 S5). Exactly the dispatch ``api/widgets/chat.py`` performs —
    ``PLUGIN_REGISTRY[vertical].handle_widget_message`` — driven directly:

    * **generic**: a regular turn with page context returns the message with
      the opaque ``(Context: ...)`` grounding prefix.
    * **shopify**: against the recorded INBUILD-flavoured fixture graph
      (``integrations/shopify/tests/fixtures/``), a ``proactive_opener``
      trigger is rewritten into a directive whose facts carry co-purchase
      provenance — the "bought together in X of Y orders" citation — and a
      regular shopper turn keeps its verbatim text but gains a
      ``[PAGE_CONTEXT]`` system preamble.

    No app-level client, no live store: the graph source is the one recorded
    fixture (the same boundary the US-011 byte-equality tests stub).
    """
    import json
    from pathlib import Path
    from uuid import uuid4 as _uuid4

    import modules.knowledge.graph_service as graph_service_mod
    from networkx.readwrite import json_graph
    from integrations import PLUGIN_REGISTRY

    # --- generic leg: opaque-context grounding, message otherwise untouched.
    generic = await PLUGIN_REGISTRY["generic"].handle_widget_message(
        message="do you ship to Ireland?",
        page_context={"pageType": "product"},
        trigger_reason=None,
        workspace_id=_uuid4(),
        db=MagicMock(),
    )
    assert generic.message.startswith('(Context: {"pageType": "product"})')
    assert generic.message.endswith("do you ship to Ireland?")

    # --- shopify leg: recorded fixture graph → grounded, provenance-cited.
    fixtures = (
        Path(__file__).resolve().parents[1]
        / "integrations" / "shopify" / "tests" / "fixtures"
    )
    graph_data = json.loads((fixtures / "inbuild_graph_snapshot.json").read_text())
    if "links" not in graph_data and "edges" in graph_data:
        graph_data["links"] = graph_data.pop("edges")
    fixture_graph = json_graph.node_link_graph(graph_data)

    class _FixtureGraphService:
        async def load_graph(self, _workspace_id):
            return fixture_graph

    monkeypatch.setattr(graph_service_mod, "GraphifyService", _FixtureGraphService)

    page_context = json.loads((fixtures / "product_page_context.json").read_text())
    shopify_plugin = PLUGIN_REGISTRY["shopify"]

    opener = await shopify_plugin.handle_widget_message(
        message="",
        page_context=page_context,
        trigger_reason="proactive_opener",
        workspace_id=_uuid4(),
        db=MagicMock(),
    )
    assert opener.message.startswith("[PROACTIVE_OPENER]")
    # The grounding facts are real graph provenance, not invention: the top
    # FBT pair with its co-purchase citation, straight from the fixture edges.
    assert (
        '"Hochiki YBN Detector Base" (bought together in 42 of 57 orders)'
        in opener.message
    )
    assert opener.telemetry["trigger_reason"] == "proactive_opener"
    assert opener.telemetry["related_count"] >= 1

    # A regular shopper turn: verbatim message + page-context preamble.
    regular = await shopify_plugin.handle_widget_message(
        message="what wattage is this detector?",
        page_context=page_context,
        trigger_reason=None,
        workspace_id=_uuid4(),
        db=MagicMock(),
    )
    assert regular.message == "what wattage is this detector?"
    assert regular.system_preamble is not None
    assert regular.system_preamble.startswith("[PAGE_CONTEXT]")
    assert "Hochiki ALN Optical Smoke Detector" in regular.system_preamble


# ===========================================================================
# J4 — mission lifecycle + restart durability
# ===========================================================================


def test_j4_mission_lifecycle_and_restart_durability():
    pytest.skip(
        "the restart-durability half is ALREADY a real-DB regression: see "
        "test_w2s6_restart_recovery_realdb.py (reap_orphaned_runs sweeps stale "
        "in-flight rows to terminal on boot) and test_w2s5_idle_in_tx_realdb.py. "
        "The create→plan→execute→verify half needs the coordinator + an LLM; "
        "fill with recorded LLM fixtures in Wave 3 rather than duplicate W2-S6."
    )


# ===========================================================================
# J5 — document → RAG ingest → retrieval-assembly
# The ingest→quality→budget→assemble pipeline, composed. Deterministic.
# ===========================================================================


def test_j5_document_rag_ingest_and_assembly():
    """A raw document flows through the real retrieval primitives end-to-end:
    chunk → score quality → token-budget-select → assemble a
    context block. W2-S10 unit-tests each primitive; this proves they *compose*
    into a usable, budget-respecting context — the break a unit test can't see."""
    from modules.rag.service import RAGService

    svc = RAGService.__new__(RAGService)  # bypass DB-reading __init__

    document = (
        "The quarterly revenue grew twenty percent driven by enterprise "
        "expansion and net revenue retention above one hundred and twenty "
        "percent across the install base this fiscal year overall. " * 8
    )

    # 1. Ingest: chunk the document.
    chunks = svc._basic_chunk(document)
    assert len(chunks) >= 2, "document should split into multiple chunks"

    # 2. Score + weigh each chunk (quality as value, length as token weight).
    texts = [c["content"] for c in chunks]
    values = [svc._calculate_content_quality(t) for t in texts]
    weights = [max(1, len(t) // 4) for t in texts]  # ~4 chars/token
    assert all(v > 0.0 for v in values)

    # 3. Select the best chunks that fit a tight token budget (PRD-157 budgeter
    #    replaced the knapsack DP; whole-chunk, score-ordered, within budget).
    from modules.rag.budget import select_within_budget

    budget = max(weights)  # only room for roughly one chunk
    candidates = [
        {"source_file": "q4.md", "similarity": values[i], "content": texts[i], "tokens": weights[i]}
        for i in range(len(texts))
    ]
    selection = select_within_budget(candidates, budget, max_chunks=len(texts))
    assert selection.chunks, "budgeter should select at least one chunk within budget"
    assert selection.total_tokens <= budget, "selection must fit the budget"

    # 4. Assemble the selected chunks into a context block with numbered citations.
    context = svc._format_context(selection.chunks, "how did revenue grow")
    assert "## Retrieved context for: how did revenue grow" in context
    assert any(c["content"][:20] in context for c in selection.chunks)


# ===========================================================================
# J6 — marketplace install → cascade → agent usable
# ===========================================================================


def test_j6_marketplace_install_cascade():
    pytest.skip(
        "needs DB + S3 + the install service: install a template → cascade "
        "(agents, skills, plugins, playbooks) → newly-installed agent is usable. "
        "Requires real S3 (plugin payloads) and the cascade installer. Fill with "
        "a local-DB + minio/S3-stub fixture in Wave 3."
    )


# ===========================================================================
# J7 — playbook schedule → run → recover
# ===========================================================================


def test_j7_playbook_schedule_run_recover():
    pytest.skip(
        "needs the scheduler + agent execution: schedule a cron Playbook → run → "
        "recover a failed step. Run/recover require the coordinator + an LLM. "
        "Fill with recorded LLM fixtures + a time-travel scheduler clock in Wave 3."
    )


# ===========================================================================
# J8 — NL2SQL
# ===========================================================================


def test_j8_nl2sql_query():
    pytest.skip(
        "PRD-142 lists NL2SQL as STRETCH, rolled to Wave 3. Needs DB + an LLM to "
        "translate NL → SQL → result. Fill with recorded LLM fixtures + a seeded "
        "local DB in Wave 3."
    )


# ===========================================================================
# J9 — Shopify sync → Knowledge Graph → FBT proactive opener
# (the marquee commerce journey: what the merchant installed Automatos FOR)
# ===========================================================================


@pytest.mark.asyncio
async def test_j9_shopify_sync_to_fbt_opener(monkeypatch):
    """The full marquee path, un-skipped (PRD-189 S5), with Shopify mocked at
    the boundary only — recorded catalog/orders Bulk-Op JSONL fixtures stand in
    for Composio + the signed-URL download; everything after that boundary is
    the real production code:

    1. recorded catalog JSONL → real ``map_shopify_catalog`` → real
       ``GraphifyService.import_graph`` (normalize/cluster/cache);
    2. recorded orders JSONL → real ``map_shopify_orders`` → real merge —
       ``frequently_bought_with`` edges land in the workspace Knowledge Graph;
    3. the real widget plugin (``_resolve_graph_related_products`` +
       ``_build_proactive_opener_message`` via ``handle_widget_message``)
       produces a provenance-cited opener from those edges;
    4. **the PRD-189 S1 guarantee, folded in**: a subsequent catalog re-sync —
       the REAL ``_product_sync_impl`` over an updated catalog fixture — leaves
       the FBT edges intact (before S1 this exact step wiped them to 0);
    5. the opener still cites the co-purchase pair afterwards.
    """
    from uuid import UUID, uuid4 as _uuid4

    import modules.knowledge.graph_service as graph_service_mod
    from integrations import PLUGIN_REGISTRY
    from modules.knowledge.graph_extraction import (
        map_shopify_catalog,
        map_shopify_orders,
    )
    from tests.helpers_shopify_sync import (
        CATALOG_JSONL_PATH,
        CATALOG_RESYNC_JSONL_PATH,
        ORDERS_JSONL_PATH,
        FakeDb,
        FakeWorkspace,
        fbt_edges_of,
        make_graph_service,
        mock_sync_boundaries,
        silence_graph_primitive,
    )

    workspace_id = str(_uuid4())
    silence_graph_primitive(monkeypatch)
    svc = make_graph_service()

    # 1. Catalog sync — recorded bulk-op JSONL through the real mapper+import.
    catalog = map_shopify_catalog(
        CATALOG_JSONL_PATH.read_text().splitlines(),
        bulk_op_id="gid://shopify/BulkOperation/j9-catalog",
    )
    await svc.import_graph(workspace_id, catalog, merge=False)

    # 2. Orders sync — FBT edges merge into the workspace Knowledge Graph.
    orders = map_shopify_orders(
        ORDERS_JSONL_PATH.read_text().splitlines(),
        bulk_op_id="gid://shopify/BulkOperation/j9-orders",
        min_support=2,
    )
    await svc.import_graph(workspace_id, orders, merge=True)

    graph = await svc.load_graph(workspace_id)
    assert len(fbt_edges_of(graph)) == 2, "orders sync must land the FBT edges"

    # 3. The widget grounds a proactive opener in those edges.
    monkeypatch.setattr(graph_service_mod, "GraphifyService", lambda: svc)
    page_context = {
        "pageType": "product",
        "productTitle": "Aurora Desk Lamp",
        "productType": "Lighting",
        "productVendor": "Lumenworks",
        "productPrice": "49.00",
        "productAvailable": True,
        "productHandle": "aurora-desk-lamp",
        "shopDomain": "fixture-lighting.myshopify.com",
        "shopCurrency": "GBP",
    }
    citation = '"Nimbus Bulb Duo" (bought together in 3 of 5 orders)'

    opener = await PLUGIN_REGISTRY["shopify"].handle_widget_message(
        message="",
        page_context=page_context,
        trigger_reason="proactive_opener",
        workspace_id=UUID(workspace_id),
        db=MagicMock(),
    )
    assert opener.message.startswith("[PROACTIVE_OPENER]")
    assert citation in opener.message, "the opener must cite real co-purchase data"

    # 4. S1's preservation guarantee: a catalog re-sync (product renamed,
    #    one product deleted from the store) keeps the cross-sell edges. The
    #    workspace carries the orders_sync block a real orders sync would have
    #    written (2 FBT edges reported — what step 2 merged), so the S2
    #    integrity check inside the re-sync has its reported side to compare.
    workspace = FakeWorkspace(
        workspace_id,
        settings={
            "shopify_domain": "fixture-lighting.myshopify.com",
            "orders_sync": {"status": "complete", "fbt_edges_added": 2},
        },
    )
    db = FakeDb(workspace)
    mock_sync_boundaries(
        monkeypatch,
        graph_service=svc,
        jsonl_texts=[CATALOG_RESYNC_JSONL_PATH.read_text()],
    )
    from api import shopify as shopify_api

    resync = await shopify_api._product_sync_impl(workspace_id, db)
    assert resync.status == "complete"

    graph_after = await svc.load_graph(workspace_id)
    pairs = {frozenset((u, v)): attrs for u, v, attrs in fbt_edges_of(graph_after)}
    lamp_bulbs = pairs.get(
        frozenset(("shopify_product_9001", "shopify_product_9002"))
    )
    assert lamp_bulbs is not None, "catalog re-sync must not wipe the FBT edge"
    assert lamp_bulbs.get("attrs", {}).get("co_count") == 3
    assert workspace.settings["product_sync"]["fbt_integrity"]["ok"] is True

    # 5. And the shopper-facing citation survives normal store activity.
    opener_after = await PLUGIN_REGISTRY["shopify"].handle_widget_message(
        message="",
        page_context=page_context,
        trigger_reason="proactive_opener",
        workspace_id=UUID(workspace_id),
        db=MagicMock(),
    )
    assert citation in opener_after.message


# ===========================================================================
# J10 — cross-workspace isolation (the P0 security property) — REAL Postgres
# ===========================================================================

_EPHEMERAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_EPHEMERAL_DBS = {"test_db", "test"}


@pytest.fixture(scope="module")
def safe_session():
    """A session against a *disposable* local Postgres only — never production.

    create_engine is lazy, so we read engine.url to gate on host/db before
    opening any connection (a Railway URL is rejected without a byte sent). Skips
    cleanly when no local PG is reachable, keeping the DB-less run green.
    """
    from sqlalchemy import text
    from sqlalchemy.orm import sessionmaker

    from core.database.database import engine

    url = engine.url
    if (url.host or "").lower() not in _EPHEMERAL_HOSTS or (
        url.database or ""
    ).lower() not in _EPHEMERAL_DBS:
        pytest.skip(
            f"refusing to run the write+commit isolation test against a "
            f"non-ephemeral database ({url.host}/{url.database})"
        )
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"no reachable Postgres for integration test: {exc}")

    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def two_workspaces(safe_session):
    """Commit two workspaces, each owning one agent. Tears down exactly those
    rows (a commit can't be rolled back)."""
    from core.models import Agent
    from core.models.workspaces import Workspace

    db = safe_session
    ws_a = Workspace(id=uuid4(), name="golden-j10-a")
    ws_b = Workspace(id=uuid4(), name="golden-j10-b")
    db.add_all([ws_a, ws_b])
    db.flush()

    atlas = Agent(
        name="Atlas Scout", agent_type="custom", slug="atlas-scout", workspace_id=ws_a.id
    )
    borg = Agent(
        name="Borg Drone", agent_type="custom", slug="borg-drone", workspace_id=ws_b.id
    )
    db.add_all([atlas, borg])
    db.commit()

    ids = types.SimpleNamespace(
        db=db, ws_a=ws_a.id, ws_b=ws_b.id, atlas=atlas.id, borg=borg.id
    )
    try:
        yield ids
    finally:
        # The seed rows were committed (a commit can't be rolled back), so they
        # must be DELETEd explicitly. The leading rollback only resets any
        # failed-transaction state the test body may have left so the DELETEs can
        # issue — it does NOT undo the seed. Agents are deleted before workspaces
        # (FK child-first) in one transaction; if any step raises we roll the
        # delete-tx back to leave a clean session for the next test and re-raise
        # so the leak is loud, never silent.
        db.rollback()
        try:
            db.query(Agent).filter(Agent.id.in_([ids.atlas, ids.borg])).delete(
                synchronize_session=False
            )
            db.query(Workspace).filter(
                Workspace.id.in_([ids.ws_a, ids.ws_b])
            ).delete(synchronize_session=False)
            db.commit()
        except Exception:
            db.rollback()
            raise


@pytest.mark.integration
def test_j10_agent_lookup_is_workspace_isolated(two_workspaces):
    """The real ``resolve_agent`` path (behind every platform agent tool) must
    never resolve an agent outside the caller's workspace — by name *or* by exact
    id. This is the P0 cross-workspace leak guard: drop the
    ``Agent.workspace_id == workspace_id`` filter and every cross assertion trips.
    """
    from modules.tools.discovery.handlers_assignments import resolve_agent

    ids = two_workspaces
    db = ids.db

    # In-workspace lookups succeed.
    a_agent, a_err = resolve_agent(db, ids.ws_a, {"agent_name": "Atlas"})
    assert a_err is None and a_agent is not None
    assert a_agent.id == ids.atlas and a_agent.workspace_id == ids.ws_a

    b_agent, b_err = resolve_agent(db, ids.ws_b, {"agent_id": ids.borg})
    assert b_err is None and b_agent is not None and b_agent.id == ids.borg

    # Cross-workspace by NAME is invisible.
    leaked, err = resolve_agent(db, ids.ws_a, {"agent_name": "Borg"})
    assert leaked is None and err is not None

    # Cross-workspace by EXACT id is still blocked (the strongest guarantee:
    # knowing the victim's primary key does not grant access).
    leaked_by_id, err2 = resolve_agent(db, ids.ws_a, {"agent_id": ids.borg})
    assert leaked_by_id is None and err2 is not None

    leaked_sym, err3 = resolve_agent(db, ids.ws_b, {"agent_id": ids.atlas})
    assert leaked_sym is None and err3 is not None
