"""
PRD-232 US-004 — thread the scopes; reconcile the bootstrap seeds.
=================================================================

C9: two unreconciled defects on the catalog graph path.

(a) SmartChatOrchestrator built its ContextService context WITHOUT putting
    ``agent_id`` in ctx.kwargs, so PlatformActionsSection._build_graph_filtered's
    ``ctx.kwargs.get("agent_id")`` was permanently None — the catalog graph read
    could never apply per-agent edges/affinities. US-004 threads the real agent
    id through the orchestrator seam.

(b) PRD-143's metadata_graph_seed writes GLOBAL (workspace_id IS NULL)
    meta_sibling cold-start edges, but PRD-177 S5's per-tenant read lock filtered
    them out. US-004 first reconciled this by admitting NULL rows for meta_sibling
    ONLY. PRD-232 §6.5 (RVW-2, Gerard's ruling) then AMENDED that to the TWO-LAYER
    graph: a tenant read admits its own rows PLUS the text-free global prior for
    EVERY edge type, discounted so the tenant's own signal dominates (the AC3 greps
    below now assert the two-layer shape + the discount). The tenant-isolation
    coverage for (b) lives in test_prd177_graph_router_tenant.py (extended there);
    this file covers (a) end-to-end + the two-layer read grep.

The write-seam and grep assertions read source text directly (no heavy import);
the read-seam test drives the real PlatformActionsSection with a faked GraphRouter.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ORCH_ROOT = Path(__file__).resolve().parents[1]
_SMART_ORCH = _ORCH_ROOT / "consumers" / "chatbot" / "smart_orchestrator.py"
_GRAPH_ROUTER = _ORCH_ROOT / "modules" / "tools" / "discovery" / "graph_router.py"


# ---------------------------------------------------------------------------
# AC1 (write side) — the orchestrator seam threads agent_id into build_context
# ---------------------------------------------------------------------------

def test_orchestrator_seam_passes_agent_id_into_build_context():
    """SmartChatOrchestrator.build_context(...) call carries agent_id=self.agent_id
    so it lands in ctx.kwargs for the catalog graph path (C9 write side)."""
    src = _SMART_ORCH.read_text()
    # Isolate the build_context(...) call and assert agent_id is threaded in it.
    idx = src.find("build_context(")
    assert idx != -1, "build_context call not found in smart_orchestrator"
    call = src[idx: idx + 1500]
    assert "agent_id=self.agent_id" in call, (
        "orchestrator seam does not thread agent_id into build_context kwargs"
    )


# ---------------------------------------------------------------------------
# AC1 (read side) — PlatformActionsSection graph path consumes ctx.kwargs agent_id
# ---------------------------------------------------------------------------

@pytest.fixture
def graph_spy(monkeypatch):
    calls = []

    async def spy_rank(query, workspace_id=None, agent_id=None, top_k=15, **kw):
        calls.append({"workspace_id": workspace_id, "agent_id": agent_id})
        return []

    fake_router = SimpleNamespace(rank_chains=spy_rank)
    fake_gr = types.ModuleType("modules.tools.discovery.graph_router")
    fake_gr.get_graph_router = lambda: fake_router
    monkeypatch.setitem(sys.modules, "modules.tools.discovery.graph_router", fake_gr)
    return calls


@pytest.mark.asyncio
async def test_build_graph_filtered_threads_agent_id_from_ctx_kwargs(graph_spy):
    from modules.context.sections.platform_actions import PlatformActionsSection

    section = PlatformActionsSection()
    ctx = SimpleNamespace(kwargs={"agent_id": 4242}, workspace_id="ws-xyz")
    out = await section._build_graph_filtered("close the blocked tickets", ctx)

    assert out is None  # empty chains → None, but the graph WAS consulted
    assert graph_spy, "graph path never consulted"
    assert graph_spy[-1]["agent_id"] == 4242, "agent_id not threaded from ctx.kwargs into rank_chains"
    assert graph_spy[-1]["workspace_id"] == "ws-xyz"


@pytest.mark.asyncio
async def test_agent_blind_when_kwargs_missing_agent_id(graph_spy):
    """Regression guard: with no agent_id in ctx.kwargs the read is agent-blind
    (None) — which is exactly the C9 defect the write-side seam fixes."""
    from modules.context.sections.platform_actions import PlatformActionsSection

    section = PlatformActionsSection()
    ctx = SimpleNamespace(kwargs={}, workspace_id="ws-xyz")
    await section._build_graph_filtered("q", ctx)
    assert graph_spy[-1]["agent_id"] is None


# ---------------------------------------------------------------------------
# AC3 — no unfiltered global read remains for used_after / failed_after edges
# ---------------------------------------------------------------------------

def test_query_edges_two_layer_admits_global_prior():
    """PRD-232 §6.5 (RVW-2, Gerard's ruling — AMENDS US-004's meta_sibling-only NULL
    admission): ``_query_edges`` reads TWO layers — a tenant's own rows PLUS the
    text-free GLOBAL (workspace_id IS NULL) prior for EVERY edge type, not just
    meta_sibling. A tenant read is ``or_(workspace_id == workspace_id, workspace_id
    IS NULL)``; each global row is flagged ``is_global`` so the caller can discount
    it (the next test). The moat for tenant-SPECIFIC rows is proven behaviourally in
    test_prd177_graph_router_tenant.py::test_null_workspace_used_after_is_a_global_prior_not_a_leak."""
    src = _GRAPH_ROUTER.read_text()
    m = re.search(r"def _query_edges\(.*?def _query_affinities", src, re.S)
    assert m, "could not isolate _query_edges body"
    body = m.group(0)
    # Two-layer tenant read: own rows OR the global prior, any edge type.
    assert "workspace_id == workspace_id" in body
    assert "workspace_id.is_(None)" in body
    # Each row is tagged global/not so _expand_with_graph can discount the prior.
    assert "is_global" in body
    # The meta_sibling-only NULL gate (US-004) is gone under §6.5.
    assert "meta_global" not in body


def test_global_prior_is_discounted_not_equal_weight():
    """§6.5: a global (workspace_id IS NULL) edge IS admitted for a tenant, but as a
    REDUCED-weight prior — ``_expand_with_graph`` multiplies its confidence by the
    config prior factor on a tenant read, so a borrowed global edge never routes at
    the same weight as the tenant's own learned edge (a tenant edge of equal-or-higher
    confidence wins the dedup). This is what keeps the two-layer admission from being
    the old unfiltered equal-weight global read US-004 removed."""
    src = _GRAPH_ROUTER.read_text()
    m = re.search(r"def _expand_with_graph\(.*?def _match_query_vector", src, re.S)
    assert m, "could not isolate _expand_with_graph body"
    body = m.group(0)
    assert "_global_prior_factor" in body
    assert 'edge.get("is_global")' in body
    assert "prior_factor" in body
