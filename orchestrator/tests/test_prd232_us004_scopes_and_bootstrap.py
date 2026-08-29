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
    them out. US-004 reconciles: a tenant read admits its own rows exactly PLUS
    the unscoped meta_sibling seeds; used_after globals stay excluded. The
    tenant-isolation coverage for (b) lives in test_prd177_graph_router_tenant.py
    (extended there); this file covers (a) end-to-end + the no-unfiltered-read grep.

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

def test_query_edges_admits_null_only_for_meta_sibling():
    """The used_after edge read (_query_edges) reconciles NULL-workspace rows to
    meta_sibling only; a bare `workspace_id.is_(None)` fallback (the old global
    read) must be gone."""
    src = _GRAPH_ROUTER.read_text()
    # The reconciled filter: meta_global gates the NULL admission on meta_sibling.
    assert 'edge_type == "meta_sibling"' in src
    assert "meta_global" in src

    # Isolate _query_edges and assert its NULL handling is meta_sibling-gated,
    # never a naked global read of used_after.
    m = re.search(r"def _query_edges\(.*?def _query_affinities", src, re.S)
    assert m, "could not isolate _query_edges body"
    body = m.group(0)
    # A naked `workspace_id.is_(None)` NOT inside the meta_global and_() would be
    # the unfiltered global read. The only is_(None) here is the meta_global one.
    isnull_hits = body.count("workspace_id.is_(None)")
    assert isnull_hits == 1, (
        f"expected exactly one workspace_id.is_(None) (the meta_global gate), found {isnull_hits}"
    )
    # ...and it is co-located with the meta_sibling edge_type predicate.
    meta_idx = body.find("meta_global = and_(")
    assert meta_idx != -1
    meta_block = body[meta_idx: meta_idx + 200]
    assert "workspace_id.is_(None)" in meta_block and 'edge_type == "meta_sibling"' in meta_block


def test_no_bare_workspace_none_used_after_read():
    """Grep guard: no edge read ORs an unscoped used_after row in. The single
    NULL admission is the meta_global and_() (meta_sibling), asserted above."""
    src = _GRAPH_ROUTER.read_text()
    # There must be no filter that pairs is_(None) with used_after.
    assert 'edge_type == "used_after"' not in src or "is_(None)" not in src.split('edge_type == "used_after"')[0][-120:]
