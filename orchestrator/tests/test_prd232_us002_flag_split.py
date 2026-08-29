"""
PRD-232 US-002 — each flag gates its own surface (the C2 inversion fix).
=======================================================================

C2 (deep review 2026-08-29): the GraphRouter read in ``SmartToolRouter.route()``
was gated on ``SEMANTIC_TOOL_ROUTING`` (default ON), so the learned graph — held
dark for PRD-177 S4/S6 governance — ran on effectively every turn, while
``TOOL_ROUTING_GRAPH`` (default OFF, its intended gate) reached only the prompt
catalog. The flag was inverted across surfaces.

US-002 moves the schema-path graph read behind ``TOOL_ROUTING_GRAPH`` so:
- graph OFF  → zero GraphRouter queries on EITHER surface (schema + catalog);
- graph ON   → BOTH the schema path (SmartToolRouter.route) and the catalog path
  (PlatformActionsSection) consult the graph;
- ``SEMANTIC_TOOL_ROUTING`` keeps gating embedding narrowing everywhere (the
  dispatcher enum narrowing in tool_router + the embedding catalog path), so a
  graph-off turn still narrows semantically.

``smart_tool_router`` is leaf-loaded (light); the catalog + narrowing gates come
from the real ``PlatformActionsSection`` / ``tool_router`` (the split lives
across those modules). The GraphRouter is faked in ``sys.modules`` so no learned
graph / DB is ever touched.
"""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_chatbot_dir = _root / "consumers" / "chatbot"
_PKG = "_prd232_us002_chatbot"


def _leaf_load_router():
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_chatbot_dir)]
        sys.modules[_PKG] = pkg

    def _leaf(name):
        full = f"{_PKG}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, _chatbot_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = _PKG
        sys.modules[full] = module
        spec.loader.exec_module(module)
        return module

    _leaf("intent_classifier")
    return _leaf("smart_tool_router")


_router_mod = _leaf_load_router()
SmartToolRouter = _router_mod.SmartToolRouter
_intent_mod = sys.modules[f"{_PKG}.intent_classifier"]
Intent = _intent_mod.Intent
IntentResult = _intent_mod.IntentResult

# Real catalog + narrowing surfaces — the other half of the split.
from modules.context.sections.platform_actions import PlatformActionsSection
from modules.tools.tool_router import (
    _semantic_routing_enabled,
    _narrow_dispatcher_actions_async_inputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intent_result(primary=Intent.MULTI_STEP, requires_tools=True, suggested=None):
    return IntentResult(
        primary_intent=primary,
        confidence=0.9,
        requires_tools=requires_tools,
        requires_memory=False,
        suggested_tools=suggested or [],
        reasoning="stub intent",
        is_simple=False,
    )


class _FakeClassifier:
    def __init__(self, result):
        self._result = result

    def classify(self, query, conversation_context=None):
        return self._result


def _tool(name, description=None):
    return {"type": "function", "function": {"name": name, "description": description or f"{name} desc"}}


@pytest.fixture
def flags_and_spy(monkeypatch):
    """Fake GraphRouter (records every rank_chains call, returns []) + telemetry,
    and expose the real config singleton for per-test flag setting."""
    from config import config as real_config

    calls = []

    async def spy_rank(query, workspace_id=None, agent_id=None, top_k=15, **kw):
        calls.append({"query": query, "workspace_id": workspace_id, "agent_id": agent_id})
        return []  # empty is enough: the call itself proves the surface consulted the graph

    fake_router = SimpleNamespace(rank_chains=spy_rank)
    fake_gr = types.ModuleType("modules.tools.discovery.graph_router")
    fake_gr.get_graph_router = lambda: fake_router
    monkeypatch.setitem(sys.modules, "modules.tools.discovery.graph_router", fake_gr)

    fake_tel = types.ModuleType("core.utils.exception_telemetry")
    fake_tel.record_error = lambda **kw: None
    monkeypatch.setitem(sys.modules, "core.utils.exception_telemetry", fake_tel)

    return SimpleNamespace(config=real_config, calls=calls, mp=monkeypatch)


# ---------------------------------------------------------------------------
# AC1 — graph OFF + semantic ON: route() never calls rank_chains; narrowing runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_off_no_rank_chains_but_embedding_narrowing_runs(flags_and_spy):
    cfg = flags_and_spy.config
    flags_and_spy.mp.setattr(cfg, "TOOL_ROUTING_GRAPH", False)
    flags_and_spy.mp.setattr(cfg, "SEMANTIC_TOOL_ROUTING", True)

    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result())
    tools = [_tool("search_knowledge"), _tool("platform_execute"), _tool("composio_execute")]
    result = await r.route(query="close the blocked tickets", available_tools=tools, agent_id=9)

    # The learned graph is dark on the schema path.
    assert flags_and_spy.calls == [], "route() consulted the graph with TOOL_ROUTING_GRAPH off"
    assert not result.reasoning.startswith("Graph routing")

    # Embedding narrowing is unaffected — it keys off SEMANTIC, which is still ON.
    assert _semantic_routing_enabled() is True
    # None == "no skip reason" == narrowing proceeds (would run rank_actions).
    assert _narrow_dispatcher_actions_async_inputs("close the blocked tickets") is None

    # Catalog surface: embedding path live, graph path dark.
    section = PlatformActionsSection()
    assert section._semantic_routing_enabled() is True
    assert section._graph_routing_enabled() is False


@pytest.mark.asyncio
async def test_semantic_off_disables_narrowing_gate_only(flags_and_spy):
    """Symmetry check: turning SEMANTIC off skips embedding narrowing, and the
    graph read remains independently controlled by TOOL_ROUTING_GRAPH."""
    cfg = flags_and_spy.config
    flags_and_spy.mp.setattr(cfg, "SEMANTIC_TOOL_ROUTING", False)
    flags_and_spy.mp.setattr(cfg, "TOOL_ROUTING_GRAPH", False)

    assert _semantic_routing_enabled() is False
    skip = _narrow_dispatcher_actions_async_inputs("anything")
    assert skip is not None and skip.startswith("flag SEMANTIC_TOOL_ROUTING")
    assert PlatformActionsSection()._semantic_routing_enabled() is False


# ---------------------------------------------------------------------------
# AC2 — graph ON: BOTH the schema path and the catalog path consult the graph
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_on_both_surfaces_consult_graph(flags_and_spy):
    cfg = flags_and_spy.config
    flags_and_spy.mp.setattr(cfg, "TOOL_ROUTING_GRAPH", True)
    flags_and_spy.mp.setattr(cfg, "SEMANTIC_TOOL_ROUTING", True)

    # ── schema path: SmartToolRouter.route() ──
    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result(primary=Intent.DATA_QUERY))
    tools = [_tool("search_knowledge"), _tool("platform_execute")]
    await r.route(query="how's the board looking", available_tools=tools, agent_id=9)
    schema_calls = len(flags_and_spy.calls)
    assert schema_calls >= 1, "SmartToolRouter.route() did not consult the graph with the flag on"
    assert flags_and_spy.calls[-1]["workspace_id"] is None  # threaded through (no ws in this call)

    # ── catalog path: PlatformActionsSection ──
    section = PlatformActionsSection()
    assert section._graph_routing_enabled() is True  # render() routes into the graph method
    ctx = SimpleNamespace(kwargs={"agent_id": 9}, workspace_id="ws-42")
    out = await section._build_graph_filtered("how's the board looking", ctx)
    assert out is None  # empty chains → None, but the graph WAS consulted
    assert len(flags_and_spy.calls) > schema_calls, "PlatformActionsSection did not consult the graph"
    # per-tenant scoping (PRD-177 S5) threaded on the catalog call
    assert flags_and_spy.calls[-1]["workspace_id"] == "ws-42"
    assert flags_and_spy.calls[-1]["agent_id"] == 9


# ---------------------------------------------------------------------------
# AC3 — config.py documents the split; no os.getenv outside config.py for the flags
# ---------------------------------------------------------------------------

def test_config_docstrings_state_the_split():
    import inspect
    import config as config_mod

    src = inspect.getsource(config_mod)
    # Both flag lines carry the split rationale nearby (PRD-232 US-002 anchor).
    assert "PRD-232 US-002" in src
    assert "SEMANTIC_TOOL_ROUTING gates EMBEDDING narrowing" in src
    assert "TOOL_ROUTING_GRAPH (default OFF) gates the learned tool-routing GRAPH" in src


def test_no_os_getenv_for_routing_flags_outside_config():
    """The two routing flags are read via the config singleton only — the raw
    env read lives solely in config.py (the canonical-config rule). Production
    code (everything but config.py and the test tree) must be clean."""
    import subprocess

    root = str(_root)
    proc = subprocess.run(
        ["grep", "-rn", "-E", r'os\.getenv\(\s*["\'](SEMANTIC_TOOL_ROUTING|TOOL_ROUTING_GRAPH)["\']',
         "--include=*.py", root],
        capture_output=True, text=True,
    )
    offending = []
    for ln in proc.stdout.splitlines():
        if not ln.strip():
            continue
        path = ln.split(":", 1)[0]
        if path.endswith("/config.py") or "/tests/" in path:
            continue  # config.py is the canonical read; test files may document/set flags
        offending.append(ln)
    assert offending == [], f"os.getenv for routing flags outside config.py: {offending}"
