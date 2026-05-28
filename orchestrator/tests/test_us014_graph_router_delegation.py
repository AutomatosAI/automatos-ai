"""
PRD-141 US-014: SmartToolRouter delegates ranking to GraphRouter.
=================================================================

The chatbot tool router no longer carries its own embedding path. When
``SEMANTIC_TOOL_ROUTING`` is on, ``route()`` delegates ranking to
``GraphRouter.rank_chains`` (the single tool-selection pipeline) and filters
``available_tools`` to the ranked names, always preserving CORE_TOOLS /
ALWAYS_INCLUDE / the classifier's suggested tools. On GraphRouter failure it
records a structured ``routing`` error and falls back to category filtering.

``consumers.chatbot.__init__`` eagerly imports the DB-backed chat service, so
we leaf-load ``intent_classifier`` + ``smart_tool_router`` under a synthetic
package (both are stdlib-only at import time). The GraphRouter and
exception-telemetry imports live *inside* ``route()``; tests inject fakes for
them into ``sys.modules`` so no DB-backed module is ever imported.
"""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

_chatbot_dir = _orchestrator_root / "consumers" / "chatbot"
_PKG = "_us014_chatbot"


def _load_modules():
    """Leaf-load intent_classifier + smart_tool_router under a synthetic package."""
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_chatbot_dir)]
        sys.modules[_PKG] = pkg

    def _leaf(mod_name):
        full = f"{_PKG}.{mod_name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, _chatbot_dir / f"{mod_name}.py")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = _PKG
        sys.modules[full] = module
        spec.loader.exec_module(module)
        return module

    intent_mod = _leaf("intent_classifier")
    router_mod = _leaf("smart_tool_router")
    return router_mod, intent_mod


_router_mod, _intent_mod = _load_modules()
SmartToolRouter = _router_mod.SmartToolRouter
ToolRoutingResult = _router_mod.ToolRoutingResult
Intent = _intent_mod.Intent
IntentResult = _intent_mod.IntentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intent_result(primary=Intent.SEARCH, requires_tools=True, suggested=None,
                   reasoning="stub intent"):
    return IntentResult(
        primary_intent=primary,
        confidence=0.9,
        requires_tools=requires_tools,
        requires_memory=False,
        suggested_tools=suggested or [],
        reasoning=reasoning,
        is_simple=False,
    )


class _FakeClassifier:
    def __init__(self, result):
        self._result = result

    def classify(self, query, conversation_context=None):
        return self._result


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": f"{name} desc"}}


@pytest.fixture
def graph_env(monkeypatch):
    """Force semantic routing on and inject fake GraphRouter + telemetry modules."""
    from config import config as real_config
    monkeypatch.setattr(real_config, "SEMANTIC_TOOL_ROUTING", True)

    recorder = {"errors": [], "rank_calls": []}

    def install_rank_chains(fn):
        fake_router = SimpleNamespace(rank_chains=fn)
        fake_mod = types.ModuleType("modules.tools.discovery.graph_router")
        fake_mod.get_graph_router = lambda: fake_router
        monkeypatch.setitem(sys.modules, "modules.tools.discovery.graph_router", fake_mod)

    def fake_record_error(**kwargs):
        recorder["errors"].append(kwargs)

    fake_tel = types.ModuleType("core.utils.exception_telemetry")
    fake_tel.record_error = fake_record_error
    monkeypatch.setitem(sys.modules, "core.utils.exception_telemetry", fake_tel)

    recorder["install_rank_chains"] = install_rank_chains
    return recorder


# ---------------------------------------------------------------------------
# Deletion of the embedding path
# ---------------------------------------------------------------------------

def test_no_embedding_manager_on_smart_router():
    """The PRD-64 embedding state + methods are gone — GraphRouter owns ranking now."""
    r = SmartToolRouter()
    for attr in (
        "_embedding_manager",
        "_tool_embeddings",
        "_embeddings_initialized",
        "_embeddings_init_lock",
        "_ensure_embeddings",
        "_rank_tools_by_similarity",
    ):
        assert not hasattr(r, attr), f"{attr} should have been deleted in US-014"


def test_smart_router_module_has_no_embedding_imports():
    """No residual import of the local embedding stack."""
    import inspect
    src = inspect.getsource(_router_mod)
    assert "core.math.vector_operations" not in src
    assert "core.llm.embedding_manager" not in src


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_smart_router_delegates_to_graph_router(graph_env):
    """route() ranks via GraphRouter and filters available_tools to ranked names."""
    async def ok_rank(query, agent_id=None, top_k=15, **kw):
        graph_env["rank_calls"].append((query, agent_id, top_k))
        return [
            ("platform_get_system_health", 0.9, ["platform_get_system_health"]),
            ("platform_browse_marketplace_agents", 0.8, ["platform_browse_marketplace_agents"]),
        ]

    graph_env["install_rank_chains"](ok_rank)

    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result(suggested=[]))
    tools = [
        _tool("platform_get_system_health"),         # ranked
        _tool("platform_browse_marketplace_agents"),  # ranked
        _tool("search_knowledge"),                    # CORE_TOOLS — preserved
        _tool("platform_list_agents"),                # ALWAYS_INCLUDE — preserved
        _tool("totally_unrelated_tool"),              # dropped
    ]

    result = await r.route(query="how is the system doing", available_tools=tools, agent_id=42)

    names = {t["function"]["name"] for t in result.filtered_tools}
    assert "platform_get_system_health" in names
    assert "platform_browse_marketplace_agents" in names
    assert "search_knowledge" in names
    assert "platform_list_agents" in names
    assert "totally_unrelated_tool" not in names
    assert result.reasoning.startswith("Graph routing")
    # agent_id threaded through, top_k pinned to 30 per the PRD
    assert graph_env["rank_calls"] == [("how is the system doing", 42, 30)]


@pytest.mark.asyncio
async def test_always_include_tools_present(graph_env):
    """Every ALWAYS_INCLUDE tool survives graph filtering even when unranked."""
    async def ok_rank(query, agent_id=None, top_k=15, **kw):
        return [("platform_get_system_health", 0.9, ["platform_get_system_health"])]

    graph_env["install_rank_chains"](ok_rank)

    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result(suggested=[]))
    always = sorted(SmartToolRouter.ALWAYS_INCLUDE)
    tools = [_tool("platform_get_system_health"), _tool("totally_unrelated_tool")]
    tools += [_tool(n) for n in always]

    result = await r.route(query="status?", available_tools=tools, agent_id=1)

    names = {t["function"]["name"] for t in result.filtered_tools}
    for n in always:
        assert n in names, f"ALWAYS_INCLUDE tool {n} dropped"
    assert "platform_get_system_health" in names


@pytest.mark.asyncio
async def test_suggested_tools_preserved(graph_env):
    """The classifier's suggested tools survive graph filtering even when unranked."""
    async def ok_rank(query, agent_id=None, top_k=15, **kw):
        return [("platform_get_system_health", 0.9, ["platform_get_system_health"])]

    graph_env["install_rank_chains"](ok_rank)

    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result(suggested=["generate_document"]))
    tools = [_tool("platform_get_system_health"), _tool("generate_document"), _tool("unrelated")]

    result = await r.route(query="make me a report", available_tools=tools, agent_id=5)

    names = {t["function"]["name"] for t in result.filtered_tools}
    assert "generate_document" in names      # suggested → preserved
    assert "platform_get_system_health" in names
    assert "unrelated" not in names


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_router_fallback_on_failure(graph_env):
    """A GraphRouter failure records a 'routing' error and falls back to category filtering."""
    async def boom_rank(query, agent_id=None, top_k=15, **kw):
        raise RuntimeError("graph down")

    graph_env["install_rank_chains"](boom_rank)

    r = SmartToolRouter()
    r.classifier = _FakeClassifier(
        _intent_result(primary=Intent.MULTI_STEP, suggested=["search_knowledge"])
    )
    tools = [_tool("search_knowledge"), _tool("composio_execute"), _tool("totally_unrelated_tool")]

    result = await r.route(query="do a multi step thing", available_tools=tools, agent_id=7)

    # Structured error recorded under the routing subsystem (not a bare warning)
    assert len(graph_env["errors"]) == 1
    err = graph_env["errors"][0]
    assert err["subsystem"] == "routing"
    assert err["operation"] == "graph_rank_chains"
    assert err["agent_id"] == 7
    assert isinstance(err["error"], RuntimeError)

    # Fell through to category filtering — not the graph path
    assert result.should_include_tools is True
    assert not result.reasoning.startswith("Graph routing")
    names = {t["function"]["name"] for t in result.filtered_tools}
    assert "search_knowledge" in names


@pytest.mark.asyncio
async def test_semantic_off_skips_graph_router(graph_env, monkeypatch):
    """With SEMANTIC_TOOL_ROUTING off, GraphRouter is never called."""
    from config import config as real_config
    monkeypatch.setattr(real_config, "SEMANTIC_TOOL_ROUTING", False)

    called = {"n": 0}

    async def tracking_rank(query, agent_id=None, top_k=15, **kw):
        called["n"] += 1
        return [("x", 1.0, ["x"])]

    graph_env["install_rank_chains"](tracking_rank)

    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result(primary=Intent.MULTI_STEP, suggested=["search_knowledge"]))
    tools = [_tool("search_knowledge"), _tool("composio_execute")]

    result = await r.route(query="anything", available_tools=tools, agent_id=3)

    assert called["n"] == 0  # delegation gated off → straight to category filtering
    assert result.should_include_tools is True
