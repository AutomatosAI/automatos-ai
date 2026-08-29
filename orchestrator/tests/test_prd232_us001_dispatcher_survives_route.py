"""
PRD-232 US-001 — the ``platform_execute`` dispatcher survives every route() branch.
===================================================================================

The 2026-08-29 deep review found C1: ``SmartToolRouter.route()`` built its
graph- and category-branch keep-sets from chain names ∪ CORE_TOOLS ∪
always-include ∪ suggested, and ``platform_execute`` was in *none* of them.
The dispatcher — the single door to ~136 non-promoted platform actions — was
silently stripped on every graph-branch turn whose phrasing missed AutoBrain's
phrase map (no ``tool_hints=["platform"]`` substring rescue). That is the
2026-08-28 VECTOR failure: "close all the blocked tickets from vector" reached
Auto with the board-write route absent.

US-001 folds the dispatcher into the ONE always-include mechanism
(``_always_include_names``) so it is preserved on the hint, graph, and category
branches alike — with the intent-requires-no-tools branch the sole legitimate
exception (no tools at all is a valid surface).

These tests are hermetic: ``intent_classifier`` + ``smart_tool_router`` + ``auto``
are leaf-loaded under a synthetic package (stdlib/light imports only, no
DB-backed chat service), and the GraphRouter / ActionRegistry / telemetry
imports inside ``route()`` are faked in ``sys.modules`` — so no heavy
transformers/DB chain is ever touched.
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
_PKG = "_prd232_us001_chatbot"

VECTOR_SENTENCE = "close all the blocked tickets from vector"


def _load_modules():
    """Leaf-load intent_classifier + smart_tool_router + auto under a synthetic
    package so ``consumers.chatbot.__init__`` (DB-backed chat service) is never run."""
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
    auto_mod = _leaf("auto")  # top-level imports are stdlib + dispatch_contract (a string const)
    return router_mod, intent_mod, auto_mod


_router_mod, _intent_mod, _auto_mod = _load_modules()
SmartToolRouter = _router_mod.SmartToolRouter
ToolRoutingResult = _router_mod.ToolRoutingResult
Intent = _intent_mod.Intent
IntentResult = _intent_mod.IntentResult
AutoBrain = _auto_mod.AutoBrain


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


def _tool(name, description=None):
    return {"type": "function", "function": {"name": name, "description": description or f"{name} desc"}}


def _dispatcher_tool(enum_actions):
    """The realistic ``platform_execute`` dispatcher, shaped exactly as
    ``ActionRegistry.to_dispatcher_schema`` builds it — the door whose action
    enum makes each non-promoted platform action callable."""
    return {
        "type": "function",
        "function": {
            "name": "platform_execute",
            "description": "Execute an internal Automatos platform action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(enum_actions)},
                    "params": {"type": "object"},
                },
                "required": ["action", "params"],
            },
        },
    }


def _realistic_surface():
    """dispatcher + promoted first-class + core, as ``_get_tools_for_agent_core``
    assembles a full-path turn. The dispatcher's enum carries the board tools
    (incl. ``platform_update_task_status``) that the VECTOR turn needed."""
    return [
        _dispatcher_tool([
            "platform_list_tasks",
            "platform_update_task_status",
            "platform_create_task",
            "platform_get_board",
        ]),
        _tool("platform_find_tools"),      # promoted → first-class schema
        _tool("platform_list_agents"),     # core platform pin → first-class schema
        _tool("search_knowledge"),         # CORE_TOOLS
        _tool("composio_execute"),         # CORE_TOOLS
        _tool("generate_document"),        # CORE_TOOLS
        _tool("send_email", "send an email message to a recipient"),  # hint target
        _tool("totally_unrelated_tool"),   # droppable
    ]


def _dispatcher_from(result: ToolRoutingResult):
    hits = [t for t in result.filtered_tools if t.get("function", {}).get("name") == "platform_execute"]
    return hits[0] if hits else None


def _names(result: ToolRoutingResult):
    return {t.get("function", {}).get("name", "") for t in result.filtered_tools}


@pytest.fixture
def routing_env(monkeypatch):
    """Both routing flags ON (forward-compatible across US-002's flag split) and
    fake GraphRouter + ActionRegistry + telemetry injected into sys.modules."""
    from config import config as real_config
    # US-001: graph branch is gated on SEMANTIC_TOOL_ROUTING today; US-002 moves
    # it behind TOOL_ROUTING_GRAPH. Setting BOTH keeps this suite valid across
    # that split — the graph branch fires either way.
    monkeypatch.setattr(real_config, "SEMANTIC_TOOL_ROUTING", True, raising=False)
    monkeypatch.setattr(real_config, "TOOL_ROUTING_GRAPH", True, raising=False)

    state = {"errors": [], "rank_calls": []}

    def install_rank_chains(fn):
        fake_router = SimpleNamespace(rank_chains=fn)
        fake_mod = types.ModuleType("modules.tools.discovery.graph_router")
        fake_mod.get_graph_router = lambda: fake_router
        monkeypatch.setitem(sys.modules, "modules.tools.discovery.graph_router", fake_mod)

    def install_registry(promoted=("platform_find_tools",), by_category=None):
        by_category = by_category or {}
        reg = SimpleNamespace(
            get_promoted=lambda: [SimpleNamespace(name=n) for n in promoted],
            get_by_category=lambda cat: [SimpleNamespace(name=n) for n in by_category.get(cat, [])],
            get=lambda name: SimpleNamespace(
                name=name, promoted=False, admin_only=False, super_admin_only=False
            ),
        )
        fake_mod = types.ModuleType("modules.tools.discovery.action_registry")
        fake_mod.get_action_registry = lambda: reg
        monkeypatch.setitem(sys.modules, "modules.tools.discovery.action_registry", fake_mod)
        return reg

    fake_tel = types.ModuleType("core.utils.exception_telemetry")
    fake_tel.record_error = lambda **kw: state["errors"].append(kw)
    monkeypatch.setitem(sys.modules, "core.utils.exception_telemetry", fake_tel)

    state["install_rank_chains"] = install_rank_chains
    state["install_registry"] = install_registry
    return state


# ---------------------------------------------------------------------------
# AC1 — dispatcher present in filtered_tools on graph / hint / category branches
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_survives_graph_branch(routing_env):
    """Graph branch: rank_chains returns real chains, none of which is the
    dispatcher (it is not an ActionDefinition) — yet ``platform_execute`` is
    kept via the single always-include set."""
    routing_env["install_registry"]()

    async def ok_rank(query, agent_id=None, top_k=15, **kw):
        routing_env["rank_calls"].append((query, agent_id, top_k))
        return [
            ("platform_list_tasks", 0.9, ["platform_list_tasks"]),
            ("platform_get_board", 0.8, ["platform_get_board"]),
        ]

    routing_env["install_rank_chains"](ok_rank)

    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result(primary=Intent.DATA_QUERY))
    result = await r.route(query="how's the board looking", available_tools=_realistic_surface(), agent_id=7)

    assert result.reasoning.startswith("Graph routing"), "must have taken the graph branch"
    assert "platform_execute" in _names(result), "dispatcher stripped on the graph branch (C1 regression)"
    # ...and it was NOT rescued by ranking — the dispatcher is never a chain node.
    assert "platform_execute" not in {"platform_list_tasks", "platform_get_board"}


@pytest.mark.asyncio
async def test_dispatcher_survives_hint_branch(routing_env):
    """Hint branch: a NON-platform hint ("email") matches a real tool, and the
    dispatcher rides in via the always-include set — the deliberate keep, not
    the old accidental ``"platform" in "platform_execute"`` substring rescue."""
    routing_env["install_registry"]()
    # rank_chains must never be consulted on the hint branch.
    async def boom(query, **kw):
        raise AssertionError("graph must not run on the hint branch")

    routing_env["install_rank_chains"](boom)

    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result())
    result = await r.route(
        query="send an email to the team",
        available_tools=_realistic_surface(),
        tool_hints=["email"],  # deliberately NOT "platform"
        agent_id=7,
    )

    assert result.reasoning.startswith("Tool hints"), "must have taken the hint branch"
    names = _names(result)
    assert "send_email" in names, "hint target must be present"
    assert "platform_execute" in names, "dispatcher stripped on the hint branch (C1 regression)"


@pytest.mark.asyncio
async def test_dispatcher_survives_category_fallback(routing_env):
    """Category fallback: rank_chains returns nothing → route() drops through to
    ``_filter_tools_by_intent``, and the dispatcher survives the category filter."""
    routing_env["install_registry"](by_category={"tasks": ["platform_list_tasks", "platform_update_task_status"]})

    async def empty_rank(query, agent_id=None, top_k=15, **kw):
        return []  # no chains → fall through to category filtering

    routing_env["install_rank_chains"](empty_rank)

    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result(primary=Intent.MULTI_STEP))
    result = await r.route(query="do a few things on the board", available_tools=_realistic_surface(), agent_id=7)

    assert not result.reasoning.startswith("Graph routing"), "must have fallen through to category filtering"
    assert "platform_execute" in _names(result), "dispatcher stripped on the category fallback branch (C1 regression)"


# ---------------------------------------------------------------------------
# AC2 — the VECTOR replay: phrase-map miss + dispatcher callable to the action
# ---------------------------------------------------------------------------

def test_vector_sentence_misses_phrase_map():
    """The VECTOR write sentence does NOT hit AutoBrain's phrase map, so it never
    earns the accidental ``tool_hints=["platform"]`` rescue — the exact C1
    trigger condition the dispatcher fix must cover."""
    assert AutoBrain._match_platform_query(VECTOR_SENTENCE) is None


@pytest.mark.asyncio
async def test_vector_replay_dispatcher_makes_update_task_callable(routing_env):
    """Replay: the VECTOR sentence with NO platform hint (the honest post-miss
    state) still yields a surface whose ``platform_execute`` enum makes
    ``platform_update_task_status`` callable."""
    assert AutoBrain._match_platform_query(VECTOR_SENTENCE) is None  # no hint rescue
    routing_env["install_registry"]()

    async def ok_rank(query, agent_id=None, top_k=15, **kw):
        # The graph ranks board reads; the dispatcher is never a chain node.
        return [("platform_list_tasks", 0.9, ["platform_list_tasks"])]

    routing_env["install_rank_chains"](ok_rank)

    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result(primary=Intent.EXTERNAL_ACTION))
    result = await r.route(query=VECTOR_SENTENCE, available_tools=_realistic_surface(), tool_hints=[], agent_id=7)

    dispatcher = _dispatcher_from(result)
    assert dispatcher is not None, "VECTOR turn lost the dispatcher — the 2026-08-28 failure"
    enum = dispatcher["function"]["parameters"]["properties"]["action"].get("enum", [])
    assert "platform_update_task_status" in enum, "the board-write action is not callable via the dispatcher"


# ---------------------------------------------------------------------------
# AC3 — exactly one always-include mechanism
# ---------------------------------------------------------------------------

def test_dispatcher_folded_into_single_always_include_mechanism(routing_env):
    """One door list: ``platform_execute`` is a member of ``_always_include_names``
    (the single mechanism every branch unions in), not a second parallel pins pass."""
    import inspect

    routing_env["install_registry"]()
    r = SmartToolRouter()
    assert "platform_execute" in r._always_include_names()

    src = inspect.getsource(SmartToolRouter)
    # Exactly one producer of the always-include set.
    assert src.count("def _always_include_names") == 1
    # The dispatcher pin is folded INTO that one producer (not a rival pass).
    ai_src = inspect.getsource(SmartToolRouter._always_include_names)
    assert "_DISPATCHER_PINS" in ai_src
    # Every tool-shipping branch consults the same set: def + hint + graph + 2× category.
    assert src.count("_always_include_names()") >= 4


@pytest.mark.asyncio
async def test_no_tools_branch_ships_no_dispatcher(routing_env):
    """The sole legitimate exception: an intent that requires no tools returns an
    empty surface — the dispatcher is correctly absent."""
    routing_env["install_registry"]()
    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result(requires_tools=False))
    result = await r.route(query="hello there", available_tools=_realistic_surface(), agent_id=7)
    assert result.should_include_tools is False
    assert result.filtered_tools == []
