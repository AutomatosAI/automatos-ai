"""
PRD-232 US-008 — the phrase map becomes data, not a gate (+ kill the dead field).
=================================================================================

C6: ``ComplexityAssessment.matched_tools`` was written by AutoBrain's MOLECULE
fast-path (auto.py) and read only by a log f-string (api/chat.py) — a dead
pre-selection. US-008 deletes it. ``_match_platform_query`` stays a fast-path
BOOSTER — it still classifies MOLECULE + tool_hints=["platform"] so the platform
surface loads — but the *specific* tool is now chosen by the ranker from the
seeded corpus (US-005/006), not gated by the phrase map.

The trap (spec §7): the hint branch used to rescue the dispatcher only by the
accident ``"platform" in "platform_execute"`` (smart_tool_router.py). US-001
made dispatcher survival deliberate (one always-include set). This suite is the
PARITY proof: the dispatcher is present whether or not the phrase map hits —
tool_hints=["platform"] (hit), tool_hints=[] (miss), and tool_hints=["email"]
(a non-platform hint, proving the substring accident is no longer load-bearing).

Hermetic: auto + smart_tool_router are leaf-loaded under a synthetic package and
the router's GraphRouter / ActionRegistry / telemetry imports are faked, so no
DB/transformers chain is touched (mirrors the US-001 suite).
"""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(_ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCH_ROOT))

_CHATBOT_DIR = _ORCH_ROOT / "consumers" / "chatbot"
_PKG = "_prd232_us008_chatbot"


def _load():
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_CHATBOT_DIR)]
        sys.modules[_PKG] = pkg

    def _leaf(mod_name):
        full = f"{_PKG}.{mod_name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, _CHATBOT_DIR / f"{mod_name}.py")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = _PKG
        sys.modules[full] = module
        spec.loader.exec_module(module)
        return module

    return _leaf("intent_classifier"), _leaf("smart_tool_router"), _leaf("auto")


_intent_mod, _router_mod, _auto_mod = _load()
SmartToolRouter = _router_mod.SmartToolRouter
ToolRoutingResult = _router_mod.ToolRoutingResult
Intent = _intent_mod.Intent
IntentResult = _intent_mod.IntentResult
AutoBrain = _auto_mod.AutoBrain
ComplexityAssessment = _auto_mod.ComplexityAssessment
Complexity = _auto_mod.Complexity
Action = _auto_mod.Action


# ── helpers (mirror the US-001 suite) ────────────────────────────────────────
def _tool(name, description=None):
    return {"type": "function", "function": {"name": name, "description": description or f"{name} desc"}}


def _dispatcher_tool(enum_actions):
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
    return [
        _dispatcher_tool([
            "platform_list_tasks", "platform_update_task_status",
            "platform_create_task", "platform_get_board",
        ]),
        _tool("platform_find_tools"),
        _tool("platform_list_agents"),
        _tool("search_knowledge"),
        _tool("composio_execute"),
        _tool("send_email", "send an email message to a recipient"),
        _tool("totally_unrelated_tool"),
    ]


def _names(result):
    return {t.get("function", {}).get("name", "") for t in result.filtered_tools}


def _intent_result(primary=Intent.SEARCH, requires_tools=True, suggested=None):
    return IntentResult(
        primary_intent=primary, confidence=0.9, requires_tools=requires_tools,
        requires_memory=False, suggested_tools=suggested or [],
        reasoning="stub", is_simple=False,
    )


class _FakeClassifier:
    def __init__(self, result):
        self._result = result

    def classify(self, query, conversation_context=None):
        return self._result


@pytest.fixture
def routing_env(monkeypatch):
    from config import config as real_config
    monkeypatch.setattr(real_config, "SEMANTIC_TOOL_ROUTING", True, raising=False)
    monkeypatch.setattr(real_config, "TOOL_ROUTING_GRAPH", True, raising=False)

    def install_rank_chains(fn):
        fake_mod = types.ModuleType("modules.tools.discovery.graph_router")
        fake_mod.get_graph_router = lambda: SimpleNamespace(rank_chains=fn)
        monkeypatch.setitem(sys.modules, "modules.tools.discovery.graph_router", fake_mod)

    def install_registry(by_category=None):
        by_category = by_category or {}
        reg = SimpleNamespace(
            get_promoted=lambda: [SimpleNamespace(name="platform_find_tools")],
            get_by_category=lambda cat: [SimpleNamespace(name=n) for n in by_category.get(cat, [])],
            get=lambda name: SimpleNamespace(name=name, promoted=False, admin_only=False, super_admin_only=False),
        )
        fake_mod = types.ModuleType("modules.tools.discovery.action_registry")
        fake_mod.get_action_registry = lambda: reg
        monkeypatch.setitem(sys.modules, "modules.tools.discovery.action_registry", fake_mod)

    fake_tel = types.ModuleType("core.utils.exception_telemetry")
    fake_tel.record_error = lambda **kw: None
    monkeypatch.setitem(sys.modules, "core.utils.exception_telemetry", fake_tel)

    return {"install_rank_chains": install_rank_chains, "install_registry": install_registry}


async def _route(hints, primary=Intent.SEARCH):
    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result(primary=primary))
    return await r.route(query="q", available_tools=_realistic_surface(), tool_hints=hints, agent_id=7)


# ── AC2: the dispatcher is present whether or not the phrase map hit ──────────
@pytest.mark.asyncio
async def test_dispatcher_present_on_phrase_map_hit_and_miss(routing_env):
    routing_env["install_registry"](by_category={"tasks": ["platform_list_tasks", "platform_update_task_status"]})

    async def empty_rank(query, agent_id=None, top_k=15, **kw):
        return []  # miss path falls through to category filtering

    routing_env["install_rank_chains"](empty_rank)

    # phrase-map HIT: AutoBrain would emit tool_hints=["platform"] → hint branch
    hit = await _route(["platform"])
    assert "platform_execute" in _names(hit), "dispatcher missing on the phrase-map-hit surface"

    # phrase-map MISS: no hints → graph/category branch
    miss = await _route([])
    assert "platform_execute" in _names(miss), "dispatcher missing on the phrase-map-miss surface"


@pytest.mark.asyncio
async def test_dispatcher_survival_not_a_platform_substring_accident(routing_env):
    """A non-platform hint ("email") still keeps the dispatcher — proving US-001's
    always-include is the mechanism, not the old ``"platform" in "platform_execute"``
    substring rescue the phrase map used to trigger."""
    routing_env["install_registry"]()

    async def boom(query, **kw):
        raise AssertionError("graph must not run on the hint branch")

    routing_env["install_rank_chains"](boom)

    result = await _route(["email"])
    names = _names(result)
    assert "send_email" in names
    assert "platform_execute" in names, "dispatcher stripped when the hint was not 'platform'"


# ── AC3: fast-path parity — phrase-map hits still classify MOLECULE ───────────
@pytest.mark.asyncio
async def test_phrase_map_hit_classifies_molecule_with_platform_hint():
    # _match_platform_query is a staticmethod over the lowercased message.
    assert AutoBrain._match_platform_query("list my agents") is not None, "known phrase must match"
    # a genuinely novel sentence misses the map (would fall to the LLM tier live);
    # its vocabulary now lives in the corpus, not the gate.
    assert AutoBrain._match_platform_query("zxqw novel unmapped gibberish 12345") is None


def test_molecule_fastpath_shape_is_platform_hint_only():
    """The surviving fast-path assessment carries tool_hints=["platform"] and no
    matched_tools field."""
    a = ComplexityAssessment(
        complexity=Complexity.MOLECULE, action=Action.RESPOND,
        reasoning="Platform query (platform_list_agents)",
        tool_hints=["platform"], confidence=0.90,
    )
    assert a.tool_hints == ["platform"]
    assert not hasattr(a, "matched_tools"), "the dead matched_tools field must be gone"


# ── AC1: matched_tools fully deleted — no dead writes, no dead reads ──────────
def test_matched_tools_field_and_writes_are_gone():
    assert "matched_tools" not in ComplexityAssessment.__dataclass_fields__

    import re
    for rel in ("consumers/chatbot/auto.py", "api/chat.py"):
        src = (_ORCH_ROOT / rel).read_text()
        # no write (matched_tools=...) and no attribute read (.matched_tools),
        # ignoring the explanatory comments that document the removal.
        code = "\n".join(
            line.split("#", 1)[0] for line in src.splitlines()
        )
        assert not re.search(r"matched_tools\s*=", code), f"{rel}: dead matched_tools write remains"
        assert ".matched_tools" not in code, f"{rel}: dead matched_tools read remains"
