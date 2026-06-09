"""Tests for PlatformActionsSection (PRD-138 US-004).

These tests stub out ``config``, ``ActionRegistry`` and ``ActionSemanticIndex``
imports so the section can be loaded without pulling in DB / Redis / live
registrars. Each scenario exercises one branch of the decision tree:

- flag enabled + query present  → filtered render via ActionSemanticIndex
- flag disabled                 → full dump (index never touched)
- index raises                  → fall back to full dump
- no ``query`` key in kwargs    → full dump
- empty ``query`` string        → full dump
- filtered output is shorter    → token-count surrogate for the prod-index AC

Collection-pollution discipline (PRD-142 W2-S2b): pytest imports every test
module during collection, before any test runs, so import-time ``sys.modules``
fakes leak into sibling collection. We therefore install import-time stubs
(estimator / base / section) only long enough to bind the classes and then
restore sys.modules; the runtime stubs (config / action_registry /
action_semantic_index) that the section imports lazily inside ``render()`` are
(re)installed by an autouse fixture during the test phase and torn down after
each test. Importing this module leaves sys.modules exactly as it found it.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

_THIS = Path(__file__).resolve()
_ORCH = _THIS.parents[1]
_SECTIONS = _ORCH / "modules" / "context" / "sections"
_DISCOVERY = _ORCH / "modules" / "tools" / "discovery"


# ---------------------------------------------------------------------------
# Watched sys.modules keys. Snapshot BEFORE we touch anything, restore after
# the import-time block so collection of sibling modules sees no fakes.
# ---------------------------------------------------------------------------
_IMPORT_KEYS = (
    "modules.context.estimator",
    "modules.context.sections",
    "modules.context.sections.base",
    "modules.context.sections.platform_actions",
)
# Read by the section's LAZY imports at render() time — must be live during the
# test phase, installed per-test by the autouse fixture (never at collection).
_RUNTIME_KEYS = (
    "config",
    "modules.tools.discovery.action_registry",
    "modules.tools.discovery.action_semantic_index",
)
# Parent packages we never intend to create; snapshot+restore as a guard so a
# stray real/fake parent can't survive import either.
_GUARD_KEYS = (
    "modules",
    "modules.context",
    "modules.tools",
    "modules.tools.discovery",
)
_PRE_IMPORT_SNAPSHOT = {
    k: sys.modules.get(k) for k in (_IMPORT_KEYS + _RUNTIME_KEYS + _GUARD_KEYS)
}


def _restore(snapshot: dict) -> None:
    """Restore sys.modules to a captured snapshot (None ⇒ delete the key)."""
    for key, original in snapshot.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original


# ---------------------------------------------------------------------------
# Pre-load the real ActionRegistry under a private name (does not pollute the
# watched ``modules.*`` namespace) so build_*_prompt_summary works without the
# live ``register_all_actions`` lazy-init.
# ---------------------------------------------------------------------------
_ar_spec = importlib.util.spec_from_file_location(
    "action_registry_under_test", _DISCOVERY / "action_registry.py"
)
_ar_mod = importlib.util.module_from_spec(_ar_spec)
sys.modules["action_registry_under_test"] = _ar_mod
_ar_spec.loader.exec_module(_ar_mod)
ActionDefinition = _ar_mod.ActionDefinition
ActionRegistry = _ar_mod.ActionRegistry


# ---------------------------------------------------------------------------
# Build stub module objects. Kept as module-level references; installed into
# sys.modules only transiently for the import below, then per-test by the
# autouse fixture.
# ---------------------------------------------------------------------------
class _StubConfig:
    """Mutable config stand-in for the canonical ``config.config`` singleton."""

    SEMANTIC_TOOL_ROUTING: bool = True
    SEMANTIC_TOOL_ROUTING_TOP_K: int = 3
    PLATFORM_ACTIONS_MAX_TOKENS: int = 4000


_stub_config_module = ModuleType("config")
_stub_config_module.config = _StubConfig()


class _NoopEstimator:
    def estimate(self, content: str) -> int:
        # Cheap surrogate so we don't depend on tiktoken here.
        return len(content) // 4 + 1


_estimator_mod = ModuleType("modules.context.estimator")
_estimator_mod.TokenEstimator = _NoopEstimator


# Real ActionRegistry classes exposed under the dotted name the section imports.
_ar_module = ModuleType("modules.tools.discovery.action_registry")
_ar_module.ActionDefinition = ActionDefinition
_ar_module.ActionRegistry = ActionRegistry
# get_action_registry assigned per-test via _install_registry().


class _FakeSemanticIndex:
    """Configurable async fake for ActionSemanticIndex."""

    def __init__(self) -> None:
        self.calls: List[dict] = []
        # Default return — the section ignores scores for filtering, just names.
        self.next_result: List[Tuple[str, float]] = []
        self.exception: Exception | None = None

    async def rank_actions(
        self,
        query: str,
        top_k: int = 15,
        exclude_admin: bool = True,
        exclude_promoted: bool = True,
        include_super_admin: bool = False,
    ) -> List[Tuple[str, float]]:
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "exclude_admin": exclude_admin,
            "exclude_promoted": exclude_promoted,
            "include_super_admin": include_super_admin,
        })
        if self.exception is not None:
            raise self.exception
        return list(self.next_result)


_asi_module = ModuleType("modules.tools.discovery.action_semantic_index")
_asi_module._fake_singleton = _FakeSemanticIndex()
_asi_module.get_action_semantic_index = lambda: _asi_module._fake_singleton  # type: ignore[attr-defined]


# Runtime stubs the section's lazy imports resolve at render() time.
_RUNTIME_STUBS = {
    "config": _stub_config_module,
    "modules.tools.discovery.action_registry": _ar_module,
    "modules.tools.discovery.action_semantic_index": _asi_module,
}
_runtime_live_snapshot: dict = {}


def _install_runtime_stubs() -> None:
    """Install runtime stubs, remembering whatever was there to restore later."""
    global _runtime_live_snapshot
    _runtime_live_snapshot = {k: sys.modules.get(k) for k in _RUNTIME_STUBS}
    sys.modules.update(_RUNTIME_STUBS)


def _restore_runtime_stubs() -> None:
    _restore(_runtime_live_snapshot)


# ---------------------------------------------------------------------------
# Import-time block: install just enough to bind BaseSection / SectionContext /
# PlatformActionsSection from disk, then restore sys.modules. The section's
# top-level import is ``from modules.context.sections.base import ...``; base's
# is ``from modules.context.estimator import TokenEstimator`` — both resolve
# from the dotted stubs we install here. config / registry / index are only
# touched at render() time, so they stay out of the import entirely.
# ---------------------------------------------------------------------------
sys.modules["modules.context.estimator"] = _estimator_mod

_sections_pkg = ModuleType("modules.context.sections")
_sections_pkg.__path__ = [str(_SECTIONS)]
sys.modules["modules.context.sections"] = _sections_pkg

_base_spec = importlib.util.spec_from_file_location(
    "modules.context.sections.base", _SECTIONS / "base.py"
)
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["modules.context.sections.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)
BaseSection = _base_mod.BaseSection
SectionContext = _base_mod.SectionContext

_sec_spec = importlib.util.spec_from_file_location(
    "modules.context.sections.platform_actions", _SECTIONS / "platform_actions.py"
)
_sec_mod = importlib.util.module_from_spec(_sec_spec)
sys.modules["modules.context.sections.platform_actions"] = _sec_mod
_sec_spec.loader.exec_module(_sec_mod)
PlatformActionsSection = _sec_mod.PlatformActionsSection

# Collection-safe: undo every sys.modules mutation made during import.
_restore(_PRE_IMPORT_SNAPSHOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_action(
    name: str,
    *,
    category: str = "agents",
    description: str = "",
    admin_only: bool = False,
    promoted: bool = False,
    properties: dict | None = None,
    required: list[str] | None = None,
) -> ActionDefinition:
    return ActionDefinition(
        name=name,
        description=description or f"{name} description",
        category=category,
        parameters={
            "type": "object",
            "properties": properties or {},
            "required": required or [],
        },
        admin_only=admin_only,
        promoted=promoted,
    )


def _install_registry(actions: List[ActionDefinition]) -> ActionRegistry:
    """Wire a fresh ActionRegistry into the stubbed module."""
    reg = ActionRegistry()
    reg._initialized = True  # bypass lazy live registrar
    for a in actions:
        reg.register(a)
    _ar_module.get_action_registry = lambda: reg  # type: ignore[attr-defined]
    return reg


def _install_index_result(result: List[Tuple[str, float]] | None = None,
                          exception: Exception | None = None) -> _FakeSemanticIndex:
    fake = _asi_module._fake_singleton  # type: ignore[attr-defined]
    fake.calls.clear()
    fake.next_result = list(result or [])
    fake.exception = exception
    return fake


def _ctx(query: str | None = "__unset__") -> SectionContext:
    kwargs: dict = {}
    if query != "__unset__":
        kwargs["query"] = query
    return SectionContext(agent=MagicMock(), workspace_id="ws-1", kwargs=kwargs)


def _set_flag(enabled: bool, top_k: int = 3) -> None:
    cfg = _stub_config_module.config
    cfg.SEMANTIC_TOOL_ROUTING = enabled
    cfg.SEMANTIC_TOOL_ROUTING_TOP_K = top_k


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_state():
    """Install runtime stubs + reset stub state around each test."""
    _install_runtime_stubs()
    _set_flag(True, top_k=3)
    _install_index_result(result=[])
    try:
        yield
    finally:
        _install_index_result(result=[])
        _restore_runtime_stubs()


# ---------------------------------------------------------------------------
# AC #1 + #2: filtered path runs when flag enabled and query non-empty
# ---------------------------------------------------------------------------


def test_filtered_path_when_query_and_flag_enabled():
    actions = [
        _make_action("platform_list_agents", category="agents"),
        _make_action("platform_create_agent", category="agents"),
        _make_action("platform_list_missions", category="missions"),
        _make_action("platform_unrelated_thing", category="other"),
    ]
    _install_registry(actions)
    fake = _install_index_result([
        ("platform_list_agents", 0.91),
        ("platform_create_agent", 0.85),
        ("platform_list_missions", 0.71),
    ])
    _set_flag(True, top_k=3)

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="how do I list agents")))

    assert fake.calls, "rank_actions should be invoked when flag on + query present"
    call = fake.calls[0]
    assert call["query"] == "how do I list agents"
    assert call["top_k"] == 3
    assert call["exclude_admin"] is True
    assert call["exclude_promoted"] is True

    # Returned markdown contains the ranked names but NOT the unrelated one.
    assert "platform_list_agents" in out
    assert "platform_create_agent" in out
    assert "platform_list_missions" in out
    assert "platform_unrelated_thing" not in out
    assert "## Platform Actions" in out  # preamble retained

    # AC: result contains <= top_k action lines.
    action_lines = [line for line in out.splitlines() if line.startswith("- `platform_")]
    assert len(action_lines) <= 3


# ---------------------------------------------------------------------------
# AC #4: full dump when flag disabled — index NOT called
# ---------------------------------------------------------------------------


def test_full_dump_when_flag_disabled():
    actions = [
        _make_action("platform_list_agents", category="agents"),
        _make_action("platform_create_agent", category="agents"),
    ]
    _install_registry(actions)
    fake = _install_index_result([("platform_list_agents", 0.99)])
    _set_flag(False)

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="anything")))

    assert fake.calls == [], "filtered path must be skipped when flag is False"
    assert "platform_list_agents" in out
    assert "platform_create_agent" in out


# ---------------------------------------------------------------------------
# AC #3: filtered path raises → fall back to full dump
# ---------------------------------------------------------------------------


def test_fallback_when_index_raises():
    actions = [
        _make_action("platform_list_agents", category="agents"),
        _make_action("platform_create_agent", category="agents"),
    ]
    _install_registry(actions)
    _install_index_result(exception=RuntimeError("boom"))
    _set_flag(True, top_k=3)

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="kaboom")))

    # Fell back to the full dump — both actions must be present.
    assert "platform_list_agents" in out
    assert "platform_create_agent" in out
    assert "## Platform Actions" in out


# ---------------------------------------------------------------------------
# AC #5a: ctx.kwargs without 'query' → full dump (index not called)
# ---------------------------------------------------------------------------


def test_full_dump_when_no_query():
    actions = [_make_action("platform_list_agents", category="agents")]
    _install_registry(actions)
    fake = _install_index_result([("platform_list_agents", 0.5)])
    _set_flag(True, top_k=3)

    section = PlatformActionsSection()
    # query=__unset__ sentinel skips inserting the key entirely.
    out = _run(section.render(_ctx()))

    assert fake.calls == [], "index must not be touched when no query key"
    assert "platform_list_agents" in out


# ---------------------------------------------------------------------------
# AC #5b: empty query string → full dump
# ---------------------------------------------------------------------------


def test_full_dump_when_empty_query():
    actions = [_make_action("platform_list_agents", category="agents")]
    _install_registry(actions)
    fake = _install_index_result([("platform_list_agents", 0.5)])
    _set_flag(True, top_k=3)

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="")))

    assert fake.calls == [], "index must not be touched on empty query"
    assert "platform_list_agents" in out


def test_full_dump_when_whitespace_query():
    actions = [_make_action("platform_list_agents", category="agents")]
    _install_registry(actions)
    fake = _install_index_result([("platform_list_agents", 0.5)])
    _set_flag(True, top_k=3)

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="   \t  ")))

    assert fake.calls == [], "index must not be touched on whitespace-only query"
    assert "platform_list_agents" in out


# ---------------------------------------------------------------------------
# AC #6: section priority unchanged
# ---------------------------------------------------------------------------


def test_section_priority_unchanged():
    section = PlatformActionsSection()
    assert section.priority == 5
    assert section.name == "platform_actions"


# ---------------------------------------------------------------------------
# AC #8 (token-drop surrogate): filtered render is strictly shorter than full
#
# The integration AC (~7,300 → ~1,600 tokens) requires the live production
# index over the real catalog and is exercised by the US-005 eval suite. Here
# we use a fully mocked filtered render with a populated registry and verify
# the filtered output emits fewer tokens than the full dump for the same
# registry.
# ---------------------------------------------------------------------------


def test_filtered_render_emits_fewer_tokens_than_full():
    actions = [
        _make_action(f"platform_action_{i}", category="agents", description="x" * 80)
        for i in range(20)
    ]
    _install_registry(actions)
    _install_index_result([
        (f"platform_action_{i}", 1.0 - i * 0.01) for i in range(3)
    ])
    _set_flag(True, top_k=3)

    section = PlatformActionsSection()
    filtered = _run(section.render(_ctx(query="needle")))

    # Run again with flag off to get the full dump baseline.
    _set_flag(False)
    full = _run(section.render(_ctx(query="needle")))

    filtered_tokens = section.estimate_tokens(filtered)
    full_tokens = section.estimate_tokens(full)
    assert filtered_tokens < full_tokens, (
        f"filtered={filtered_tokens} not strictly less than full={full_tokens}"
    )


# ---------------------------------------------------------------------------
# AC #2: top_k passed through from config, exclude flags fixed to True
# ---------------------------------------------------------------------------


def test_top_k_threaded_from_config():
    actions = [_make_action(f"platform_a_{i}", category="agents") for i in range(8)]
    _install_registry(actions)
    fake = _install_index_result([
        (f"platform_a_{i}", 1.0 - i * 0.05) for i in range(7)
    ])
    _set_flag(True, top_k=7)

    section = PlatformActionsSection()
    _run(section.render(_ctx(query="match many")))

    assert fake.calls[0]["top_k"] == 7
    assert fake.calls[0]["exclude_admin"] is True
    assert fake.calls[0]["exclude_promoted"] is True


# ---------------------------------------------------------------------------
# Empty ranking from index → fall back to full dump (defensive path)
# ---------------------------------------------------------------------------


def test_empty_ranking_falls_back_to_full_dump():
    actions = [
        _make_action("platform_list_agents", category="agents"),
        _make_action("platform_create_agent", category="agents"),
    ]
    _install_registry(actions)
    _install_index_result([])  # empty result
    _set_flag(True, top_k=3)

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="orphan query")))

    # Both actions must appear since we fell back to the full dump.
    assert "platform_list_agents" in out
    assert "platform_create_agent" in out
