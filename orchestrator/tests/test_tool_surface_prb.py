"""Tool-surface PR-B: platform_find_tools + relevance floor + closed-pins.

(docs/reviews/TOOL-SURFACE-DEEP-REVIEW-2026-07-23.md §5/§6, stage PR-B.)

Everything lands behind default-off dials — these tests pin BOTH postures:
the defaults must behave exactly like today, and the new modes must do what
the dossier says when switched on.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

_ORCH = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. platform_find_tools — registration + executor wiring
# ---------------------------------------------------------------------------


def _fresh_registry():
    from modules.tools.discovery.action_registry import ActionRegistry

    reg = ActionRegistry()
    reg._initialized = True
    return reg


def test_find_tools_registers_promoted_read_action() -> None:
    from modules.tools.discovery.actions_capabilities import (
        register_capabilities_actions,
    )

    reg = _fresh_registry()
    register_capabilities_actions(reg)
    action = reg.get("platform_find_tools")
    assert action is not None
    assert action.promoted is True, "discovery must always be first-class visible"
    assert action.permission_level == "read"
    assert getattr(action, "super_admin_only", False) is False
    assert "query" in (action.parameters.get("required") or [])


def test_find_tools_wired_into_registrar_and_executor() -> None:
    registrar = (_ORCH / "modules" / "tools" / "discovery" / "platform_actions.py").read_text()
    assert "register_capabilities_actions(registry)" in registrar
    executor = (_ORCH / "modules" / "tools" / "discovery" / "platform_executor.py").read_text()
    assert re.search(r'"platform_find_tools":\s*find_tools', executor), (
        "platform_find_tools has no executor handler — registered but uncallable"
    )


# ---------------------------------------------------------------------------
# 2. find_tools handler
# ---------------------------------------------------------------------------


def _action(name: str, desc: str, tags: List[str], su: bool = False, admin: bool = False):
    return SimpleNamespace(
        name=name,
        description=desc,
        category="test",
        tags=tags,
        permission_level="read",
        parameters={
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "what"}},
            "required": ["topic"],
        },
        admin_only=admin,
        super_admin_only=su,
        promoted=False,
    )


_CATALOG = [
    _action("platform_publish_blog_post", "Write and publish a blog post", ["blog", "publish"]),
    _action("platform_codegraph_search", "Search a codebase by meaning", ["code", "search"]),
    _action("platform_get_logs", "Read service logs", ["logs"], su=True),
]


class _FakeIndex:
    def __init__(self, ranked: Optional[List[Tuple[str, float]]] = None, raise_it: bool = False):
        self.ranked = ranked or []
        self.raise_it = raise_it

    async def rank_actions(self, **kwargs: Any) -> List[Tuple[str, float]]:
        if self.raise_it:
            raise RuntimeError("embed upstream down")
        return self.ranked


class _FakeRegistry:
    def get_all(self):
        return list(_CATALOG)


@pytest.mark.asyncio
async def test_find_tools_returns_schema_rich_matches() -> None:
    from modules.tools.discovery import handlers_capabilities as mod

    fake = _FakeIndex(ranked=[("platform_publish_blog_post", 0.81)])
    with patch(
        "modules.tools.discovery.action_semantic_index.get_action_semantic_index",
        return_value=fake,
    ), patch(
        "modules.tools.discovery.action_registry.get_action_registry",
        return_value=_FakeRegistry(),
    ):
        out = await mod.find_tools(None, None, {"query": "publish a blog"})

    assert out["success"] is True
    assert out["ranker"] == "semantic"
    assert out["matches"][0]["action"] == "platform_publish_blog_post"
    assert out["matches"][0]["params"]["required"] == ["topic"]
    assert "platform_execute" in out["matches"][0]["call_with"]


@pytest.mark.asyncio
async def test_find_tools_never_advertises_su_actions() -> None:
    from modules.tools.discovery import handlers_capabilities as mod

    # Ranker leaks an su name (defense in depth): the handler's eligible set
    # must drop it anyway.
    fake = _FakeIndex(ranked=[("platform_get_logs", 0.9), ("platform_codegraph_search", 0.5)])
    with patch(
        "modules.tools.discovery.action_semantic_index.get_action_semantic_index",
        return_value=fake,
    ), patch(
        "modules.tools.discovery.action_registry.get_action_registry",
        return_value=_FakeRegistry(),
    ):
        out = await mod.find_tools(None, None, {"query": "logs"})

    names = [m["action"] for m in out["matches"]]
    assert "platform_get_logs" not in names


@pytest.mark.asyncio
async def test_find_tools_keyword_fallback_when_ranker_fails() -> None:
    from modules.tools.discovery import handlers_capabilities as mod

    with patch(
        "modules.tools.discovery.action_semantic_index.get_action_semantic_index",
        return_value=_FakeIndex(raise_it=True),
    ), patch(
        "modules.tools.discovery.action_registry.get_action_registry",
        return_value=_FakeRegistry(),
    ):
        out = await mod.find_tools(None, None, {"query": "publish blog post"})

    assert out["success"] is True
    assert out["ranker"] == "keyword"
    assert out["matches"], "keyword fallback found nothing for an obvious match"
    assert out["matches"][0]["action"] == "platform_publish_blog_post"


@pytest.mark.asyncio
async def test_find_tools_requires_query() -> None:
    from modules.tools.discovery import handlers_capabilities as mod

    out = await mod.find_tools(None, None, {})
    assert out["success"] is False


# ---------------------------------------------------------------------------
# 3. Relevance floor (pure)
# ---------------------------------------------------------------------------


def test_floor_off_is_identity() -> None:
    from modules.tools.discovery.action_semantic_index import _apply_relevance_floor

    scored = [("a", 0.9), ("b", 0.1), ("c", -0.2)]
    assert _apply_relevance_floor(scored, 0, 0) == scored


def test_absolute_floor_drops_low_scores() -> None:
    from modules.tools.discovery.action_semantic_index import _apply_relevance_floor

    scored = [("a", 0.9), ("b", 0.4), ("c", 0.2)]
    assert _apply_relevance_floor(scored, 0.35, 0) == [("a", 0.9), ("b", 0.4)]


def test_ratio_floor_scales_with_best() -> None:
    from modules.tools.discovery.action_semantic_index import _apply_relevance_floor

    scored = [("a", 0.8), ("b", 0.5), ("c", 0.3)]
    # cutoff = 0.8 * 0.6 = 0.48
    assert _apply_relevance_floor(scored, 0, 0.6) == [("a", 0.8), ("b", 0.5)]


def test_ratio_floor_ignores_negative_best() -> None:
    from modules.tools.discovery.action_semantic_index import _apply_relevance_floor

    scored = [("a", -0.1), ("b", -0.5)]
    # A negative best * ratio would be a meaningless cutoff — ratio must not bite.
    assert _apply_relevance_floor(scored, 0, 0.6) == scored


def test_floor_can_empty_the_list() -> None:
    from modules.tools.discovery.action_semantic_index import _apply_relevance_floor

    scored = [("a", 0.11), ("b", 0.08)]
    assert _apply_relevance_floor(scored, 0.3, 0) == []


# ---------------------------------------------------------------------------
# 4. Closed-pins fallback narrowing
# ---------------------------------------------------------------------------


@pytest.fixture()
def _cfg(monkeypatch):
    import config as config_module

    def set_(key: str, value: Any) -> None:
        monkeypatch.setattr(config_module.config, key, value, raising=False)

    return set_


@pytest.mark.asyncio
async def test_default_open_full_on_rank_failure(_cfg) -> None:
    import modules.tools.tool_router as tr

    _cfg("SEMANTIC_TOOL_ROUTING", True)
    _cfg("TOOL_FALLBACK_MODE", "open-full")

    async def _none(**kwargs: Any):
        return None

    with patch.object(tr, "_rank_actions_for_dispatcher_async", _none):
        allowed, reason, from_pins = await tr._narrow_dispatcher_actions_async(
            "hello", is_admin=False, is_super_admin=False
        )
    assert allowed is None and from_pins is False


@pytest.mark.asyncio
async def test_closed_pins_on_rank_failure(_cfg) -> None:
    import modules.tools.tool_router as tr

    _cfg("SEMANTIC_TOOL_ROUTING", True)
    _cfg("TOOL_FALLBACK_MODE", "closed-pins")
    _cfg("TOOL_FALLBACK_PINS", "platform_find_tools, platform_store_memory")

    async def _none(**kwargs: Any):
        return None

    with patch.object(tr, "_rank_actions_for_dispatcher_async", _none):
        allowed, reason, from_pins = await tr._narrow_dispatcher_actions_async(
            "hello", is_admin=False, is_super_admin=False
        )
    assert allowed == ["platform_find_tools", "platform_store_memory"]
    assert from_pins is True
    assert "closed-pins" in (reason or "")


@pytest.mark.asyncio
async def test_closed_pins_on_missing_query(_cfg) -> None:
    import modules.tools.tool_router as tr

    _cfg("SEMANTIC_TOOL_ROUTING", True)
    _cfg("TOOL_FALLBACK_MODE", "closed-pins")
    _cfg("TOOL_FALLBACK_PINS", "platform_find_tools")

    allowed, _reason, from_pins = await tr._narrow_dispatcher_actions_async(
        None, is_admin=False, is_super_admin=False
    )
    assert allowed == ["platform_find_tools"] and from_pins is True


@pytest.mark.asyncio
async def test_flag_off_is_always_open_full(_cfg) -> None:
    """Operator turned routing OFF — that's an explicit wide-surface choice;
    closed-pins must not override it."""
    import modules.tools.tool_router as tr

    _cfg("SEMANTIC_TOOL_ROUTING", False)
    _cfg("TOOL_FALLBACK_MODE", "closed-pins")

    allowed, reason, from_pins = await tr._narrow_dispatcher_actions_async(
        "hello", is_admin=False, is_super_admin=False
    )
    assert allowed is None and from_pins is False


# ---------------------------------------------------------------------------
# 5. Dispatcher enum admits promoted pins ONLY under the flag
# ---------------------------------------------------------------------------


def _registry_with_promoted():
    from modules.tools.discovery.action_registry import ActionDefinition

    reg = _fresh_registry()
    reg.register(ActionDefinition(
        name="plain_action", description="plain", category="t",
        parameters={"type": "object", "properties": {}},
    ))
    reg.register(ActionDefinition(
        name="promoted_action", description="promoted", category="t",
        parameters={"type": "object", "properties": {}}, promoted=True,
    ))
    reg.register(ActionDefinition(
        name="su_promoted_action", description="su", category="t",
        parameters={"type": "object", "properties": {}}, promoted=True,
        super_admin_only=True,
    ))
    return reg


def _enum(schema: Dict[str, Any]) -> List[str]:
    return schema["function"]["parameters"]["properties"]["action"].get("enum", [])


def test_promoted_pin_dropped_without_flag() -> None:
    reg = _registry_with_promoted()
    schema = reg.to_dispatcher_schema(
        exclude_admin=True,
        allowed_names=["promoted_action", "plain_action"],
    )
    assert _enum(schema) == ["plain_action"]


def test_promoted_pin_admitted_with_flag() -> None:
    reg = _registry_with_promoted()
    schema = reg.to_dispatcher_schema(
        exclude_admin=True,
        allowed_names=["promoted_action", "plain_action"],
        allow_promoted_in_allowlist=True,
    )
    assert _enum(schema) == ["plain_action", "promoted_action"]


def test_su_promoted_pin_never_admitted_for_operator() -> None:
    """Fail-closed survives the new flag: su stays out even when pinned."""
    reg = _registry_with_promoted()
    schema = reg.to_dispatcher_schema(
        exclude_admin=True,
        allowed_names=["su_promoted_action", "plain_action"],
        allow_promoted_in_allowlist=True,
    )
    assert _enum(schema) == ["plain_action"]


# ---------------------------------------------------------------------------
# 6. Config dial defaults (PR-B must be inert out of the box)
# ---------------------------------------------------------------------------


def test_prb_dials_default_inert() -> None:
    import config as config_module

    cfg = config_module.config
    assert float(getattr(cfg, "SEMANTIC_TOOL_ROUTING_FLOOR", 0)) == 0.0
    assert float(getattr(cfg, "SEMANTIC_TOOL_ROUTING_FLOOR_RATIO", 0)) == 0.0
    assert str(getattr(cfg, "TOOL_FALLBACK_MODE", "open-full")) == "open-full"
    assert "platform_find_tools" in str(getattr(cfg, "TOOL_FALLBACK_PINS", ""))
