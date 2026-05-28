"""
PRD-141 US-013: ActionRegistry category/tag lookups.
=====================================================

``get_by_category(category)`` and ``get_by_tags(tags)`` let the tool pipeline
filter actions without a hardcoded dict. Both call ``_ensure_initialized()``
first. ``get_by_tags`` uses OR semantics — an action matches if it carries any
of the requested tags.

The ``modules.tools`` package ``__init__`` eagerly imports the executor chain
(DB-backed), so we load ``action_registry.py`` as an isolated leaf module — it
only uses stdlib at import time. Tests register actions directly and flip the
``_initialized`` flag, so ``_ensure_initialized`` never triggers the (DB-backed)
platform-action load.
"""
import importlib.util
import sys
from pathlib import Path

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))


def _load_action_registry_module():
    path = _orchestrator_root / "modules" / "tools" / "discovery" / "action_registry.py"
    spec = importlib.util.spec_from_file_location("_us013_action_registry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_action_registry_module()
ActionRegistry = _mod.ActionRegistry
ActionDefinition = _mod.ActionDefinition


def _action(name, category, tags=None):
    return ActionDefinition(
        name=name,
        description=f"{name} description",
        category=category,
        parameters={"type": "object", "properties": {}},
        tags=tags or [],
    )


def _registry_with(*actions):
    """A registry pre-loaded with *actions*, with init short-circuited."""
    reg = ActionRegistry()
    for a in actions:
        reg.register(a)
    reg._initialized = True  # skip the DB-backed platform-action load
    return reg


def test_action_registry_get_by_category():
    """get_by_category('harness') returns only harness-category actions."""
    reg = _registry_with(
        _action("harness_status", "harness", tags=["monitoring"]),
        _action("harness_restart", "harness"),
        _action("list_agents", "agents"),
    )

    result = reg.get_by_category("harness")

    names = {a.name for a in result}
    assert names == {"harness_status", "harness_restart"}
    assert all(a.category == "harness" for a in result)


def test_action_registry_get_by_tags():
    """get_by_tags(['monitoring']) returns only actions tagged 'monitoring'."""
    reg = _registry_with(
        _action("harness_status", "harness", tags=["monitoring", "health"]),
        _action("metrics_read", "analytics", tags=["monitoring"]),
        _action("list_agents", "agents", tags=["agents"]),
    )

    result = reg.get_by_tags(["monitoring"])

    names = {a.name for a in result}
    assert names == {"harness_status", "metrics_read"}


def test_get_by_tags_is_or_semantics():
    """Multiple requested tags union their matches (OR, not AND)."""
    reg = _registry_with(
        _action("a", "x", tags=["monitoring"]),
        _action("b", "x", tags=["billing"]),
        _action("c", "x", tags=["unrelated"]),
    )

    result = reg.get_by_tags(["monitoring", "billing"])

    assert {a.name for a in result} == {"a", "b"}


def test_lookups_return_empty_when_no_match():
    reg = _registry_with(_action("a", "x", tags=["monitoring"]))
    assert reg.get_by_category("nope") == []
    assert reg.get_by_tags(["nope"]) == []


def test_lookups_trigger_initialization():
    """A fresh (uninitialized) registry initializes lazily on lookup."""
    reg = ActionRegistry()
    calls = {"n": 0}

    def fake_init():
        calls["n"] += 1
        reg._initialized = True

    reg._ensure_initialized = fake_init

    reg.get_by_category("harness")
    reg.get_by_tags(["monitoring"])

    assert calls["n"] == 2  # both lookups went through _ensure_initialized
