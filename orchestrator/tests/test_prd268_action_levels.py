"""PRD-268 US-C02a: every platform action states whether it reads or writes.

The check-in lane (US-C02b) runs only what is labelled ``read``, so the label has
to be stated, never defaulted:

- ``ActionDefinition.permission_level`` has no default; a definition built without
  one is refused with a TypeError naming the action, which surfaces at registration
  (the registry loads lazily, in ``_ensure_initialized``);
- every registered action is ``read``, ``write`` or ``destructive``;
- the field tools state theirs, and ``platform_field_inject`` (its handler calls
  ``field.inject``) is a write the hierarchy gate accounts for;
- the hierarchy gate no longer reads a missing level as ``read``: it stops, naming
  the action.
"""
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
LEVELS = ("read", "write", "destructive")
PROBE = "platform_cafe_probe"
PARAMS = {"type": "object", "properties": {}}


def _gate():
    spec = importlib.util.spec_from_file_location("check_hierarchy_gate", ORCH / "scripts" / "check_hierarchy_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── the dataclass ───────────────────────────────────────────────────────────

def test_the_level_has_no_default():
    from modules.tools.discovery.action_registry import ActionDefinition

    (level,) = [f for f in dataclasses.fields(ActionDefinition) if f.name == "permission_level"]
    assert level.default is dataclasses.MISSING and level.default_factory is dataclasses.MISSING


@pytest.mark.parametrize("args,kwargs", [
    ((), {"name": PROBE, "description": "d", "category": "t", "parameters": PARAMS}),
    ((PROBE, "d", "t", PARAMS), {}),
    ((PROBE,), {"description": "d", "category": "t", "parameters": PARAMS, "promoted": True}),
], ids=["keywords", "positional", "mixed"])
def test_a_definition_that_states_no_level_is_refused_naming_the_action(args, kwargs):
    from modules.tools.discovery.action_registry import ActionDefinition

    with pytest.raises(TypeError, match=f"'{PROBE}' states no permission_level"):
        ActionDefinition(*args, **kwargs)


def test_a_definition_that_states_its_level_builds_copies_and_replaces():
    from modules.tools.discovery.action_registry import ActionDefinition

    positional = ActionDefinition(PROBE, "d", "t", PARAMS, "write")
    stated = ActionDefinition(name=PROBE, description="d", category="t", parameters=PARAMS, permission_level="read")

    assert positional.permission_level == "write" and stated.permission_level == "read"
    assert copy.copy(stated) == stated and copy.deepcopy(stated) == stated
    assert dataclasses.replace(stated, description="e").permission_level == "read"


def test_the_refusal_surfaces_at_registration(monkeypatch):
    """Registration is lazy: the first lookup runs the registrars, and one that
    builds a definition with no level stops the load, naming the action."""
    from modules.tools.discovery import platform_actions
    from modules.tools.discovery.action_registry import ActionDefinition, ActionRegistry

    def register_all_actions(registry):
        registry.register(ActionDefinition(name=PROBE, description="d", category="t", parameters=PARAMS))

    monkeypatch.setattr(platform_actions, "register_all_actions", register_all_actions)
    registry = ActionRegistry()

    with pytest.raises(TypeError, match=PROBE):
        registry.get_all()
    assert registry._initialized is False


# ── the registry ────────────────────────────────────────────────────────────

def test_every_registered_action_reads_writes_or_destroys():
    from modules.tools.discovery import get_action_registry

    actions = get_action_registry().get_all()
    unlabelled = sorted(f"{a.name} [{a.permission_level!r}]" for a in actions if a.permission_level not in LEVELS)

    assert actions and unlabelled == []


@pytest.mark.parametrize("name,level", [
    ("platform_field_query", "read"),
    ("platform_field_inject", "write"),
    ("platform_field_stability", "read"),
])
def test_the_field_tools_state_their_levels(name, level):
    from modules.tools.discovery import get_action_registry

    assert get_action_registry().get(name).permission_level == level


def test_platform_field_inject_is_a_write_the_hierarchy_gate_accounts_for():
    gate = _gate()

    assert gate.collect_registrations()["platform_field_inject"].permission_level == "write"
    assert "platform_field_inject" in gate.ALLOW_LIST
    assert gate.main() == 0


def test_platform_field_inject_is_judged_a_write_wherever_the_level_is_read():
    """The relabel is meant: the policy gate classes it internal_write (Strict asks
    first), and a turn's account counts it as a change, not a look."""
    from modules.policy.policy_document import RISK_INTERNAL_WRITE, classify_action
    from modules.tools.discovery import get_action_registry
    from modules.tools.execution.turn_account import is_read

    level = get_action_registry().get("platform_field_inject").permission_level

    assert classify_action("platform_field_inject", permission_level=level) == RISK_INTERNAL_WRITE
    assert is_read("platform_field_inject") is False
    assert is_read("platform_field_query") is True


# ── the hierarchy gate ──────────────────────────────────────────────────────

def _audit(tmp_path, *lines):
    gate = _gate()
    body = "".join(f"        {line}\n" for line in (f'name="{PROBE}",', *lines))
    (tmp_path / "actions_cafe.py").write_text(
        "from .action_registry import ActionDefinition\n\n\ndef register(registry):\n"
        f"    registry.register(ActionDefinition(\n{body}    ))\n")
    (tmp_path / "platform_executor.py").write_text("_HIERARCHY_TARGETS = {\n}\n")
    gate.ACTIONS_DIR, gate.EXECUTOR = tmp_path, tmp_path / "platform_executor.py"
    gate.ALLOW_LIST = set()
    return gate


def test_the_gate_stops_on_a_registration_with_no_level_naming_it(tmp_path, capsys):
    gate = _audit(tmp_path, 'description="reads the menu",')

    with pytest.raises(SystemExit) as stopped:
        gate.main()

    assert stopped.value.code == 2
    assert f"{PROBE} states no permission_level" in capsys.readouterr().err


def test_the_gate_passes_a_stated_read(tmp_path):
    assert _audit(tmp_path, 'permission_level="read",').main() == 0


def test_the_gate_keeps_no_read_default():
    assert not hasattr(_gate(), "DEFAULT_PERMISSION_LEVEL")

