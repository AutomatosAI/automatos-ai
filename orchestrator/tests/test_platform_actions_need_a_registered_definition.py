"""A platform action runs only when it has a registered ActionDefinition.

Every permission gate in ``PlatformActionExecutor.execute`` reads the action's
definition, so a handler without one is refused as an unknown action. The
legacy ``platform_*_recipe*`` aliases, which had handlers but no definitions,
are removed; their ``platform_*_playbook*`` twins remain.
"""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from modules.tools.discovery import get_action_registry
from modules.tools.discovery.platform_executor import PlatformActionExecutor

LEGACY_ALIASES = (
    "platform_list_recipes",
    "platform_get_recipe",
    "platform_create_recipe",
    "platform_update_recipe",
    "platform_add_recipe_step",
    "platform_update_recipe_step",
    "platform_delete_recipe_step",
    "platform_execute_recipe",
    "platform_get_recipe_execution",
    "platform_delete_recipe",
)
OWNER = {"workspace_role": "owner"}


def _executor():
    return PlatformActionExecutor(MagicMock(), uuid.uuid4())


def test_every_handler_has_a_registered_definition():
    registry = get_action_registry()
    assert sorted(name for name in _executor()._handlers if registry.get(name) is None) == []


@pytest.mark.parametrize("alias", LEGACY_ALIASES)
def test_a_legacy_recipe_alias_is_an_unknown_action(alias):
    reply = asyncio.run(_executor().execute(alias, {"playbook_id": 81}, OWNER))
    assert reply == {"success": False, "error": f"Unknown platform action: {alias}"}


def test_a_handler_without_a_definition_never_runs():
    executor = _executor()
    handler = AsyncMock(return_value={"success": True})
    executor._handlers["platform_delete_recipe"] = handler
    reply = asyncio.run(executor.execute("platform_delete_recipe", {"playbook_id": 81}, OWNER))
    assert reply == {"success": False, "error": "Unknown platform action: platform_delete_recipe"}
    handler.assert_not_called()
