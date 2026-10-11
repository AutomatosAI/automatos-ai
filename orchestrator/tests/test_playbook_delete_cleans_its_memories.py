"""Deleting a playbook deletes the memories scoped to it.

The cleanup called ``memory_service.delete_memory(mem_id)`` without the
required ``workspace_id``; the TypeError was caught and logged at info level,
so a deleted playbook's memories were never removed. They live under the
playbook's own namespace, so they are deleted there in one scoped call.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import workflow_recipes  # noqa: E402

_WS = uuid.uuid4()


def _delete(memories):
    recipe = SimpleNamespace(id=81, template_id="weekly-numbers", is_system=False)
    service = MagicMock()
    service.get_all_memories_scoped = AsyncMock(return_value=memories)
    service.delete_memories_scoped = AsyncMock(return_value=True)
    service.delete_memory = AsyncMock(return_value=True)
    db = MagicMock()
    with patch.object(workflow_recipes, "playbook_at", return_value=recipe), \
         patch.object(workflow_recipes, "_cleanup_trigger_subscriptions"), \
         patch.object(workflow_recipes.config, "RECIPE_SCHEDULER_ENABLED", False), \
         patch("modules.memory.unified_memory_service.get_unified_memory_service", return_value=service):
        result = asyncio.run(workflow_recipes.delete_workflow_recipe(
            "81", ctx=SimpleNamespace(workspace_id=_WS), db=db,
        ))
    return result, service, db, recipe


def test_a_deleted_playbooks_memories_are_deleted_under_its_scope():
    result, service, db, recipe = _delete([{"id": "m1"}, {"id": "m2"}, {"memory": "no id"}])

    assert result["message"] == "Recipe deleted successfully"
    service.delete_memories_scoped.assert_awaited_once_with(
        ["m1", "m2"], f"mem:{_WS}:recipe:weekly-numbers", str(_WS),
    )
    service.delete_memory.assert_not_awaited()
    db.delete.assert_called_once_with(recipe)


def test_a_playbook_with_no_memories_skips_the_delete():
    _, service, db, recipe = _delete([])

    service.delete_memories_scoped.assert_not_awaited()
    db.delete.assert_called_once_with(recipe)
