"""F105 (4 Nov): the permission gates read the database off the event loop.

``require_workspace_permission``'s dependency guards almost every workspace route, and
``require_workspace_admin``, ``require_workspace_answerer`` and ``require_socials_enabled``
guard the rest of their families. Each was ``async def`` over a synchronous session and
awaited nothing, so every gated request read the caller's role on the event loop, and a
query waiting for a pooled connection stopped every request on that worker. They are plain
``def`` now, which FastAPI runs in its threadpool. What each lets through is unchanged
(their own tests pin it).
"""
from __future__ import annotations

import inspect

import pytest

from core.auth.workspace_admin import require_workspace_admin, require_workspace_answerer
from core.auth.workspace_permission import require_workspace_permission
from modules.socials.settings import require_socials_enabled

GATES = {
    "require_workspace_permission": require_workspace_permission("agents:create"),
    "require_workspace_admin": require_workspace_admin,
    "require_workspace_answerer": require_workspace_answerer,
    "require_socials_enabled": require_socials_enabled,
}


@pytest.mark.parametrize("name", sorted(GATES))
def test_the_gate_is_a_plain_def(name):
    assert not inspect.iscoroutinefunction(GATES[name]), f"{name} would read the database on the event loop"
