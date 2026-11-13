"""PRD-274 Wave 1 (§5.2): a product's server key starts a Playbook and writes a Deliverable.

``POST /api/workflow-recipes/{id}/execute`` takes a member with ``playbooks:execute`` or a server
key listing ``playbooks:execute``; ``POST /api/documents/generate`` (a Deliverable from data and a
template) a member with ``documents:create`` or a key listing ``documents:write``. A run a key
starts says so (``triggered_by: api_key``, the key in its metadata), and its details are an
outside product's input, never read as the owner's words (``services.playbook_given``).
"""
from __future__ import annotations

from types import SimpleNamespace

from api import playbook_run_start as run_start
from core.auth import server_key_gate as gate
from core.auth.scopes import DOCUMENTS_WRITE, PLAYBOOKS_EXECUTE

KEY = SimpleNamespace(workspace_id="ws", auth_type="sdk_key", api_key_id="k1",
                      user=SimpleNamespace(id="sdk:k1", email=None, clerk_user_id=None))
PERSON = SimpleNamespace(workspace_id="ws", auth_type="clerk",
                         user=SimpleNamespace(id="user_1", email="priya@example.com", clerk_user_id="user_1"))


def test_a_run_a_key_starts_says_so_and_names_the_key():
    assert run_start._starter(KEY) == "api_key" and run_start._key_of(KEY) == {"api_key": "sdk:k1"}
    assert run_start._starter(PERSON) == "priya@example.com" and run_start._key_of(PERSON) == {}


def test_a_key_started_runs_details_are_never_the_owners_words():
    from services.playbook_given import started_by_a_trigger

    by_key = SimpleNamespace(triggered_by="api_key", retry_of=None, workspace_id="ws")
    by_person = SimpleNamespace(triggered_by="priya@example.com", retry_of=None, workspace_id="ws")
    assert started_by_a_trigger(None, by_key, {"customer": "Acme"}) is True
    assert started_by_a_trigger(None, by_person, {"customer": "Acme"}) is False


def test_the_routes_carry_the_gate_and_leave_slowapi():
    from fastapi.routing import APIRoute, iter_route_contexts

    from api.api_keys import VALID_PERMISSIONS
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from main import app, limiter

    routes = {(m, r.path): r for r in iter_route_contexts(app.routes)
              if isinstance(r.original_route, APIRoute) for m in r.methods}
    for key, permission, scope in ((("POST", "/api/workflow-recipes/{recipe_id}/execute"), "playbooks:execute",
                                    PLAYBOOKS_EXECUTE),
                                   (("POST", "/api/documents/generate"), "documents:create", DOCUMENTS_WRITE)):
        route = routes[key]
        calls = [d.call for d in route.dependant.dependencies]
        marked = [(getattr(c, PERMISSION_MARKER_ATTR, None), getattr(c, gate.KEY_SCOPE_MARKER_ATTR))
                  for c in calls if getattr(c, gate.KEY_SCOPE_MARKER_ATTR, None)]
        assert marked == [(permission, scope)], key
        assert f"{route.endpoint.__module__}.{route.endpoint.__name__}" in limiter._exempt_routes
    assert {PLAYBOOKS_EXECUTE, DOCUMENTS_WRITE} <= set(VALID_PERMISSIONS)
