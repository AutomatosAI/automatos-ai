"""PRD-205 S8 -- chat-router declaration-order locks.

FastAPI matches routes in declaration order, so a literal path declared
below its param sibling is dead: ``PATCH /api/chat/vote`` resolved as
``chat_id="vote"`` and ``GET /api/chat/agents`` as ``chat_id="agents"``
(the exact PRD-220 ``/search`` failure mode). S8 moves both above
``/{chat_id}``; these tests lock all three literal routes -- ``/vote``,
``/agents`` and ``/search`` -- so a future reshuffle cannot silently
re-kill them. Same paths, same handlers, no manifest delta.

Pure unit tests over ``router.routes`` (the proven PRD-220 pattern from
``test_p3_widget_chat_persistence.py``) -- no DB, no server.
"""
from __future__ import annotations

import pytest


def _api_chat():
    try:
        import api.chat as chat_module
    except Exception as e:  # env without the heavy router deps
        pytest.skip(f"api.chat not importable in this env: {e}")
    return chat_module


def _paths_for_method(router, method: str):
    return [
        r.path
        for r in router.routes
        if method in (getattr(r, "methods", None) or set())
    ]


def _assert_literal_before_param(router, method: str, literal: str, param: str):
    paths = _paths_for_method(router, method)
    assert literal in paths, f"{method} {literal} missing from router: {paths}"
    assert param in paths, f"{method} {param} missing from router: {paths}"
    assert paths.index(literal) < paths.index(param), (
        f"{method} {literal} must be declared before {method} {param} or it "
        f"resolves as chat_id='{literal.rsplit('/', 1)[-1]}'. "
        f"Current {method} order: {paths}"
    )


def test_vote_route_registered_before_chat_id_param_route():
    chat_module = _api_chat()
    _assert_literal_before_param(
        chat_module.router, "PATCH", "/api/chat/vote", "/api/chat/{chat_id}"
    )


def test_agents_route_registered_before_chat_id_param_route():
    chat_module = _api_chat()
    _assert_literal_before_param(
        chat_module.router, "GET", "/api/chat/agents", "/api/chat/{chat_id}"
    )


def test_search_route_stays_before_chat_id_param_route():
    """The PRD-220 fix stays locked alongside the two S8 moves."""
    chat_module = _api_chat()
    _assert_literal_before_param(
        chat_module.router, "GET", "/api/chat/search", "/api/chat/{chat_id}"
    )


def test_no_other_literal_chat_route_is_shadowed():
    """Belt-and-braces: EVERY literal single-segment route under /api/chat
    must precede the /{chat_id} param route for its method -- catches the
    next /vote-style regression before it ships, whatever its name."""
    chat_module = _api_chat()
    for method in ("GET", "PATCH", "DELETE", "POST", "PUT"):
        paths = _paths_for_method(chat_module.router, method)
        if "/api/chat/{chat_id}" not in paths:
            continue
        param_idx = paths.index("/api/chat/{chat_id}")
        literals = [
            p for p in paths
            if p.startswith("/api/chat/")
            and "{" not in p
            and p.count("/") == 3  # single segment under the /api/chat prefix
        ]
        for lit in literals:
            assert paths.index(lit) < param_idx, (
                f"{method} {lit} is declared after {method} /api/chat/{{chat_id}} "
                f"and is dead-shadowed. Order: {paths}"
            )
