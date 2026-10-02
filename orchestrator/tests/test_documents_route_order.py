"""#827 — ``GET /api/documents/reprocess-status`` reaches its handler.

FastAPI matches routes in registration order. ``GET /{document_id}`` sat long
before ``GET /reprocess-status`` in ``api/documents.py``, so the static path was
parsed as a document id and refused with 422 (``int_parsing``). These ask the
router itself which handler a request reaches, so a later re-ordering or a new
one-segment static route cannot bring the shadowing back unnoticed.
"""
from __future__ import annotations

from typing import Optional

from starlette.routing import Match

from api.documents import router


def _first_match(method: str, path: str) -> Optional[str]:
    scope = {"type": "http", "method": method, "path": path, "root_path": ""}
    for route in router.routes:
        match, _ = route.matches(scope)
        if match is Match.FULL:
            return route.endpoint.__name__
    return None


def test_reprocess_status_reaches_its_own_handler():
    assert _first_match("GET", "/api/documents/reprocess-status") == "get_reprocess_status"


def test_a_document_id_still_reaches_get_document():
    assert _first_match("GET", "/api/documents/42") == "get_document"


def test_every_static_one_segment_get_reaches_its_own_handler():
    """No one-segment static GET on this router may be swallowed by the
    parameterised document route, wherever it is registered."""
    statics = [r for r in router.routes
               if "GET" in getattr(r, "methods", set()) and "{" not in r.path
               and r.path.count("/") == 3]  # /api/documents/<one segment>
    assert statics, "expected static one-segment GET routes on the documents router"
    for route in statics:
        assert _first_match("GET", route.path) == route.endpoint.__name__, route.path
