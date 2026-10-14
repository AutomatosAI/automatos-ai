"""PRD-259 S-001 — GET /api/skins/schema serves the skin format's rule set.

The Skin Kit syncs its copy of the schema from this route. Authenticated
(get_request_context_hybrid); the payload is ``schema_document()``.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from uuid import uuid4

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import config  # noqa: E402,F401


def _client(authenticated: bool = True):
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    from api.marketplace_skins import router
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid

    app = FastAPI()
    app.include_router(router)

    def _ctx():
        if not authenticated:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return RequestContext(
            workspace_id=uuid4(),
            user=UserContext(id="test-user", email="test@example.com", system_role="user"),
            auth_type="clerk",
        )

    app.dependency_overrides[get_request_context_hybrid] = _ctx
    return TestClient(app)


def test_schema_route_returns_the_schema_document():
    from services.skins.schema import ALLOWED_TOKENS, SKIN_SCHEMA_ID, contrast_pairs_document, schema_document

    response = _client().get("/api/skins/schema")
    assert response.status_code == 200
    body = response.json()
    assert body == schema_document()
    assert body["schema"] == SKIN_SCHEMA_ID == "automatos-skin/v1"
    assert body["allowedTokens"] == list(ALLOWED_TOKENS)
    assert body["contrastPairs"] == contrast_pairs_document()
    assert body["derivedAliases"] == ["paper", "bg-1", "bg-2", "fg", "fg-2", "fg-3"]
    assert body["shippedFonts"] == ["geist", "geist-mono", "newsreader"]
    assert body["mascot"]["shells"] == ["bronze", "gold", "silver", "cobalt"]
    assert body["mascot"]["lights"] == ["mint", "amber", "ice", "rose"]
    assert len(body["mascot"]["states"]) == 12
    assert body["shape"] == {"radius": {"min": 0, "max": 24}, "glow": {"min": 0, "max": 40}}
    assert body["text"]["limits"]["name"] == {"min": 1, "max": 40}
    assert "url(" in body["text"]["forbidden"]


def test_schema_route_serves_the_package_limits_from_config(monkeypatch):
    """S-002: the Skin Kit reads the package caps from the route; they come from config."""
    from config import config as settings

    body = _client().get("/api/skins/schema").json()
    assert body["limits"] == {
        "packageMb": settings.SKIN_MAX_PACKAGE_MB,
        "fontKb": settings.SKIN_MAX_FONT_KB,
        "mascotMb": settings.SKIN_MAX_MASCOT_MB,
        "previewKb": settings.SKIN_MAX_PREVIEW_KB,
    }
    assert body["limits"] == {"packageMb": 15, "fontKb": 400, "mascotMb": 8, "previewKb": 1024}

    monkeypatch.setattr(settings, "SKIN_MAX_FONT_KB", 123)
    assert _client().get("/api/skins/schema").json()["limits"]["fontKb"] == 123


def test_schema_route_needs_a_caller():
    response = _client(authenticated=False).get("/api/skins/schema")
    assert response.status_code == 401


def test_schema_route_is_a_plain_def_behind_the_hybrid_dependency():
    from api.marketplace_skins import get_skin_schema
    from core.auth.hybrid import get_request_context_hybrid

    assert not inspect.iscoroutinefunction(get_skin_schema)
    default = inspect.signature(get_skin_schema).parameters["ctx"].default
    assert default.dependency is get_request_context_hybrid


def test_the_router_is_registered_in_main():
    source = (ORCH_ROOT / "main.py").read_text(encoding="utf-8")
    assert "from api.marketplace_skins import router as marketplace_skins_router" in source
    assert "app.include_router(marketplace_skins_router)" in source
