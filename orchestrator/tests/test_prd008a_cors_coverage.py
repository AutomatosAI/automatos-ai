"""
PRD-008-A CORS path-coverage trip-wire
========================================

Locks in which path prefixes are covered by ``WidgetCORSMiddleware``.
If a future PR adds a new public surface (e.g. ``/api/admin-widgets/*``)
without extending the middleware, this test forces the conversation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import config  # noqa: E402,F401


def test_covered_path_prefixes_lock():
    """Trip-wire: changing this list is a deliberate decision."""
    from api.widgets.cors import COVERED_PATH_PREFIXES

    assert COVERED_PATH_PREFIXES == ("/api/widgets", "/api/sites")


def test_path_is_covered_widget_routes():
    from api.widgets.cors import _path_is_covered

    assert _path_is_covered("/api/widgets/config") is True
    assert _path_is_covered("/api/widgets/callback") is True
    assert _path_is_covered("/api/widgets/session") is True


def test_path_is_covered_sites_routes():
    """PRD-008-A added Sites — the CORS middleware must cover them so
    the dashboard browser can issue PATCH /api/sites/{id}/settings."""
    from api.widgets.cors import _path_is_covered

    assert _path_is_covered("/api/sites") is True
    assert _path_is_covered("/api/sites/abc-123") is True
    assert _path_is_covered("/api/sites/abc-123/settings") is True


def test_path_is_covered_excludes_unrelated_paths():
    """Don't accidentally claim CORS responsibility for paths owned by
    FastAPI's default CORSMiddleware (with its own allowlist)."""
    from api.widgets.cors import _path_is_covered

    assert _path_is_covered("/api/agents") is False
    assert _path_is_covered("/api/workspaces/current") is False
    assert _path_is_covered("/api/composio/anything") is False
    # Edge: a path that contains /api/sites as a substring but doesn't start with it
    assert _path_is_covered("/api/some-other-resource/sites") is False


def test_origin_allowed_when_no_allowlist_configured():
    """Empty allowlist → permissive — matches the documented default."""
    from api.widgets.cors import _origin_allowed
    import api.widgets.cors as cors_mod

    original = cors_mod.WIDGET_ORIGIN_ALLOWLIST
    cors_mod.WIDGET_ORIGIN_ALLOWLIST = set()
    try:
        assert _origin_allowed("https://random-store.myshopify.com") is True
        assert _origin_allowed("https://app.automatos.app") is True
    finally:
        cors_mod.WIDGET_ORIGIN_ALLOWLIST = original


def test_origin_allowed_with_explicit_allowlist():
    from api.widgets.cors import _origin_allowed
    import api.widgets.cors as cors_mod

    original = cors_mod.WIDGET_ORIGIN_ALLOWLIST
    cors_mod.WIDGET_ORIGIN_ALLOWLIST = {
        "https://app.automatos.app",
        "https://besafe-ltd.myshopify.com",
    }
    try:
        assert _origin_allowed("https://app.automatos.app") is True
        assert _origin_allowed("https://besafe-ltd.myshopify.com") is True
        assert _origin_allowed("https://malicious.example.com") is False
    finally:
        cors_mod.WIDGET_ORIGIN_ALLOWLIST = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
