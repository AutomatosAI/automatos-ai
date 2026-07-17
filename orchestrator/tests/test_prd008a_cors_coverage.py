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


def test_origin_allowed_when_no_allowlist_configured(monkeypatch):
    """Empty allowlist → permissive ONLY in the local edition (PRD-194 S4 /
    P2-13). In saas the boot guard forbids the state; if reached anyway the
    public plane fails CLOSED."""
    from api.widgets.cors import _origin_allowed
    import api.widgets.cors as cors_mod

    monkeypatch.setattr(cors_mod, "WIDGET_ORIGIN_ALLOWLIST", set())

    monkeypatch.setattr(cors_mod.config, "AUTH_EDITION", "local")
    assert _origin_allowed("https://random-store.myshopify.com") is True
    assert _origin_allowed("https://app.automatos.app") is True

    monkeypatch.setattr(cors_mod.config, "AUTH_EDITION", "saas")
    assert _origin_allowed("https://random-store.myshopify.com") is False
    assert _origin_allowed("https://app.automatos.app") is False


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


# ---------------------------------------------------------------------------
# PRD-TUTOR-LIVE S0 — key-allowlist preflight fallback
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def _isolated_dynamic(monkeypatch):
    """Fresh cache + saas edition + an env allowlist that misses the origin,
    so every test exercises the fallback path deliberately."""
    import api.widgets.cors as cors_mod

    monkeypatch.setattr(cors_mod, "_dynamic_origin_cache", {})
    monkeypatch.setattr(cors_mod, "WIDGET_ORIGIN_ALLOWLIST", {"https://app.automatos.app"})
    monkeypatch.setattr(cors_mod.config, "AUTH_EDITION", "saas")
    return cors_mod


def test_preflight_falls_back_to_key_allowlist(monkeypatch):
    """An origin absent from the env allowlist but named on an active public
    key's allowed_domains passes the dynamic check (the academy case)."""
    cors_mod = _isolated_dynamic(monkeypatch)
    monkeypatch.setattr(cors_mod, "_origin_allowed_by_key_sync", lambda origin: True)

    assert asyncio.run(cors_mod._origin_allowed_dynamic("https://academy.automatos.app")) is True


def test_key_fallback_denies_unknown_origin(monkeypatch):
    cors_mod = _isolated_dynamic(monkeypatch)
    monkeypatch.setattr(cors_mod, "_origin_allowed_by_key_sync", lambda origin: False)

    assert asyncio.run(cors_mod._origin_allowed_dynamic("https://malicious.example.com")) is False


def test_env_fast_path_skips_the_db(monkeypatch):
    cors_mod = _isolated_dynamic(monkeypatch)

    def _boom(origin):
        raise AssertionError("DB lookup must not run for env-allowlisted origins")

    monkeypatch.setattr(cors_mod, "_origin_allowed_by_key_sync", _boom)
    assert asyncio.run(cors_mod._origin_allowed_dynamic("https://app.automatos.app")) is True


def test_key_fallback_verdict_is_cached(monkeypatch):
    """Second ask within the TTL answers from cache — no second DB scan."""
    cors_mod = _isolated_dynamic(monkeypatch)
    calls = []

    def _count(origin):
        calls.append(origin)
        return True

    monkeypatch.setattr(cors_mod, "_origin_allowed_by_key_sync", _count)
    assert asyncio.run(cors_mod._origin_allowed_dynamic("https://academy.automatos.app")) is True
    assert asyncio.run(cors_mod._origin_allowed_dynamic("https://academy.automatos.app")) is True
    assert len(calls) == 1


def test_key_fallback_fails_closed_and_does_not_cache_failures(monkeypatch):
    """A lookup ERROR denies now but is retried next time (only verdicts
    cache); a scanning client cannot poison the cache with an outage."""
    cors_mod = _isolated_dynamic(monkeypatch)

    def _down(origin):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(cors_mod, "_origin_allowed_by_key_sync", _down)
    assert asyncio.run(cors_mod._origin_allowed_dynamic("https://academy.automatos.app")) is False

    monkeypatch.setattr(cors_mod, "_origin_allowed_by_key_sync", lambda origin: True)
    assert asyncio.run(cors_mod._origin_allowed_dynamic("https://academy.automatos.app")) is True


def test_origin_allowed_by_any_key_uses_check_domain_matcher():
    """Service-level: the fallback consults the same matcher widget_auth
    uses (host-only, fnmatch wildcards); empty-allowlist keys never count."""
    from core.services.api_key_service import ApiKeyService

    class _StubQuery:
        def __init__(self, rows):
            self._rows = rows

        def filter(self, *_args):
            return self

        def all(self):
            return self._rows

    class _StubDb:
        def __init__(self, rows):
            self._rows = rows

        def query(self, _model):
            return _StubQuery(self._rows)

    keyed = SimpleNamespace(allowed_domains=["academy.automatos.app", "*.up.railway.app"])
    unrestricted = SimpleNamespace(allowed_domains=[])  # "any origin" opt-in — must NOT count

    db = _StubDb([keyed])
    assert ApiKeyService.origin_allowed_by_any_key(db, "https://academy.automatos.app") is True
    assert ApiKeyService.origin_allowed_by_any_key(db, "https://automatos-academy-production.up.railway.app") is True
    assert ApiKeyService.origin_allowed_by_any_key(db, "https://malicious.example.com") is False

    assert ApiKeyService.origin_allowed_by_any_key(_StubDb([unrestricted]), "https://anything.example.com") is False
    assert ApiKeyService.origin_allowed_by_any_key(_StubDb([]), "https://academy.automatos.app") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
