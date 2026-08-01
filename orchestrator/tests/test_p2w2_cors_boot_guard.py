"""PRD-194 S4 (P2-13, security §1.2.b) — widget CORS empty-allowlist boot guard.

``api/widgets/cors.py`` allowed ALL origins when ``WIDGET_ORIGIN_ALLOWLIST``
was unset — which is the default — and nothing enforced the docstring's "in
production the env var should always be set". These tests pin the locked
decision: **a saas boot with an empty widget allowlist ABORTS** (added to
``config.validate_security``, the same hard-fail boot phase and posture as
the ``SHOPIFY_INTERNAL_API_KEY`` and Clerk edition guards), while the
``local`` edition keeps the permissive dev default — choosing
``AUTH_EDITION=local`` is the explicit opt-in. Belt-and-braces: if the
empty-allowlist state is ever reached in saas anyway, ``_origin_allowed``
fails CLOSED instead of allow-all.

Pure config-object tests — attributes are monkeypatched on the singleton
(the established ``validate_*`` shape); no env reloads, no boot, no network.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from config import config as cfg  # noqa: E402


@pytest.fixture()
def _security_baseline(monkeypatch):
    """Satisfy every OTHER validate_security check so the widget-allowlist
    guard is the only variable under test."""
    monkeypatch.setattr(cfg, "SHOPIFY_INTERNAL_API_KEY", "test-internal-key")
    monkeypatch.setattr(cfg, "S3_VECTORS_ENABLED", False)
    monkeypatch.setattr(cfg, "CLERK_JWKS_URL", "https://x/.well-known/jwks.json")
    monkeypatch.setattr(cfg, "CLERK_SECRET_KEY", "sk_test")
    monkeypatch.setattr(cfg, "DEFAULT_WORKSPACE_ID", "00000000-0000-0000-0000-0000000000aa")
    return monkeypatch


def test_saas_empty_widget_allowlist_aborts_boot(_security_baseline):
    """AUTH_EDITION=saas + empty WIDGET_ORIGIN_ALLOWLIST ⇒ RuntimeError
    (boot abort) — the public plane must never rest at allow-all in prod."""
    mp = _security_baseline
    mp.setattr(cfg, "AUTH_EDITION", "saas")
    mp.setattr(cfg, "WIDGET_ORIGIN_ALLOWLIST", "")
    with pytest.raises(RuntimeError) as ei:
        cfg.validate_security()
    assert "WIDGET_ORIGIN_ALLOWLIST" in str(ei.value)


def test_saas_whitespace_allowlist_also_aborts(_security_baseline):
    """A whitespace-only value is still 'unset' — no accidental pass."""
    mp = _security_baseline
    mp.setattr(cfg, "AUTH_EDITION", "saas")
    mp.setattr(cfg, "WIDGET_ORIGIN_ALLOWLIST", "   ")
    with pytest.raises(RuntimeError):
        cfg.validate_security()


def test_saas_with_allowlist_boots(_security_baseline):
    mp = _security_baseline
    mp.setattr(cfg, "AUTH_EDITION", "saas")
    mp.setattr(
        cfg, "WIDGET_ORIGIN_ALLOWLIST",
        "https://inbuilduk.myshopify.com,https://app.automatos.app",
    )
    cfg.validate_security()  # must not raise


def test_local_empty_allowlist_permitted(_security_baseline):
    """The local edition keeps the permissive dev default — an empty
    allowlist must NOT abort a local boot."""
    mp = _security_baseline
    mp.setattr(cfg, "AUTH_EDITION", "local")
    mp.setattr(cfg, "WIDGET_ORIGIN_ALLOWLIST", "")
    cfg.validate_security()  # must not raise


# ---------------------------------------------------------------- runtime belt

def test_cors_empty_allowlist_denies_in_saas(monkeypatch):
    """If the empty-allowlist state is ever reached in saas (guard bypassed),
    _origin_allowed fails CLOSED — never allow-all in production."""
    import api.widgets.cors as cors_mod

    monkeypatch.setattr(cors_mod, "WIDGET_ORIGIN_ALLOWLIST", set())
    monkeypatch.setattr(cors_mod.config, "AUTH_EDITION", "saas")
    assert cors_mod._origin_allowed("https://any-store.example") is False


def test_cors_empty_allowlist_permits_in_local(monkeypatch):
    import api.widgets.cors as cors_mod

    monkeypatch.setattr(cors_mod, "WIDGET_ORIGIN_ALLOWLIST", set())
    monkeypatch.setattr(cors_mod.config, "AUTH_EDITION", "local")
    assert cors_mod._origin_allowed("https://any-store.example") is True


def test_cors_configured_allowlist_still_authoritative(monkeypatch):
    """A configured allowlist behaves identically in both editions."""
    import api.widgets.cors as cors_mod

    monkeypatch.setattr(
        cors_mod, "WIDGET_ORIGIN_ALLOWLIST", {"https://app.automatos.app"}
    )
    for edition in ("saas", "local"):
        monkeypatch.setattr(cors_mod.config, "AUTH_EDITION", edition)
        assert cors_mod._origin_allowed("https://app.automatos.app") is True
        assert cors_mod._origin_allowed("https://evil.example") is False
