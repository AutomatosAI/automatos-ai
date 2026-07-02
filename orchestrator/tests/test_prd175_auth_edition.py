"""PRD-175 W5 — Auth decoupling: AUTH_EDITION flag + local session + F075 staff domain.

These pin the *deployability-critical* half of the absent PRD-150 (F008/F075):

  1. Headline (review §13): with ``AUTH_EDITION=local`` and NO ``CLERK_*`` env, the
     backend resolves a request with no bearer to an authenticated *local*
     ``RequestContext`` (anonymous auth_type, the configured default workspace) —
     not a 401. This is the exact gap that blocks ``git clone && docker compose up``.
  2. ``local`` edition *implies* ``REQUIRE_AUTH=false`` — one flag, not three that
     can contradict (PRD §4.1/§4.3).
  3. Boot guard (PRD §4.3, review §5.3): a ``saas`` boot with no Clerk env fails
     fast (RuntimeError), never a silent anonymous downgrade; a ``local`` boot
     with no ``DEFAULT_WORKSPACE_ID`` fails fast.
  4. F075: the platform-staff check reads ``config.PLATFORM_STAFF_EMAIL_DOMAIN``,
     not a ``@automatos.app`` literal — changing the config changes the accepted
     domain, and no literal remains in ``clerk.py``.

All tests here are pure/DB-free: the local-session test stubs the DB seam
(``SessionLocal`` + workspace helpers) so it runs without Postgres, matching the
lean-venv idiom used across the suite. CI runs the full DB-backed path.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

# Dummy POSTGRES_* satisfies the config import chain without a live DB (blessed
# pattern; a closed port refuses instantly, nothing here touches a real DB).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

LOCAL_WS = UUID("00000000-0000-0000-0000-0000000000aa")


@pytest.fixture(autouse=True)
def _contain_reload_blast_radius():
    """Snapshot + restore the *attributes* of the two modules these tests reload.

    This file deliberately reloads ``config`` (``_fresh_config`` — to re-run the
    class-body edition resolution under a mutated env) and ``core.auth.hybrid``
    (the local-session test — to re-bind its module-level ``from config import
    config``).

    CRITICAL: ``importlib.reload`` re-executes the module body into the SAME
    module object (``sys.modules[name]`` identity is preserved) but REPLACES every
    attribute — it mints a brand-new ``config.config`` singleton and a brand-new
    ``hybrid.get_request_context_hybrid`` function object. So snapshotting
    ``sys.modules[name]`` is a no-op; the leak is at the attribute level.

    Left un-restored, that leaks two ways into sibling test files:

    * ``config``: later ``from config import config`` (e.g. ``tool_router``'s lazy
      read) binds this reloaded singleton with env defaults, silently defeating
      ``test_tool_router_semantic``'s ``_FAKE_CONFIG`` (SEMANTIC_TOOL_ROUTING back
      to its ``true`` default → ``assert True is False``).
    * ``core.auth.hybrid``: the endpoint suites override ``get_request_context_hybrid``
      via ``app.dependency_overrides``. FastAPI keys that override by function
      identity. After a reload, the routers still hold the ORIGINAL function
      (captured at their import) while the test imports the NEW one, so the
      override silently misses and the request falls through to the real
      anonymous resolver ("Workspace not resolved").

    Snapshot each module's ``__dict__`` before the test and restore it verbatim
    after (put original attributes back, drop any the reload added), so the reload
    can never escape this module. Mirrors ``test_config.py``'s
    ``_restore_config_module`` intent, corrected to operate at attribute level and
    extended to ``core.auth.hybrid``.
    """
    import config as _config_mod
    import core.auth.hybrid as _hybrid_mod

    snapshots = {
        _config_mod: dict(_config_mod.__dict__),
        _hybrid_mod: dict(_hybrid_mod.__dict__),
    }
    try:
        yield
    finally:
        for mod, saved_dict in snapshots.items():
            mod.__dict__.clear()
            mod.__dict__.update(saved_dict)


def _fresh_config(**env):
    """Reimport config.py under a controlled environment and return its singleton.

    ``AUTH_EDITION``/``REQUIRE_AUTH`` are resolved at class-definition time, so the
    module must be reloaded to pick up a changed edition. We snapshot/restore the
    relevant env keys so tests don't leak into each other.
    """
    keys = [
        "AUTH_EDITION", "REQUIRE_AUTH", "DEFAULT_WORKSPACE_ID",
        "CLERK_JWKS_URL", "CLERK_SECRET_KEY", "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY",
        "PLATFORM_STAFF_EMAIL_DOMAIN",
    ]
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in env.items() if v is not None})
    try:
        import config as config_mod
        importlib.reload(config_mod)
        return config_mod.config
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# §5.2 (backend) / §4.1 — the edition flag & its implications
# ---------------------------------------------------------------------------

def test_default_edition_is_saas_and_require_auth_stays_secure():
    """Default (no flag) must remain the SaaS product: edition=saas, auth on."""
    cfg = _fresh_config()  # no AUTH_EDITION, no REQUIRE_AUTH override
    assert cfg.AUTH_EDITION == "saas"
    assert cfg.IS_SAAS_EDITION is True
    assert cfg.IS_LOCAL_EDITION is False
    # REQUIRE_AUTH secure-by-default in saas (unset env → true).
    assert cfg.REQUIRE_AUTH is True


def test_local_edition_forces_require_auth_false():
    """PRD §4.1: local *implies* the no-login posture; REQUIRE_AUTH is forced false
    even if the raw env says true, so operators set one flag, not three."""
    cfg = _fresh_config(AUTH_EDITION="local", DEFAULT_WORKSPACE_ID=str(LOCAL_WS),
                        REQUIRE_AUTH="true")
    assert cfg.AUTH_EDITION == "local"
    assert cfg.IS_LOCAL_EDITION is True
    assert cfg.REQUIRE_AUTH is False


def test_saas_edition_respects_require_auth_env():
    cfg = _fresh_config(AUTH_EDITION="saas", REQUIRE_AUTH="false",
                        CLERK_JWKS_URL="https://x/.well-known/jwks.json",
                        CLERK_SECRET_KEY="sk_test")
    assert cfg.REQUIRE_AUTH is False


def test_unknown_edition_falls_back_to_saas():
    """An invalid value must not silently disable auth — fail safe to saas."""
    cfg = _fresh_config(AUTH_EDITION="banana")
    assert cfg.AUTH_EDITION == "saas"
    assert cfg.REQUIRE_AUTH is True


# ---------------------------------------------------------------------------
# §4.3 — boot guard (the silent-downgrade mitigation)
# ---------------------------------------------------------------------------

def test_boot_guard_saas_without_clerk_fails_fast():
    """review §5.3: a saas boot that lost its Clerk env must abort boot, NOT fall
    through to the anonymous local identity and serve tenant data unauthenticated."""
    cfg = _fresh_config(AUTH_EDITION="saas")  # no CLERK_* set
    with pytest.raises(RuntimeError) as exc:
        cfg.validate_auth_edition()
    assert "CLERK" in str(exc.value).upper()


def test_boot_guard_saas_with_clerk_passes():
    cfg = _fresh_config(AUTH_EDITION="saas",
                        CLERK_JWKS_URL="https://x/.well-known/jwks.json",
                        CLERK_SECRET_KEY="sk_test")
    cfg.validate_auth_edition()  # must not raise


def test_boot_guard_local_requires_default_workspace():
    cfg = _fresh_config(AUTH_EDITION="local")  # no DEFAULT_WORKSPACE_ID
    with pytest.raises(RuntimeError) as exc:
        cfg.validate_auth_edition()
    assert "DEFAULT_WORKSPACE_ID" in str(exc.value)


def test_boot_guard_local_with_workspace_passes_without_clerk():
    """The headline: local needs a workspace, needs NO Clerk env."""
    cfg = _fresh_config(AUTH_EDITION="local", DEFAULT_WORKSPACE_ID=str(LOCAL_WS))
    cfg.validate_auth_edition()  # must not raise even with zero CLERK_* env


def test_validate_auth_edition_is_wired_into_validate_security():
    """The guard runs at the same hard-fail boot phase as PRD-172 (main.py:178)."""
    cfg = _fresh_config(AUTH_EDITION="saas")  # no clerk → must fail via validate_security
    with pytest.raises(RuntimeError):
        cfg.validate_security()


# ---------------------------------------------------------------------------
# §5.1 headline — backend serves a local session with zero Clerk env
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_session_no_bearer_no_clerk_resolves_authenticated():
    """With AUTH_EDITION=local, DEFAULT_WORKSPACE_ID seeded, and NO CLERK_* env, a
    request with no bearer resolves to an authenticated anonymous/local context on
    the configured workspace — not a 401."""
    import config as config_mod
    importlib.reload(config_mod)
    import core.auth.hybrid as hybrid
    importlib.reload(hybrid)

    # Drive the module's live config into the local posture.
    with patch.object(hybrid.config, "AUTH_EDITION", "local"), \
         patch.object(hybrid.config, "REQUIRE_AUTH", False), \
         patch.object(hybrid.config, "DEFAULT_WORKSPACE_ID", str(LOCAL_WS)), \
         patch.object(hybrid.config, "WORKSPACE_ID", None), \
         patch.object(hybrid.config, "AUTH_DEBUG", False), \
         patch.object(hybrid, "SessionLocal", MagicMock(return_value=MagicMock())), \
         patch.object(hybrid, "_workspace_exists", return_value=True), \
         patch.object(hybrid, "_assert_workspace_usable", return_value=None), \
         patch.object(hybrid, "_enrich_log_context", return_value=None):

        request = MagicMock()
        request.method = "GET"
        request.headers = {}
        request.query_params = {}
        request.state = MagicMock(spec=[])  # no pre-resolved workspace

        ctx = await hybrid.get_request_context_hybrid(request)

    assert ctx.auth_type == "anonymous"
    assert ctx.workspace_id == LOCAL_WS


# ---------------------------------------------------------------------------
# §5.4 — F075: staff domain comes from config, not a literal
# ---------------------------------------------------------------------------

def _make_clerk_auth():
    import core.auth.clerk as clerk_mod
    importlib.reload(clerk_mod)
    return clerk_mod, clerk_mod.ClerkAuth()


def test_admin_kept_for_configured_staff_domain():
    clerk_mod, auth = _make_clerk_auth()
    with patch.object(clerk_mod.config, "PLATFORM_STAFF_EMAIL_DOMAIN", "automatos.app"):
        info = auth.extract_user_info(
            {"sub": "u1", "email": "gerard@automatos.app", "metadata": {"role": "admin"}}
        )
    assert info["system_role"] == "admin"


def test_admin_demoted_for_non_staff_domain():
    clerk_mod, auth = _make_clerk_auth()
    with patch.object(clerk_mod.config, "PLATFORM_STAFF_EMAIL_DOMAIN", "automatos.app"):
        info = auth.extract_user_info(
            {"sub": "u2", "email": "attacker@evil.com", "metadata": {"role": "admin"}}
        )
    assert info["system_role"] == "user"


def test_staff_domain_is_configurable():
    """Changing config value changes the accepted domain (self-host operator sets
    their own staff domain) — proves it's config, not a baked-in literal."""
    clerk_mod, auth = _make_clerk_auth()
    with patch.object(clerk_mod.config, "PLATFORM_STAFF_EMAIL_DOMAIN", "acme.example"):
        kept = auth.extract_user_info(
            {"sub": "u3", "email": "ops@acme.example", "metadata": {"role": "admin"}}
        )
        demoted = auth.extract_user_info(
            {"sub": "u4", "email": "x@automatos.app", "metadata": {"role": "admin"}}
        )
    assert kept["system_role"] == "admin"
    assert demoted["system_role"] == "user"


def test_no_automatos_literal_remains_in_clerk_py():
    """review §5.4: assert by grep that the hardcoded @automatos.app staff-gate
    literal is gone from clerk.py (the git-identity default lives elsewhere)."""
    clerk_py = Path(__file__).resolve().parents[1] / "core" / "auth" / "clerk.py"
    src = clerk_py.read_text()
    assert "@automatos.app" not in src
    assert "PLATFORM_STAFF_EMAIL_DOMAIN" in src
