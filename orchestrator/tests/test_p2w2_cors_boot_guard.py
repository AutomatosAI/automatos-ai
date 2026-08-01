"""The security boot guard must actually abort a boot, and widget CORS must
fail closed for origins nobody named.

History. PRD-194 S4 added a guard that aborted a saas boot when
``WIDGET_ORIGIN_ALLOWLIST`` was empty. Two things were wrong with it:

1. **It never ran.** ``config.validate_security()`` was called inside
   ``_boot_phase_1_core``, which ``run_stage`` wraps in ``except Exception``
   — the stage is recorded failed, a warning is logged, and boot continues.
   The guard landed 2026-07-11; that wrapper predates it by two months, so it
   had never once aborted a boot. Production served traffic with the exact
   config the guard was written to reject, and everything after the raise in
   ``_boot_phase_1_core`` (``create_tables``, the idempotent column
   migrations) was skipped on every boot.

2. **It guarded the wrong thing.** ``WIDGET_ORIGIN_ALLOWLIST`` was OR'd with
   the per-key ``SdkApiKey.allowed_domains`` lookup, so it could only ever
   WIDEN access beyond what a key permitted — no defence in depth, and a
   second place to forget that threw a 403 reading as a key problem. The var
   is deleted; merchant origins resolve from the key, first-party origins
   from ``config.CORS_ALLOW_ORIGINS``.

These tests pin both: the guard is reachable, and the plane fails closed.

Pure tests — static AST read of main.py, plus config-object and pure-function
checks. No boot, no DB, no network.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from config import config as cfg  # noqa: E402

_MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


# ---------------------------------------------------------------------------
# The guard must be reachable — not wrapped in a swallowing run_stage
# ---------------------------------------------------------------------------

def _calls_validate_security(func_name: str) -> bool:
    """True if the named top-level function calls config.validate_security()."""
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != func_name:
            continue
        return any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "validate_security"
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    raise AssertionError(f"{func_name} not found in main.py")


def test_lifespan_calls_validate_security_directly():
    """It must run in lifespan, where a raise aborts the boot."""
    assert _calls_validate_security("lifespan") is True


def test_boot_phase_1_core_does_not_call_validate_security():
    """The regression guard. _boot_phase_1_core runs inside
    run_stage(...), which swallows every exception — a fail-closed check in
    there logs a warning and lets the server serve traffic anyway."""
    assert _calls_validate_security("_boot_phase_1_core") is False


def test_run_stage_still_swallows_which_is_why_placement_matters():
    """Documents the mechanism the tests above defend against. run_stage is
    deliberately forgiving (a failed extension must not stop boot) — that is
    exactly why a hard security guard cannot live inside one."""
    import inspect

    from core.models.bootstrap import run_stage

    src = inspect.getsource(run_stage)
    assert "except Exception" in src
    # No bare re-raise of the captured stage exception.
    assert "raise" not in src.split("except Exception")[1]


# ---------------------------------------------------------------------------
# validate_security still enforces the checks it kept
# ---------------------------------------------------------------------------

@pytest.fixture()
def _security_baseline(monkeypatch):
    """Satisfy every other validate_security check so one is isolated."""
    monkeypatch.setattr(cfg, "SHOPIFY_INTERNAL_API_KEY", "test-internal-key")
    monkeypatch.setattr(cfg, "S3_VECTORS_ENABLED", False)
    monkeypatch.setattr(cfg, "CLERK_JWKS_URL", "https://x/.well-known/jwks.json")
    monkeypatch.setattr(cfg, "CLERK_SECRET_KEY", "sk_test")
    monkeypatch.setattr(cfg, "DEFAULT_WORKSPACE_ID", "00000000-0000-0000-0000-0000000000aa")
    return monkeypatch


def test_shopify_fail_open_still_aborts(_security_baseline):
    """F004 is untouched by the widget-allowlist removal."""
    mp = _security_baseline
    mp.setattr(cfg, "SHOPIFY_INTERNAL_API_KEY", "")
    with pytest.raises(RuntimeError) as ei:
        cfg.validate_security()
    assert "SHOPIFY_INTERNAL_API_KEY" in str(ei.value)


def test_widget_cors_is_no_longer_a_boot_concern(_security_baseline):
    """No global widget allowlist exists, so boot has nothing to validate —
    a saas boot with no widget env config must succeed."""
    mp = _security_baseline
    mp.setattr(cfg, "AUTH_EDITION", "saas")
    cfg.validate_security()  # must not raise

    assert not hasattr(cfg, "WIDGET_ORIGIN_ALLOWLIST"), (
        "WIDGET_ORIGIN_ALLOWLIST is deleted — a merchant origin belongs on "
        "the merchant's key, not in a global env var"
    )


# ---------------------------------------------------------------------------
# Runtime: the plane fails closed
# ---------------------------------------------------------------------------

def test_unknown_origin_denied_when_no_platform_origins(monkeypatch):
    """Empty platform origins is not allow-all. It denies."""
    import api.widgets.cors as cors_mod

    monkeypatch.setattr(cors_mod, "PLATFORM_ORIGINS", set())
    assert cors_mod._origin_is_platform("https://any-store.example") is False


def test_platform_origin_matching_is_exact(monkeypatch):
    """No suffix/substring matching — evil-automatos.app is not automatos.app."""
    import api.widgets.cors as cors_mod

    monkeypatch.setattr(cors_mod, "PLATFORM_ORIGINS", {"https://automatos.app"})
    assert cors_mod._origin_is_platform("https://automatos.app") is True
    assert cors_mod._origin_is_platform("https://evil-automatos.app") is False
    assert cors_mod._origin_is_platform("https://automatos.app.evil.com") is False
