"""PRD-196 S7 (P2-15, governance I.4) — GDPR onto the canonical admin dep.

The hand-rolled ``_require_workspace_admin`` in ``api/gdpr.py`` (a role-string
check, different semantics from the canonical membership-row check) is deleted:
all three GDPR routes now depend on the ONE ``require_workspace_admin`` (PRD-185
S12), the same gate the governance + approval-grant surfaces use. Pinned here:
the local helper is gone and every route carries the canonical dependency.

Pure: route-table introspection (the PRD-196 S2 wiring-test shape) — no DB.
"""
from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from api import gdpr as gdpr_api  # noqa: E402
from core.auth.workspace_admin import require_workspace_admin  # noqa: E402


def _dependant_calls(dependant) -> set:
    calls = {getattr(dependant, "call", None)}
    for sub in getattr(dependant, "dependencies", []) or []:
        calls |= _dependant_calls(sub)
    return calls


def test_hand_rolled_admin_helper_is_gone():
    assert not hasattr(gdpr_api, "_require_workspace_admin"), (
        "the hand-rolled _require_workspace_admin must be deleted — one admin semantic"
    )


def test_every_gdpr_route_carries_canonical_admin_gate():
    expected = {
        ("/api/v1/gdpr/export", "GET"),
        ("/api/v1/gdpr/erase", "POST"),
        ("/api/v1/gdpr/erase-subject", "POST"),
    }
    seen = set()
    for route in gdpr_api.router.routes:
        for method in route.methods or ():
            key = (route.path, method)
            if key in expected:
                assert require_workspace_admin in _dependant_calls(route.dependant), (
                    f"{method} {route.path} is not gated by the canonical require_workspace_admin"
                )
                seen.add(key)
    assert seen == expected, f"missing routes: {expected - seen}"
