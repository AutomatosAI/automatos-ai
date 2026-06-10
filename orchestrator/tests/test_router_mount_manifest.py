"""PRD-155 S3 — startup mount honesty.

The loader replaces main.py's silent ``try/except ImportError: router = None``
pattern. A *required* router that fails to import must fail boot loudly, naming
the router; an *optional* one (gated on an optional integration package) is
logged and skipped. ``ALLOW_DEGRADED_BOOT`` downgrades a required failure to a
logged skip so an operator can still boot a degraded instance on purpose.

These tests inject a fake importer, so they touch no real ``api.*`` module, no
app, and no database — fast and collection-order-safe (no ``modules.*`` /
``consumers.*`` import, hence no ``_sys_guard`` block needed).
"""
from __future__ import annotations

import types

import pytest

from router_manifest import (
    MANIFEST_ROUTERS,
    RouterMountError,
    RouterSpec,
    load_routers,
    mount_manifest_routers,
)


def _fake_importer(fail: set[str] | None = None):
    """Return an importer that yields a module exposing the usual router attrs,
    or raises ImportError for any module name in ``fail``."""
    fail = fail or set()

    def _import(module: str):
        if module in fail:
            raise ImportError(f"No module named {module!r}")
        return types.SimpleNamespace(
            router=object(),
            agent_telemetry_router=object(),
        )

    return _import


def test_required_router_failure_raises_naming_the_router():
    specs = [RouterSpec("api.alpha"), RouterSpec("api.beta")]
    importer = _fake_importer(fail={"api.beta"})
    with pytest.raises(RouterMountError) as exc:
        load_routers(specs, allow_degraded=False, importer=importer)
    # The message must name the failed router so a red boot is actionable.
    assert "api.beta" in str(exc.value)
    assert "api.alpha" not in str(exc.value)
    assert "ALLOW_DEGRADED_BOOT" in str(exc.value)


def test_allow_degraded_boot_downgrades_required_failure():
    specs = [RouterSpec("api.alpha"), RouterSpec("api.beta")]
    importer = _fake_importer(fail={"api.beta"})
    mounted, failures = load_routers(specs, allow_degraded=True, importer=importer)
    assert [s.module for s, _ in mounted] == ["api.alpha"]
    assert [s.module for s, _ in failures] == ["api.beta"]


def test_optional_router_failure_never_raises():
    specs = [RouterSpec("api.alpha"), RouterSpec("api.gamma", optional=True)]
    importer = _fake_importer(fail={"api.gamma"})
    # Even with allow_degraded=False an optional failure is tolerated.
    mounted, failures = load_routers(specs, allow_degraded=False, importer=importer)
    assert [s.module for s, _ in mounted] == ["api.alpha"]
    assert [s.module for s, _ in failures] == ["api.gamma"]


def test_all_present_mounts_everything():
    specs = [RouterSpec("api.alpha"), RouterSpec("api.beta", optional=True)]
    mounted, failures = load_routers(specs, allow_degraded=False, importer=_fake_importer())
    assert {s.module for s, _ in mounted} == {"api.alpha", "api.beta"}
    assert failures == []


def test_mount_manifest_routers_includes_each_resolved_router():
    calls: list[object] = []
    app = types.SimpleNamespace(include_router=lambda r, **kw: calls.append(r))
    specs = [RouterSpec("api.alpha"), RouterSpec("api.beta", attr="agent_telemetry_router")]
    mount_manifest_routers(app, allow_degraded=False, specs=specs, importer=_fake_importer())
    assert len(calls) == 2


def test_real_manifest_is_well_formed_and_excludes_dead_modules():
    assert MANIFEST_ROUTERS, "manifest must not be empty"
    modules = {s.module for s in MANIFEST_ROUTERS}
    # The two imports that fail every boot (modules deleted in this story) must
    # never reappear in the manifest.
    assert "api.auth" not in modules
    assert "api.evaluation" not in modules
    for s in MANIFEST_ROUTERS:
        assert isinstance(s, RouterSpec)
        assert s.module.startswith("api.")
        assert s.attr in {"router", "agent_telemetry_router"}
        assert s.name == f"{s.module}:{s.attr}"
