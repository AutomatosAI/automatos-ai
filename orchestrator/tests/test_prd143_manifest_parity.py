"""PRD-143 S4 — the obs-tier manifest and the live registry never drift.

The Rev 2 inversion: after S4, ZERO actions are exclusion-gated from Auto
except the obs/oversight 7 (super_admin_only). The manifest at
docs/PRDS/PRD-143-OBS-TIER-MANIFEST.md is the human-readable, sign-off-able
source of truth; this suite enforces exact set-equality with the registry so
the tier stays honest forever.

Loads the REAL catalogue via register_all_actions() — idiom mirrors
tests/test_platform_actions_registration.py (dummy POSTGRES_* + apscheduler
stub), with S2's closed-port twist so a wedged local postgres proxy cannot
hang the fail-soft import-time connect. Nothing here touches a DB.
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import re
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


def _install_fake_apscheduler():
    if "apscheduler" in sys.modules:
        return
    aps = types.ModuleType("apscheduler")
    schedulers = types.ModuleType("apscheduler.schedulers")
    asyncio_mod = types.ModuleType("apscheduler.schedulers.asyncio")
    asyncio_mod.AsyncIOScheduler = type("AsyncIOScheduler", (), {})
    jobstores = types.ModuleType("apscheduler.jobstores")
    memory_mod = types.ModuleType("apscheduler.jobstores.memory")
    memory_mod.MemoryJobStore = type("MemoryJobStore", (), {})
    aps.schedulers = schedulers
    aps.jobstores = jobstores
    schedulers.asyncio = asyncio_mod
    jobstores.memory = memory_mod
    sys.modules.update({
        "apscheduler": aps,
        "apscheduler.schedulers": schedulers,
        "apscheduler.schedulers.asyncio": asyncio_mod,
        "apscheduler.jobstores": jobstores,
        "apscheduler.jobstores.memory": memory_mod,
    })


_install_fake_apscheduler()


# Lean-venv shim: modules/tools/__init__ pulls modules.rag's ingestion chain
# (camelot at module top). Stub the missing leaf only when truly absent.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.platform_actions import register_all_actions  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "docs" / "PRDS" / "PRD-143-OBS-TIER-MANIFEST.md"

# First table cell looks like `| `platform_x` | ...` — backticks optional.
_TOOL_ROW = re.compile(r"^\|\s*`?(platform_\w+)`?\s*\|")


@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    reg = ActionRegistry()
    # Direct registration (not the singleton) so _ensure_initialized cannot
    # trigger a second full init; read reg._actions per the blessed idiom.
    register_all_actions(reg)
    reg._initialized = True
    return reg


def _manifest_action_names() -> set[str]:
    assert _MANIFEST_PATH.exists(), (
        f"obs-tier manifest missing at {_MANIFEST_PATH} — S4 must author it"
    )
    names: set[str] = set()
    for line in _MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        match = _TOOL_ROW.match(line.strip())
        if match:
            names.add(match.group(1))
    return names


def test_manifest_matches_registry(registry):
    """Exact set-equality: manifest table ⇆ super_admin_only actions."""
    registry_su = {a.name for a in registry.get_all() if a.super_admin_only}
    manifest_su = _manifest_action_names()

    assert manifest_su, "manifest table parsed to an empty set — table malformed?"
    assert registry_su, "registry has no super_admin_only actions — S4 not applied?"
    assert manifest_su == registry_su, (
        "obs-tier drift: manifest and registry disagree.\n"
        f"  in manifest only: {sorted(manifest_su - registry_su)}\n"
        f"  in registry only: {sorted(registry_su - manifest_su)}\n"
        "Update docs/PRDS/PRD-143-OBS-TIER-MANIFEST.md (needs Gerard's sign-off)."
    )


def test_no_admin_only_actions_remain(registry):
    """Rev 2 inversion: the admin_only TIER is empty (mechanism kept for future)."""
    admin_only = sorted(a.name for a in registry.get_all() if a.admin_only)
    assert admin_only == [], (
        f"actions still admin_only after the Rev 2 reclassification: {admin_only}"
    )


def test_get_autonomy_level_is_operator_tier(registry):
    """Auto may READ its own dial — get stays operator-reachable."""
    action = registry._actions.get("platform_get_autonomy_level")
    assert action is not None, "platform_get_autonomy_level not registered"
    assert action.super_admin_only is False
    assert action.admin_only is False


def test_set_autonomy_level_is_su_tier(registry):
    """The kill-switch dial stays HUMAN — set is super_admin_only."""
    action = registry._actions.get("platform_set_autonomy_level")
    assert action is not None, "platform_set_autonomy_level not registered"
    assert action.super_admin_only is True
    assert action.admin_only is False
