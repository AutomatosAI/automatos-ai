"""Config deletion guard — a setting must never vanish from main unnoticed.

WHY THIS EXISTS (2026-08-29). A PR cut from a stale local main carried a
three-week-old ``config.py`` through a squash merge, and ``ONBOARDING_RESET_ENABLED``
silently disappeared from the tree. The route that reads it does
``config.ONBOARDING_RESET_ENABLED``, so every call became an AttributeError →
HTTP 500 in production, and the dev-reset endpoint was dead until someone
diagnosed it by hand.

The same squash also regressed ``route-manifest.json`` — and THAT went red in
CI within minutes, because the manifest has a guard. Identical clobber, same
commit, opposite outcomes, decided purely by whether a guard existed. This is
that guard for the config surface.

CONTRACT (deliberately one-directional):
  * BLOCKING — every name in ``reports/config-surface.json`` must still resolve
    on the live Config. A deletion fails CI before it can reach production.
  * NON-BLOCKING — a NEW setting missing from the manifest is reported, never
    failed. Adding a flag must stay frictionless, or people route around the
    guard and it stops protecting anything.

There is deliberately NO hand-maintained count in the manifest: route-manifest's
count collided twice in one week when parallel branches each hand-bumped it, and
reintroducing that pattern here would recreate the collision class this guard is
meant to close. The list of names IS the guard.
"""
from __future__ import annotations

import inspect
import json
import pathlib

import pytest

MANIFEST_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "reports" / "config-surface.json"
)


def _manifest_names() -> list[str]:
    return json.loads(MANIFEST_PATH.read_text())["settings"]


def _live_names() -> set[str]:
    from config import config

    from scripts.regen_config_surface import setting_names

    return set(setting_names(config))


# --------------------------------------------------------------------------- #
# BLOCKING — a setting may never disappear
# --------------------------------------------------------------------------- #


def test_manifest_exists_and_is_names_only():
    doc = json.loads(MANIFEST_PATH.read_text())
    assert isinstance(doc["settings"], list) and doc["settings"]
    # Names only — this repo is public and several settings hold credentials.
    assert all(isinstance(n, str) and n.isupper() for n in doc["settings"])


def test_no_setting_has_been_deleted():
    """The guard. Every manifest name must still resolve on Config.

    If this fails, a setting was REMOVED. Either restore it (usually the right
    answer — check for a stale-base merge, which is how this class arises), or
    if the removal is intended, regenerate:
        python3 scripts/regen_config_surface.py
    """
    missing = sorted(set(_manifest_names()) - _live_names())
    assert not missing, (
        f"{len(missing)} config setting(s) vanished from Config: {missing}. "
        "A setting disappearing is almost always a stale-base merge or a bad "
        "conflict resolution — the 2026-08-29 incident took down the dev-reset "
        "endpoint in prod this way. Restore them, or regenerate the manifest "
        "with scripts/regen_config_surface.py if the removal is deliberate."
    )


@pytest.mark.parametrize(
    "critical",
    [
        # Named explicitly because their loss is silent at import and fatal at
        # runtime — the route/handler reads the attribute directly.
        "ONBOARDING_RESET_ENABLED",
        "TRIAL_CREDIT_USD",
        "TRIAL_GLOBAL_DAILY_USD",
        "SEMANTIC_TOOL_ROUTING",
        "TOOL_ROUTING_ENUM_CAP",
    ],
)
def test_critical_settings_resolve(critical):
    from config import config

    assert hasattr(config, critical), (
        f"{critical} is missing from Config — a route or handler reads it "
        "directly and will raise AttributeError (HTTP 500) at runtime."
    )


# --------------------------------------------------------------------------- #
# NON-BLOCKING — new settings are reported, never failed
# --------------------------------------------------------------------------- #


def test_report_settings_absent_from_manifest_without_failing(recwarn):
    """Nag, don't block. Adding a setting must stay frictionless."""
    new = sorted(_live_names() - set(_manifest_names()))
    if new:
        import warnings

        warnings.warn(
            f"{len(new)} setting(s) not yet in config-surface.json: {new}. "
            "Run scripts/regen_config_surface.py to bring them under the "
            "deletion guard. This does NOT fail the build.",
            UserWarning,
            stacklevel=2,
        )
    assert True  # informational only, by design
