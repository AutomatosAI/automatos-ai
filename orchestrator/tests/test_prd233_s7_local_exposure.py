"""PRD-233 S7 (backend half) — the local edition is never plan-gated.

The entrypoint seeds the single local workspace with ``plan = NULL``; falling
back to the entry tier hid Analytics and put plan chips on marketplace items
in an edition where nothing is gated (owner rule: paid tiers gate
organisational features and hosting, never product capability).
"""
from __future__ import annotations

import re
from pathlib import Path

from services.plan_tiers import (
    LOCAL_EDITION_PLAN,
    _tiers,
    exposure_for_local_edition,
    exposure_for_plan,
)

_WORKSPACES_SRC = (Path(__file__).resolve().parents[1] / "api" / "workspaces.py").read_text()


def test_local_edition_exposure_is_unrestricted():
    e = exposure_for_local_edition()
    assert e["plan"] == LOCAL_EDITION_PLAN
    assert e["families"] == {}
    assert e["nav"] and all(e["nav"].values()), e["nav"]
    assert e["display_price_usd"] is None and e["price_label"] is None


def test_local_marketplace_depth_is_the_deepest_tier():
    e = exposure_for_local_edition()
    deepest = max(int(t.get("marketplace_depth", 1) or 1) for t in _tiers().values() if isinstance(t, dict))
    assert e["marketplace_depth"] == deepest
    assert e["marketplace_depth"] >= exposure_for_plan("business")["marketplace_depth"]


def test_saas_tiers_unchanged():
    # The entry tier still restricts (nav.team off) — S7 only adds a profile.
    basic = exposure_for_plan("basic")
    assert basic["plan"] == "basic"
    assert basic["nav"].get("team") is False
    assert exposure_for_plan("nonsense")["plan"] == "nonsense"  # unknown ⇒ entry-tier profile, label kept


def test_current_workspace_branches_on_edition_not_role():
    # One seam: api/workspaces.py picks the profile by AUTH_EDITION, and the
    # saas branch is the exact pre-existing call.
    assert re.search(r'exposure_for_local_edition\(\)\s*\n\s*if config\.AUTH_EDITION == "local"', _WORKSPACES_SRC)
    assert 'exposure_for_plan(workspace.plan or "basic")' in _WORKSPACES_SRC
    assert _WORKSPACES_SRC.count("exposure_for_local_edition()") == 1
