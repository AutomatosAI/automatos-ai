"""PRD-222 W2·S1b (US-024) — exposure profiles: nav + tool-surface + marketplace.

Pure tests: the exposure block derived from PLAN_TIERS per tier, a config
override flipping a nav key with no code change (config-driven proof), and the
tool-surface family filter — basic drops the CodeGraph first-class schemas and
prunes the dispatcher enum of nl2sql/team actions, business is untouched.

The two paths fail open DIFFERENTLY, and that asymmetry is deliberate (RVW-1):
the UI's ``exposure_for_plan`` falls back to the entry tier for an unknown plan
(the client always has a profile), while the tool path (``filter_tools_by_plan``
and the ``_apply_tier_exposure`` seam) returns the FULL surface for an unknown
plan — a stale plan string must never HIDE a paid tool.
"""
from __future__ import annotations

import copy

from config import TOOL_FAMILIES, load_plan_tiers
from services import plan_tiers as pt


# --------------------------------------------------------------------------- #
# Exposure block per tier
# --------------------------------------------------------------------------- #


def test_exposure_basic_hides_gated_nav_and_families():
    exp = pt.exposure_for_plan("basic")
    assert exp["plan"] == "basic"
    assert exp["display_name"] == "Basic"
    assert exp["display_price_usd"] == 19
    assert exp["marketplace_depth"] == 1
    # basic: nl2sql + team OFF ⇒ both gated nav items hidden.
    assert exp["nav"] == {"analytics": False, "team": False}
    assert exp["families"]["codegraph"] is False
    assert exp["families"]["voice"] is False


def test_exposure_pro_shows_analytics_and_team_hides_voice():
    exp = pt.exposure_for_plan("pro")
    assert exp["nav"] == {"analytics": True, "team": True}
    assert exp["families"]["codegraph"] is True
    assert exp["families"]["voice"] is False
    assert exp["marketplace_depth"] == 2


def test_exposure_business_shows_everything():
    exp = pt.exposure_for_plan("business")
    assert exp["nav"] == {"analytics": True, "team": True}
    assert exp["families"] == {"codegraph": True, "nl2sql": True, "team": True, "voice": True}
    assert exp["marketplace_depth"] == 3


def test_exposure_unknown_plan_falls_back_to_entry_tier():
    exp = pt.exposure_for_plan("nonexistent")
    # Falls back to basic's profile so the UI is never left without one.
    assert exp["nav"] == {"analytics": False, "team": False}
    assert exp["display_name"] == "Basic"


def test_config_override_flips_a_nav_key_without_code_change():
    # Turn nl2sql ON for basic via a config/env override → the analytics nav key
    # flips to visible with no code change (US-024 AC5).
    override = load_plan_tiers(env_override='{"basic": {"families": {"nl2sql": true}}}')
    exp = pt.exposure_for_plan("basic", tiers=override)
    assert exp["nav"]["analytics"] is True
    # The module default is unaffected.
    assert pt.exposure_for_plan("basic")["nav"]["analytics"] is False


# --------------------------------------------------------------------------- #
# Tool-surface family filter
# --------------------------------------------------------------------------- #


def _surface():
    """A representative platform surface: the platform_execute dispatcher (enum
    of action names across families) + first-class promoted schemas."""
    dispatcher = {
        "type": "function",
        "function": {
            "name": "platform_execute",
            "parameters": {
                "properties": {
                    "action": {
                        "enum": [
                            "platform_create_agent",       # core
                            "platform_query_data",         # nl2sql
                            "platform_get_llm_usage",      # nl2sql
                            "platform_list_members",       # team
                            "platform_invite_member",      # team
                            "platform_get_activity_feed",  # core (Command Center — NOT gated)
                        ]
                    }
                }
            },
        },
    }
    codegraph = {"type": "function", "function": {"name": "platform_codegraph_search"}}
    core_tool = {"type": "function", "function": {"name": "composio_execute"}}
    return [dispatcher, codegraph, core_tool]


def test_basic_drops_codegraph_schema_and_prunes_enum():
    trimmed = pt.filter_tools_by_plan(_surface(), "basic")
    names = [t["function"]["name"] for t in trimmed]
    # The CodeGraph first-class schema is dropped; core tools + dispatcher stay.
    assert "platform_codegraph_search" not in names
    assert "composio_execute" in names
    assert "platform_execute" in names
    # The dispatcher enum is pruned of nl2sql + team actions; core + Command
    # Center (activity_feed) survive.
    enum = [t for t in trimmed if t["function"]["name"] == "platform_execute"][0][
        "function"
    ]["parameters"]["properties"]["action"]["enum"]
    assert enum == ["platform_create_agent", "platform_get_activity_feed"]


def test_business_surface_is_unchanged():
    surface = _surface()
    trimmed = pt.filter_tools_by_plan(surface, "business")
    assert [t["function"]["name"] for t in trimmed] == [
        t["function"]["name"] for t in surface
    ]
    # Every family enabled ⇒ fast path, full enum retained.
    enum = [t for t in trimmed if t["function"]["name"] == "platform_execute"][0][
        "function"
    ]["parameters"]["properties"]["action"]["enum"]
    assert len(enum) == 6


def test_filter_does_not_mutate_input_schemas():
    surface = _surface()
    original = copy.deepcopy(surface)
    pt.filter_tools_by_plan(surface, "basic")
    assert surface == original  # rebuild-don't-mutate: inputs untouched


def test_pro_keeps_codegraph_and_analytics_drops_nothing_gated_off():
    # pro has codegraph + nl2sql + team ON, voice OFF (no voice tools exist), so
    # the representative surface is unchanged.
    trimmed = pt.filter_tools_by_plan(_surface(), "pro")
    assert [t["function"]["name"] for t in trimmed] == [
        t["function"]["name"] for t in _surface()
    ]


def test_tool_families_map_covers_all_analytics_and_team_names():
    # Guard against a gated tool leaking to basic because it is missing from the
    # config family map: every analytics/team action name the surface can carry
    # must resolve to a family.
    for name in ("platform_query_data", "platform_get_llm_usage", "platform_get_bottlenecks"):
        assert pt._tool_family(name, TOOL_FAMILIES) == "nl2sql"
    for name in ("platform_list_members", "platform_remove_member"):
        assert pt._tool_family(name, TOOL_FAMILIES) == "team"
    assert pt._tool_family("platform_codegraph_search", TOOL_FAMILIES) == "codegraph"
    # Command Center + generic tools are CORE (no family) — never gated.
    assert pt._tool_family("platform_get_activity_feed", TOOL_FAMILIES) is None
    assert pt._tool_family("composio_execute", TOOL_FAMILIES) is None


# --------------------------------------------------------------------------- #
# Unresolvable plan — TRUE fail-open on the tool path (RVW-1)
# --------------------------------------------------------------------------- #


def test_get_tier_distinguishes_unknown_from_known():
    # The renamed-away entry tier is unresolvable; 'basic' is a real trimming tier.
    assert pt.get_tier("starter") is None
    assert pt.get_tier("basic") is not None


def test_unknown_plan_tool_surface_fails_open_full_not_basic():
    # filter_tools_by_plan distinguishes an UNKNOWN plan from 'basic': a plan not
    # in PLAN_TIERS (a stale/renamed 'starter' row) returns the surface UNCHANGED
    # — the full surface, NOT the silent entry-tier trim exposure_for_plan uses
    # for the UI. A lookup fault must never hide a paid tool (RVW-1).
    surface = _surface()
    trimmed = pt.filter_tools_by_plan(surface, "starter")
    assert [t["function"]["name"] for t in trimmed] == [
        t["function"]["name"] for t in surface
    ]
    enum = [t for t in trimmed if t["function"]["name"] == "platform_execute"][0][
        "function"
    ]["parameters"]["properties"]["action"]["enum"]
    assert len(enum) == 6  # full enum retained — no basic-profile pruning


def test_unknown_plan_does_not_mutate_input():
    surface = _surface()
    original = copy.deepcopy(surface)
    pt.filter_tools_by_plan(surface, "starter")
    assert surface == original  # fail-open returns a new list, inputs untouched


# --------------------------------------------------------------------------- #
# Direct coverage of the _apply_tier_exposure seam (had none before — RVW-1)
# --------------------------------------------------------------------------- #


class _FakeQuery:
    def __init__(self, ws):
        self._ws = ws

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._ws


class _FakeSession:
    """Minimal stand-in: query(Workspace).filter(...).first() → the fake row."""

    def __init__(self, ws):
        self._ws = ws

    def query(self, *a, **k):
        return _FakeQuery(self._ws)


class _FakeWorkspace:
    def __init__(self, plan):
        self.plan = plan


def test_apply_tier_exposure_unknown_plan_keeps_full_surface():
    # The seam the finding flagged with zero direct coverage: a workspace on a
    # stale 'starter' plan (no longer in PLAN_TIERS) keeps the FULL surface —
    # true fail-open, not a silent strip to basic.
    from modules.tools.tool_router import _apply_tier_exposure

    surface = _surface()
    out = _apply_tier_exposure(_FakeSession(_FakeWorkspace("starter")), "ws-1", surface, "t")
    assert [t["function"]["name"] for t in out] == [
        t["function"]["name"] for t in surface
    ]
    enum = [t for t in out if t["function"]["name"] == "platform_execute"][0][
        "function"
    ]["parameters"]["properties"]["action"]["enum"]
    assert len(enum) == 6  # nothing pruned


def test_apply_tier_exposure_basic_still_trims_at_the_seam():
    # A KNOWN gated tier still trims through the seam — the fail-open guard did
    # not disable real gating.
    from modules.tools.tool_router import _apply_tier_exposure

    out = _apply_tier_exposure(_FakeSession(_FakeWorkspace("basic")), "ws-1", _surface(), "t")
    assert "platform_codegraph_search" not in [t["function"]["name"] for t in out]


def test_apply_tier_exposure_no_workspace_id_keeps_full_surface():
    from modules.tools.tool_router import _apply_tier_exposure

    surface = _surface()
    out = _apply_tier_exposure(_FakeSession(_FakeWorkspace("basic")), None, surface, "t")
    assert out is surface  # early-out: nothing to resolve
