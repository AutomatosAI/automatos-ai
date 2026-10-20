"""PRD-222 W2·S1b (US-024) — exposure profiles: nav + tool-surface + marketplace.

Pure tests: the exposure block derived from PLAN_TIERS per tier, a config
override flipping a nav key with no code change (config-driven proof), and the
tool-surface family filter — a tier with codegraph/nl2sql/team OFF drops the
CodeGraph first-class schemas and prunes the dispatcher enum of nl2sql/team
actions, business is untouched. PRD-265 US-001 made ``basic`` (Own keys)
carry EXACTLY Pro's (permissive) families, so the trimming-MECHANISM tests
below inject a restricted override (``_RESTRICTED_BASIC_TIERS``) to keep
direct coverage of that machinery now that basic itself trims nothing live.

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

# PRD-265 US-001 made basic (Own keys) non-restrictive by default — it now
# carries EXACTLY Pro's families. The mechanism tests below exist to prove
# the trimming/override machinery itself, not basic's live numbers, so they
# inject this RESTRICTED override (the tier's OLD shape) to keep direct
# coverage of that machinery now that basic itself trims nothing out of the box.
_RESTRICTED_BASIC_TIERS = load_plan_tiers(
    env_override='{"basic": {"families": {"codegraph": false, "nl2sql": false, "team": false}}}'
)


# --------------------------------------------------------------------------- #
# Exposure block per tier
# --------------------------------------------------------------------------- #


def test_exposure_basic_shows_pro_profile_with_beta_label():
    # PRD-265 US-001: Own keys (basic) carries EXACTLY Pro's limits/families —
    # it is no longer a trimming tier. price_label is "beta" while
    # config.BILLING_OPEN is False (the default in tests).
    exp = pt.exposure_for_plan("basic")
    assert exp["plan"] == "basic"
    assert exp["display_name"] == "Own keys"
    assert exp["display_price_usd"] == 9
    assert exp["price_label"] == "beta"
    assert exp["marketplace_depth"] == 2
    # basic: nl2sql + team ON ⇒ both nav items visible, same as Pro.
    assert exp["nav"] == {"analytics": True, "team": True}
    assert exp["families"]["codegraph"] is True
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
    # PRD-265: that profile is now the permissive, Pro-equivalent one.
    assert exp["nav"] == {"analytics": True, "team": True}
    assert exp["display_name"] == "Own keys"


def test_config_override_flips_a_nav_key_without_code_change():
    # basic now defaults to Pro's permissive families, so this proof starts
    # from an injected RESTRICTED basic (the mechanism fixture above, with
    # analytics/team off) and flips just nl2sql ON — the analytics nav key
    # flips to visible with no code change (US-024 AC5), keeping direct
    # coverage of the override mechanism now that basic itself isn't restrictive.
    assert pt.exposure_for_plan("basic", tiers=_RESTRICTED_BASIC_TIERS)["nav"]["analytics"] is False
    flipped = load_plan_tiers(
        env_override='{"basic": {"families": {"codegraph": false, "nl2sql": true, "team": false}}}'
    )
    assert pt.exposure_for_plan("basic", tiers=flipped)["nav"]["analytics"] is True
    # The module default (permissive) is unaffected by either override.
    assert pt.exposure_for_plan("basic")["nav"]["analytics"] is True


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
    # PRD-265: basic's live default no longer trims anything, so this proof
    # of the TRIMMING MECHANISM injects the restricted override fixture.
    trimmed = pt.filter_tools_by_plan(_surface(), "basic", tiers=_RESTRICTED_BASIC_TIERS)
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
    # Restricted override again — the fast (no-op) path on a permissive basic
    # would prove nothing about rebuild-don't-mutate on the trimming branch.
    surface = _surface()
    original = copy.deepcopy(surface)
    pt.filter_tools_by_plan(surface, "basic", tiers=_RESTRICTED_BASIC_TIERS)
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
    # The renamed-away entry tier is unresolvable; 'basic' is a real, known tier.
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


def test_apply_tier_exposure_basic_still_trims_at_the_seam(monkeypatch):
    # A KNOWN gated tier still trims through the seam — the fail-open guard did
    # not disable real gating. basic's live default no longer trims anything
    # (PRD-265), so patch the LIVE config.PLAN_TIERS to the restricted fixture:
    # _apply_tier_exposure takes no ``tiers`` param — it reads ``config.PLAN_TIERS``
    # fresh on every call (``services.plan_tiers._tiers`` does
    # ``from config import PLAN_TIERS`` inside the function body), so patching
    # the module attribute is the correct seam to flip for this proof.
    from modules.tools.tool_router import _apply_tier_exposure
    import config as config_module

    monkeypatch.setattr(config_module, "PLAN_TIERS", _RESTRICTED_BASIC_TIERS)
    out = _apply_tier_exposure(_FakeSession(_FakeWorkspace("basic")), "ws-1", _surface(), "t")
    assert "platform_codegraph_search" not in [t["function"]["name"] for t in out]


def test_apply_tier_exposure_no_workspace_id_keeps_full_surface():
    from modules.tools.tool_router import _apply_tier_exposure

    surface = _surface()
    out = _apply_tier_exposure(_FakeSession(_FakeWorkspace("basic")), None, surface, "t")
    assert out is surface  # early-out: nothing to resolve


# --------------------------------------------------------------------------- #
# RVW-5 — a KNOWN family-less tier ('enterprise') is UNRESTRICTED in BOTH layers
# (nav shows all + full tool surface), so the two enforcement layers agree. Was
# a live divergence (nav hid analytics/team while tools stayed full) latent
# until enterprise is promoted assignable without a families key.
# --------------------------------------------------------------------------- #


def _enum_of(surface):
    return [t for t in surface if t["function"]["name"] == "platform_execute"][0][
        "function"
    ]["parameters"]["properties"]["action"]["enum"]


def test_exposure_enterprise_family_less_is_unrestricted():
    # enterprise declares NO families key → unrestricted nav (not the old
    # all-hidden default that disagreed with the full tool surface).
    exp = pt.exposure_for_plan("enterprise")
    assert exp["display_name"] == "Enterprise"
    assert exp["nav"] == {"analytics": True, "team": True}


def test_enterprise_tool_surface_and_nav_agree():
    # AC2: the tool surface and exposure_for_plan('enterprise').nav AGREE — a nav
    # item is visible IFF the tools backing it are present. A KNOWN tier with no
    # families is unrestricted on both paths (RVW-5), never nav-hidden/tools-full.
    nav = pt.exposure_for_plan("enterprise")["nav"]
    trimmed = pt.filter_tools_by_plan(_surface(), "enterprise")
    # full surface — nothing dropped, full enum retained
    assert [t["function"]["name"] for t in trimmed] == [
        t["function"]["name"] for t in _surface()
    ]
    enum = _enum_of(trimmed)
    assert len(enum) == 6
    # the invariant: nav visibility matches tool availability on both families.
    assert nav["analytics"] is ("platform_query_data" in enum)      # nl2sql
    assert nav["team"] is ("platform_list_members" in enum)         # team
    assert nav["analytics"] is True and nav["team"] is True          # both shown


def test_enterprise_filter_does_not_mutate_input():
    surface = _surface()
    original = copy.deepcopy(surface)
    pt.filter_tools_by_plan(surface, "enterprise")
    assert surface == original  # fail-open returns a new list, inputs untouched


def test_apply_tier_exposure_enterprise_keeps_full_surface():
    # The seam for a KNOWN family-less tier: enterprise keeps the FULL surface
    # (unrestricted), consistent with its nav (RVW-5). Zero coverage before.
    from modules.tools.tool_router import _apply_tier_exposure

    surface = _surface()
    out = _apply_tier_exposure(_FakeSession(_FakeWorkspace("enterprise")), "ws-1", surface, "t")
    assert [t["function"]["name"] for t in out] == [
        t["function"]["name"] for t in surface
    ]
    assert len(_enum_of(out)) == 6  # nothing pruned


def test_promoted_enterprise_without_families_stays_consistent():
    # The exact reachability the finding named: ops flips enterprise assignable
    # via env WITHOUT adding a families key. Both layers must still AGREE
    # (unrestricted) — no nav-hides-while-tools-show divergence.
    override = load_plan_tiers(env_override='{"enterprise": {"assignable": true, "coming_soon": false}}')
    assert pt.is_assignable("enterprise", override) is True  # now assignable
    nav = pt.exposure_for_plan("enterprise", tiers=override)["nav"]
    trimmed = pt.filter_tools_by_plan(_surface(), "enterprise", tiers=override)
    assert nav == {"analytics": True, "team": True}
    assert len(_enum_of(trimmed)) == 6  # full surface — agrees with nav
