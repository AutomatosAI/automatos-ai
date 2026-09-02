"""
PRD-232 US-014 — promotion-as-prior (§6.2 LOCKED).
==================================================

C5: 47 promoted actions attached UNCONDITIONALLY as first-class OpenAI schemas
every full-path turn (~4-7k tokens, zero query-relevance gating). US-014 turns
promotion into a PRIOR: a config-listed pin set (~14, from PRD-122's original
promoted list + platform_find_tools) always attaches first-class; every OTHER
promoted action loses unconditional attachment — its flag becomes a ranking BOOST,
and it attaches first-class ONLY when it ranks into the query surface, otherwise it
lives in the platform_execute dispatcher enum like any action.

These tests are hermetic: action_registry.py is leaf-loaded directly (no DB, no
config-of-actions, no torch), and the registry is a realistically-SIZED synthetic
set so the token measurement is meaningful. The boost proof lives in the sibling
test_action_semantic_index.py (test_promoted_action_boosted_over_equal_cosine_unpromoted),
which owns the real cosine ranker. Tier fail-closed is proven here AND left green in
test_prd143_su_registry (defaults untouched).

The router-reads-config check is a source grep (no heavy import of tool_router).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()
_ORCH = _THIS.parents[1]
_AR_PATH = _ORCH / "modules" / "tools" / "discovery" / "action_registry.py"

# Leaf-load action_registry.py under a unique name (no package import → no torch/DB).
_spec = importlib.util.spec_from_file_location("action_registry_prd232_us014", _AR_PATH)
_ar_mod = importlib.util.module_from_spec(_spec)
sys.modules["action_registry_prd232_us014"] = _ar_mod
_spec.loader.exec_module(_ar_mod)
ActionDefinition = _ar_mod.ActionDefinition
ActionRegistry = _ar_mod.ActionRegistry

# The pin set as the router will read it — straight from config (the ONE source).
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
from config import config  # noqa: E402

PINS = {p.strip() for p in config.TOOL_ROUTING_PROMOTION_PINS.split(",") if p.strip()}


# ---------------------------------------------------------------------------
# A realistically-sized synthetic registry (schemas near real promoted sizes)
# ---------------------------------------------------------------------------

_LONG_DESC = (
    "Operate on the platform resource described by this action. Provide the required "
    "parameters exactly; the handler validates them and returns a structured result "
    "the caller renders. Prefer this first-class schema over the generic dispatcher "
    "when the intent clearly matches."
)


def _params():
    """A non-trivial parameter block so a schema's serialized size is realistic."""
    return {
        "type": "object",
        "properties": {
            "target_id": {"type": "integer", "description": "Identifier of the target resource to act on."},
            "mode": {"type": "string", "enum": ["read", "write", "sync"], "description": "The operation mode to run in."},
            "options": {"type": "object", "description": "Optional keyword settings passed through to the handler."},
        },
        "required": ["target_id"],
    }


def _act(name, *, promoted=False, admin_only=False, super_admin_only=False):
    return ActionDefinition(
        name=name,
        description=_LONG_DESC,
        category="agents",
        parameters=_params(),
        admin_only=admin_only,
        super_admin_only=super_admin_only,
        promoted=promoted,
    )


def _registry():
    reg = ActionRegistry()
    reg._initialized = True  # bypass the lazy live-loader
    # The config pins, as promoted actions (so first_class_names=PINS matches them).
    for name in sorted(PINS):
        reg.register(_act(name, promoted=True))
    # ~31 OTHER promoted actions (so the legacy surface is ~45, like production).
    for i in range(31):
        reg.register(_act(f"platform_extra_promoted_{i:02d}", promoted=True))
    # A few non-promoted + one admin-promoted + one su-promoted for tier tests.
    reg.register(_act("platform_plain_thing", promoted=False))
    reg.register(_act("platform_admin_promoted", promoted=True, admin_only=True))
    reg.register(_act("platform_su_promoted", promoted=True, super_admin_only=True))
    return reg


def _tok(schemas) -> int:
    """Serialized-token proxy (chars/4) — the metric the PR body records."""
    return len(json.dumps(schemas)) // 4


def _enum(schema):
    return schema["function"]["parameters"]["properties"]["action"].get("enum", [])


def _fc_names(schemas):
    return {s["function"]["name"] for s in schemas}


# ===========================================================================
# AC1 — pins live in config; the router reads them from there (no hardcoded list)
# ===========================================================================


def test_pins_live_in_config_and_keep_find_tools():
    assert "platform_find_tools" in PINS, "the when-required discovery seam MUST stay pinned"
    assert 10 <= len(PINS) <= 16, f"pin set should be ~10-14, got {len(PINS)}"
    # Traceable to PRD-122's original promoted list (a representative spread).
    for name in (
        "platform_list_agents", "platform_get_agent", "platform_install_skill",
        "platform_search_memory", "platform_store_memory", "platform_get_activity_feed",
    ):
        assert name in PINS


def test_router_reads_pins_from_config_no_hardcoded_list():
    """The pin set lives in config.py; the router reads it via config, never as a
    literal list in the router (CLAUDE.md §4 — no hardcoded values)."""
    router_src = (_ORCH / "modules" / "tools" / "tool_router.py").read_text()
    assert 'getattr(config, "TOOL_ROUTING_PROMOTION_PINS"' in router_src, "router must read pins from config"
    # No inline CSV of pin names smuggled into the router.
    assert "platform_find_tools,platform_list_agents" not in router_src
    config_src = (_ORCH / "config.py").read_text()
    assert "TOOL_ROUTING_PROMOTION_PINS" in config_src


# ===========================================================================
# AC2 — token measurement: first-class + dispatcher payload before vs after
# ===========================================================================


def test_tool_payload_drops_at_least_2k_tokens():
    reg = _registry()
    # BEFORE (legacy PRD-122): every promoted action first-class + dispatcher
    # excludes all promoted.
    before = reg.to_first_class_schemas() + [reg.to_dispatcher_schema(exclude_promoted=True)]
    # AFTER (US-014, no query ranked): only the pins first-class; the dispatcher
    # enum now carries the non-pin promoted actions (exclude_promoted=False).
    after = (
        reg.to_first_class_schemas(first_class_names=PINS)
        + [reg.to_dispatcher_schema(exclude_promoted=False, exclude_names=PINS)]
    )
    tb, ta = _tok(before), _tok(after)
    print(f"[US-014] tool payload tokens: before={tb} after={ta} NET reduction={tb - ta}")
    # First-class schema count collapses from the full promoted surface to the pins.
    assert len(reg.to_first_class_schemas()) >= 40
    assert len(reg.to_first_class_schemas(first_class_names=PINS)) == len(PINS)
    assert tb - ta >= 2000, f"expected >=2k token reduction, got {tb - ta}"


# ===========================================================================
# AC3 — reachability: unpinned+unranked promoted absent from first-class, in enum;
#       a ranked promoted attaches first-class
# ===========================================================================


def test_unpinned_unranked_promoted_absent_from_first_class_but_in_enum():
    reg = _registry()
    victim = "platform_extra_promoted_00"  # promoted, NOT a pin, NOT ranked
    fc = _fc_names(reg.to_first_class_schemas(first_class_names=PINS))
    assert victim not in fc, "a non-pin promoted action must NOT attach first-class when unranked"
    enum = _enum(reg.to_dispatcher_schema(exclude_promoted=False, exclude_names=PINS))
    assert victim in enum, "an unattached promoted action must stay reachable via the dispatcher enum"
    # A pin, by contrast, IS first-class and NOT duplicated in the enum.
    pin = sorted(PINS)[0]
    assert pin in fc
    assert pin not in enum


def test_narrowed_steady_state_unranked_promoted_reachable_only_via_find_tools():
    """P232-RVW-1 AC4: the reachability contract in the NARROWED steady state —
    the case allowed_names=None (test above) cannot exercise.

    With a realistic narrowed allow-list that EXCLUDES a promoted action, that
    action is absent from BOTH the first-class schemas AND the narrowed dispatcher
    enum this turn. It is not stranded: it stays reachable via platform_find_tools
    (a config pin, first-class every turn) — the seam the LLM uses to pull in any
    action the ranker did not surface. Proves 'no action becomes unreachable' for
    the narrowed path, not just the un-narrowed full enum."""
    reg = _registry()
    victim = "platform_extra_promoted_00"        # promoted, NOT a pin
    ranked_other = "platform_extra_promoted_07"  # a DIFFERENT promoted action that ranked in
    # A ranked top-K for some unrelated intent: one plain action + one promoted
    # action ranked in. The victim did NOT rank.
    allowed = ["platform_plain_thing", ranked_other]
    assert victim not in allowed

    # The loader's first_class = config pins ∪ (promoted ∩ allowed_names).
    first_class = PINS | {ranked_other}

    fc = _fc_names(reg.to_first_class_schemas(first_class_names=first_class))
    assert victim not in fc, "an unranked non-pin promoted action must NOT be first-class"

    enum = _enum(reg.to_dispatcher_schema(
        exclude_promoted=False,
        exclude_names=first_class,
        allowed_names=allowed,
    ))
    assert victim not in enum, (
        "in the narrowed steady state an unranked promoted action is NOT an enum "
        "member — it is reachable only via platform_find_tools this turn"
    )
    # find_tools is the discovery seam that keeps the victim reachable.
    assert "platform_find_tools" in PINS
    assert "platform_find_tools" in fc
    # Sanity: the ranked promoted DID attach first-class and left the enum.
    assert ranked_other in fc
    assert ranked_other not in enum


def test_ranked_promoted_attaches_first_class_and_leaves_the_enum():
    reg = _registry()
    ranked = "platform_extra_promoted_07"  # simulate it ranking into the surface
    first_class = PINS | {ranked}
    fc = _fc_names(reg.to_first_class_schemas(first_class_names=first_class))
    assert ranked in fc, "a promoted action that ranked in must attach first-class"
    enum = _enum(reg.to_dispatcher_schema(exclude_promoted=False, exclude_names=first_class))
    assert ranked not in enum, "a first-class action must not be duplicated in the enum"


def test_narrowed_enum_excludes_first_class_keeps_non_promoted():
    """With a narrowed allowed_names (incl a ranked promoted + a plain action), the
    enum keeps the plain action but drops the first-class (promoted) one."""
    reg = _registry()
    ranked_promoted = "platform_extra_promoted_03"
    first_class = PINS | {ranked_promoted}
    schema = reg.to_dispatcher_schema(
        exclude_promoted=False,
        exclude_names=first_class,
        allowed_names=[ranked_promoted, "platform_plain_thing"],
    )
    enum = _enum(schema)
    assert enum == ["platform_plain_thing"]  # promoted-ranked went first-class, plain stays in enum


# ===========================================================================
# AC4 — tier gating stays fail-closed, even when a gated name is in first_class
# ===========================================================================


def test_super_admin_never_first_class_even_when_named():
    reg = _registry()
    su = "platform_su_promoted"
    # Maliciously name the su action first-class → the su filter still drops it.
    fc = _fc_names(reg.to_first_class_schemas(first_class_names=PINS | {su}, include_super_admin=False))
    assert su not in fc, "super_admin_only must never attach first-class for an operator"
    # And it never leaks into the operator dispatcher enum either.
    enum = _enum(reg.to_dispatcher_schema(
        exclude_promoted=False, exclude_names=PINS | {su}, include_super_admin=False,
    ))
    assert su not in enum


def test_admin_promoted_excluded_for_non_admin_caller():
    reg = _registry()
    admin_action = "platform_admin_promoted"
    fc = _fc_names(reg.to_first_class_schemas(
        first_class_names=PINS | {admin_action}, exclude_admin=True,
    ))
    assert admin_action not in fc, "admin_only must not attach first-class for a non-admin"
    enum = _enum(reg.to_dispatcher_schema(
        exclude_promoted=False, exclude_names=PINS | {admin_action}, exclude_admin=True,
    ))
    assert admin_action not in enum  # excluded from the enum too (role gate first)


def test_closed_pins_fallback_keeps_minimal_enum_not_full():
    """Regression guard for the closed-pins interaction: the fallback's curated pin
    list is mostly PROMOTED config pins; excluding them from the enum (as US-014
    does normally) would collapse the narrowed enum into the full-enum defensive
    fallback — the opposite of the mode's minimal-surface intent. This mirrors the
    loader's ``from_pins`` branch (exclude_names = first_class − allowed_names)."""
    reg = _registry()
    # A closed-pins fallback surface: a config pin + a non-pin promoted extra.
    allowed = ["platform_find_tools", "platform_extra_promoted_00"]
    promoted = {a.name for a in reg.get_all() if a.promoted}
    ranked_promoted = {n for n in allowed if n in promoted}
    first_class = (PINS & promoted) | ranked_promoted
    enum_exclude = first_class - set(allowed)  # loader's from_pins computation
    enum = _enum(reg.to_dispatcher_schema(
        exclude_promoted=False, exclude_names=enum_exclude,
        allowed_names=allowed, allow_promoted_in_allowlist=True,
    ))
    assert sorted(enum) == ["platform_extra_promoted_00", "platform_find_tools"]
    assert len(enum) == 2, "closed-pins enum must stay minimal, not blow up to the full set"


def test_catalog_documents_non_pin_promoted_and_excludes_pins():
    """US-014 reachability completeness (the 'reachable like any action' AC): the
    UNNARROWED platform_execute catalog (PlatformActionsSection._build) documents
    every non-pin promoted action — they now live in the dispatcher enum, so without
    documentation the model would see a bare enum name with no params. The config
    pins are excluded (self-documented by their own first-class schemas). This tests
    the registry mechanism the _build fix uses: build_prompt_summary(exclude_promoted
    =False, exclude_names=pins)."""
    reg = _registry()
    summary = reg.build_prompt_summary(
        exclude_promoted=False, exclude_admin=True, exclude_names=sorted(PINS),
    )
    # a non-pin promoted action IS documented (it is in the enum, needs its params)
    assert "platform_extra_promoted_00" in summary
    # a config pin is NOT duplicated here (it attaches as its own first-class schema)
    assert "platform_find_tools" not in summary
    # non-promoted actions are documented exactly as before
    assert "platform_plain_thing" in summary
    # the unnarrowed catalog set == the unnarrowed dispatcher enum set (consistency)
    enum = set(_enum(reg.to_dispatcher_schema(exclude_promoted=False, exclude_names=PINS, exclude_admin=True)))
    documented = {ln.split("`")[1] for ln in summary.splitlines() if "`platform_" in ln or "`workspace_" in ln}
    # every enum action is documented in the catalog (no bare undocumented enum names)
    assert enum <= documented, f"enum actions missing from catalog: {enum - documented}"


def test_legacy_default_unchanged_when_no_first_class_names():
    """Back-compat: with no first_class_names / exclude_names, behaviour is exactly
    the pre-US-014 contract — every eligible promoted action first-class, none in the
    enum. Guards the tier suites (test_prd143_su_registry) that rely on the default."""
    reg = _registry()
    legacy_fc = _fc_names(reg.to_first_class_schemas())
    assert "platform_extra_promoted_00" in legacy_fc  # all promoted first-class by default
    legacy_enum = _enum(reg.to_dispatcher_schema())  # exclude_promoted=True default
    assert "platform_extra_promoted_00" not in legacy_enum
    assert "platform_plain_thing" in legacy_enum
