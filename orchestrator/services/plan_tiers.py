"""PRD-222 W2·S1 (US-023) — plan-tier resolution + assignment.

The single service seam for the config-driven tier contract (``config.PLAN_TIERS``):
read which tiers are assignable, resolve a tier, derive the ``plan_limits`` a tier
implies, and assign a plan to a workspace. Reused by US-025's proposal-acceptance
tool path so that plan changes flow through ONE writer (FR-4 auditability).

Assignment writes the tier's limits into ``workspaces.plan_limits`` as a NEW dict
(PRD-220 rebuild-don't-mutate), MERGED onto whatever the workspace already carries
so unmanaged keys survive, under the keys the LIVE consumers already read:

  * ``max_members`` — the seat key. ``core/workspaces/invitations.py`` enforces it
    and the onboarding checklist gates the invite item on it, so the tier's
    ``seats`` lands HERE, not in a parallel ``seats`` key.
  * ``max_agents``  — documented concurrency cap (0 = unlimited).
  * ``budget``      — ``{window, max_cost_usd, source}``, read by
    ``modules/policy/budget.py`` (which ignores ``source``). A tier with a
    positive ``budget_usd`` writes a tier-OWNED ceiling (``source="tier"``);
    a tier with ``budget_usd = 0`` (business = custom / no ceiling) sets no
    budget key and :func:`assign_plan` CLEARS a tier-owned ceiling a prior tier
    wrote, so no stale ceiling survives an upgrade (RVW-4). An admin custom
    budget (written by ``policy.budget.set_budget``, which strips ``source``) is
    admin-owned and is NEVER cleared by a tier change.

``mission_concurrency`` / ``watcher_limit`` / ``marketplace_depth`` are stored too
(the tier's declared limits) but have no enforcement consumer yet — this wave adds
NO quota hardening. No billing anywhere: ``display_price_usd`` is a label only
(PRD §12 Q5). ``enterprise`` is coming-soon and is rejected by :func:`assign_plan`.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _tiers(tiers: Optional[dict] = None) -> dict:
    """The tier map to resolve against — an injected override (tests / callers
    that already loaded an env override) or the module ``config.PLAN_TIERS``."""
    if tiers is not None:
        return tiers
    from config import PLAN_TIERS

    return PLAN_TIERS


def get_tier(plan: str, tiers: Optional[dict] = None) -> Optional[dict]:
    """The tier config for ``plan`` (any tier, incl. non-assignable), or None."""
    return _tiers(tiers).get(plan)


def is_assignable(plan: str, tiers: Optional[dict] = None) -> bool:
    """True only for a tier a workspace can actually be put on — present,
    ``assignable``, and not a coming-soon placeholder (``enterprise``)."""
    tier = _tiers(tiers).get(plan)
    return bool(tier) and bool(tier.get("assignable")) and not tier.get("coming_soon")


def assignable_tiers(tiers: Optional[dict] = None) -> dict:
    """The subset of tiers a workspace can be assigned to (excludes enterprise)."""
    resolved = _tiers(tiers)
    return {name: t for name, t in resolved.items() if is_assignable(name, resolved)}


def plan_limits_for_tier(plan: str, tiers: Optional[dict] = None) -> dict:
    """The ``plan_limits`` fragment a tier implies, keyed for the LIVE consumers.

    Raises :class:`ValueError` for a non-assignable / unknown plan — the caller
    must never write limits for a tier that cannot be assigned.
    """
    resolved = _tiers(tiers)
    if not is_assignable(plan, resolved):
        raise ValueError(f"plan {plan!r} is not assignable")
    tier = resolved[plan]
    limits: dict[str, Any] = {
        "max_members": tier["seats"],
        "max_agents": tier["max_agents"],
        "mission_concurrency": tier["mission_concurrency"],
        "watcher_limit": tier["watcher_limit"],
        "marketplace_depth": tier["marketplace_depth"],
    }
    budget_usd = tier.get("budget_usd") or 0
    if budget_usd and budget_usd > 0:
        # A tier-OWNED ceiling (``source="tier"``): re-derived on every assignment
        # and cleared when the workspace moves to a no-ceiling tier (see
        # :func:`assign_plan`). An admin budget carries no such marker and is never
        # tier-cleared (RVW-4). ``budget_usd = 0`` ⇒ no key ⇒ custom / no ceiling.
        limits["budget"] = {
            "window": "month", "max_cost_usd": float(budget_usd), "source": "tier",
        }
    return limits


def assign_plan(
    db: Any, workspace: Any, plan: str, tiers: Optional[dict] = None, commit: bool = True
) -> dict:
    """Set ``workspace.plan`` and merge the tier's limits into ``plan_limits``.

    Rebuild-don't-mutate (PRD-220): a NEW ``plan_limits`` dict is built from the
    existing one (so keys this tier does not manage survive) with the tier's
    limits overlaid, then reassigned so SQLAlchemy marks the JSONB column dirty.
    ``db is None`` performs the in-memory assignment only — the escape hatch for
    pure-logic tests (mirrors ``services/onboarding_state._persist``).
    ``commit=False`` flushes the write but leaves the transaction OPEN so a caller
    that also stamps the ``plan_accepted`` funnel event lands BOTH in one commit
    (FR-4 atomicity — see ``handlers_onboarding.update_onboarding``). Returns the
    new ``plan_limits``. Raises :class:`ValueError` for a non-assignable plan.

    Budget provenance (RVW-4): moving to a no-ceiling tier (``business``) CLEARS a
    tier-owned ``budget`` a prior tier wrote, so no stale ceiling lingers on
    ``plan_limits`` (GET /budget would otherwise still show it, and an enforce
    stage would throttle at it). An admin custom budget — ``source != "tier"`` —
    is the customer's own explicit ceiling and is left untouched on any tier.
    """
    limits = plan_limits_for_tier(plan, tiers)  # validates assignability first
    new_limits = dict(workspace.plan_limits or {})
    new_limits.update(limits)
    # RVW-4: a 0-budget tier implies NO 'budget' key, so the merge above cannot
    # clear a ceiling a prior tier wrote. Drop a tier-OWNED budget so no stale
    # ceiling survives the upgrade; an admin custom budget (source != "tier") is
    # a deliberate ceiling — valid on any tier, incl. business — and is preserved.
    if "budget" not in limits:
        existing = new_limits.get("budget")
        if isinstance(existing, dict) and existing.get("source") == "tier":
            new_limits.pop("budget", None)
    workspace.plan = plan
    workspace.plan_limits = new_limits
    if db is not None:
        db.add(workspace)
        if commit:
            db.commit()
        else:
            db.flush()
    logger.info("[plan_tiers] assigned plan=%s to workspace=%s", plan, getattr(workspace, "id", "?"))
    return new_limits


# --------------------------------------------------------------------------- #
# US-024 — exposure profile (nav + families + marketplace depth) from the tier
# --------------------------------------------------------------------------- #


def enabled_families(plan: str, tiers: Optional[dict] = None) -> dict:
    """The capability families enabled for ``plan`` (unknown plan ⇒ entry tier)."""
    resolved = _tiers(tiers)
    tier = resolved.get(plan) or resolved.get("basic") or {}
    return dict(tier.get("families") or {})


def _nav_exposure(families: dict) -> dict:
    """Nav visibility derived from the tier's families. Only the surfaces that
    are ACTUAL top-level nav items are keyed here: ``analytics`` (the /analytics
    item, gated by the nl2sql family) and ``team`` (the /team item). CodeGraph is
    folded into Knowledge Base and Voice lives in chat — neither is a rail item,
    so they are gated by the tool surface (Auto) not by nav. A hidden item is
    simply absent from the rail; its route still resolves (D5 — hidden ≠ deleted).
    """
    return {
        "analytics": bool(families.get("nl2sql")),
        "team": bool(families.get("team")),
    }


def exposure_for_plan(plan: str, tiers: Optional[dict] = None) -> dict:
    """The exposure profile for a workspace on ``plan``, derived entirely from
    PLAN_TIERS: nav visibility, capability families, marketplace depth, and the
    tier's display info for the UI. Unknown plans fall back to the entry tier so
    the client is never left without a profile.
    """
    resolved = _tiers(tiers)
    tier = resolved.get(plan) or resolved.get("basic") or {}
    families = dict(tier.get("families") or {})
    return {
        "plan": plan,
        "display_name": tier.get("display_name"),
        "display_price_usd": tier.get("display_price_usd"),
        "price_label": tier.get("price_label"),
        "families": families,
        "marketplace_depth": tier.get("marketplace_depth", 1),
        "nav": _nav_exposure(families),
    }


# --------------------------------------------------------------------------- #
# US-024 — Auto's per-turn tool surface, trimmed to the tier's families
# --------------------------------------------------------------------------- #


def _tool_family(tool_name: str, family_map: dict) -> Optional[str]:
    """The family a platform tool name belongs to, or None (⇒ CORE, always on).
    Match is exact, or a prefix when the map entry ends in ``_``."""
    for family, patterns in (family_map or {}).items():
        for pat in patterns or []:
            if tool_name == pat or (pat.endswith("_") and tool_name.startswith(pat)):
                return family
    return None


def _prune_dispatcher(schema: dict, disabled: set, family_map: dict) -> dict:
    """A copy of the ``platform_execute`` dispatcher with disabled-family actions
    removed from its ``action.enum`` (rebuild, don't mutate). Returns the schema
    unchanged if it carries no prunable enum."""
    try:
        enum = schema["function"]["parameters"]["properties"]["action"].get("enum")
    except (KeyError, TypeError, AttributeError):
        return schema
    if not enum:
        return schema
    kept = [a for a in enum if _tool_family(a, family_map) not in disabled]
    if len(kept) == len(enum):
        return schema
    new_schema = copy.deepcopy(schema)
    new_schema["function"]["parameters"]["properties"]["action"]["enum"] = kept
    return new_schema


def filter_tools_by_plan(
    tools: list, plan: str, tiers: Optional[dict] = None, family_map: Optional[dict] = None
) -> list:
    """Trim Auto's per-turn tool surface to the workspace tier's families.

    Drops platform tool schemas whose family is disabled for ``plan`` (e.g. the
    9 first-class CodeGraph schemas for a basic workspace) and prunes the
    ``platform_execute`` dispatcher's ``action.enum`` of disabled-family actions.
    CORE tools (no family) and the dispatcher itself are always kept. Returns a
    NEW list and never mutates an input schema (the dispatcher is rebuilt with a
    pruned enum). When every family is enabled, returns the list unchanged (fast
    path). Pure — no I/O; the caller resolves ``plan`` from the workspace.

    FAIL-OPEN on an UNRESOLVABLE plan: a plan that is not a known tier — falsy,
    or a stale/renamed/stray string such as a legacy ``'starter'`` row — returns
    the surface UNCHANGED. Unlike the UI's :func:`exposure_for_plan`, which
    deliberately falls back to ``basic`` so the client always has a profile, a
    lookup fault on the tool path must never HIDE a tool: better to over-expose
    than to strip a paying tier's tools because its plan string drifted. A KNOWN
    tier (including ``basic``) still trims. This is the guarantee that
    ``tool_router._apply_tier_exposure`` documents.
    """
    from config import TOOL_FAMILIES

    if get_tier(plan, tiers) is None:
        return list(tools)  # unresolvable plan ⇒ true fail-open (never hide tools)

    fam_map = family_map if family_map is not None else TOOL_FAMILIES
    fams = enabled_families(plan, tiers)
    disabled = {f for f, on in fams.items() if not on}
    if not disabled:
        return list(tools)

    out: list = []
    for schema in tools or []:
        name = (schema.get("function") or {}).get("name", "")
        if name == "platform_execute":
            out.append(_prune_dispatcher(schema, disabled, fam_map))
            continue
        if _tool_family(name, fam_map) in disabled:
            continue  # drop a promoted/registry tool in a disabled family
        out.append(schema)
    return out


# --------------------------------------------------------------------------- #
# US-025 — plan recommendation (proposal stage). Pure, explainable rules; NO
# pricing math (display prices come straight from PLAN_TIERS).
# --------------------------------------------------------------------------- #

# Signals in the free-text segment that point at a tier. Kept small and
# explainable — the recommendation is a starting point the user can override.
_BUSINESS_SIGNALS = (
    "agency", "multiple teams", "several teams", "multiple pods", "departments",
    "organisation", "organization", "franchise", "enterprise",
)
_PRO_SIGNALS = (
    "developer", "engineering", "software", "code", "analytics", "dashboard",
    "sql", "data team", "automation",
)


def recommend_plan(segment: Optional[dict], team_size=None, tiers: Optional[dict] = None):
    """Recommend an assignable tier from the stored segment + expressed team size.

    Simple, explainable rules (NO pricing math): an org-scale operation (≥6 seats
    or multi-team language) → business; a small team, technical comfort, or
    code/data needs → pro; a solo operator → basic. Returns ``(plan, reason)``
    where reason is a short plain-language phrase for the proposal copy.

    ``team_size`` is read from the STORED segment (``segment['team_size']``, the
    key Auto captures through the tool) when not passed explicitly — so the two
    real callers, the proposal display and the ``plan_recommended`` funnel stamp,
    which both pass only the segment, always agree (an explicit arg still wins,
    for direct/unit use).
    """
    seg = segment or {}
    text = " ".join(str(seg.get(k) or "") for k in ("business", "goal", "comfort")).lower()
    raw_size = team_size if team_size is not None else seg.get("team_size")
    size = raw_size if isinstance(raw_size, int) and not isinstance(raw_size, bool) and raw_size > 0 else None

    if (size is not None and size >= 6) or any(s in text for s in _BUSINESS_SIGNALS):
        return "business", "you're running multiple teams or pods"
    if (size is not None and size >= 2) or "technical" in text or any(s in text for s in _PRO_SIGNALS):
        return "pro", "you've got a small team or code/data needs"
    return "basic", "you're a solo operator getting started"


def plan_proposal_copy(segment: Optional[dict], team_size=None, tiers: Optional[dict] = None) -> str:
    """The plan-recommendation line injected into the proposal stage guidance.

    Names the recommended tier with its DISPLAY price + early-access label, lists
    every assignable tier's price, states Enterprise is coming soon, and tells
    Auto to set the accepted plan via ``platform_update_onboarding`` (only
    assignable tiers). Display strings come from PLAN_TIERS — no charge is ever
    computed (Q5).
    """
    resolved = _tiers(tiers)
    plan, reason = recommend_plan(segment, team_size, resolved)
    tier = resolved.get(plan) or {}
    name = tier.get("display_name") or plan.title()
    price = tier.get("display_price_usd")
    options = " · ".join(
        f"{t.get('display_name') or p.title()} ${t.get('display_price_usd')}/mo"
        for p, t in assignable_tiers(resolved).items()
    )
    return (
        f"Recommend the **{name} plan — ${price}/mo** (early-access pricing) because {reason}. "
        f"Show the options — {options} (all early access) — and note Enterprise is coming soon. "
        f"On an explicit yes, set it with `platform_update_onboarding` (plan: \"{plan}\"); "
        f"only basic/pro/business are assignable."
    )
