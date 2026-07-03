"""Policy plane — EU-AI-Act Art.14 oversight tiers (PRD-181 S6, scaffold).

Maps the policy plane's pure risk classes (``policy_document.classify_action`` →
read / internal_write / publish / external_side_effect / destructive) onto a
small **human-oversight tier** with a plain-language rationale. This is the
machine half of the S6 scaffold: the S5 approval card reads it so a human
approver sees the AI-Act risk classification and *why* they are in the loop.

Owner decision: this is a **scaffold**, not the full formal risk-classification
technical file. The mapping is intentionally coarse and conservative
(unknown ⇒ highest oversight). The formal Annex-IV / risk-classification write-up
is the flagged fast-follow (see ``docs/compliance/EU-AI-ACT-ANNEX-IV.md``).

Pure: stdlib only, no DB, no config, no network — it is a lookup over the risk
classes plus a re-use of the pure ``classify_action`` classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from modules.policy.policy_document import (
    RISK_DESTRUCTIVE,
    RISK_EXTERNAL,
    RISK_INTERNAL_WRITE,
    RISK_PUBLISH,
    RISK_READ,
    classify_action,
)


class OversightTier(str, Enum):
    """EU-AI-Act Art.14 human-oversight posture for an action.

    ``MONITOR``            — no human in the loop; the action is logged (Art.12)
                             and reviewable after the fact. Reads / low-risk work.
    ``HUMAN_ON_THE_LOOP``  — proceeds, but a human can intervene / is notified;
                             internal writes that change state but are reversible.
    ``HUMAN_IN_THE_LOOP``  — a human must approve *before* the action runs;
                             destructive, external side-effects, publish.
    """

    MONITOR = "monitor"
    HUMAN_ON_THE_LOOP = "human_on_the_loop"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"


@dataclass(frozen=True)
class OversightMapping:
    """One risk class mapped to its oversight tier + rationale (approval-card ready)."""

    risk_class: str
    tier: OversightTier
    rationale: str
    requires_approval: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_class": self.risk_class,
            "tier": self.tier.value,
            "rationale": self.rationale,
            "requires_approval": self.requires_approval,
        }


# The Art.14 mapping. Conservative: the two "always ask" classes (destructive,
# external, publish) are human-in-the-loop; internal writes are on-the-loop;
# reads are monitor-only. Rationales are written for a human approver to read.
_OVERSIGHT_BY_RISK: Dict[str, OversightMapping] = {
    RISK_READ: OversightMapping(
        RISK_READ, OversightTier.MONITOR,
        "Read-only: retrieves information and has no side effect. Logged for "
        "traceability (Art.12); no prior human approval required.",
        requires_approval=False,
    ),
    RISK_INTERNAL_WRITE: OversightMapping(
        RISK_INTERNAL_WRITE, OversightTier.HUMAN_ON_THE_LOOP,
        "Internal write: changes workspace state (drafts, memory, own task "
        "status) but stays inside the tenant and is reversible. Proceeds under "
        "monitoring; a human can intervene.",
        requires_approval=False,
    ),
    RISK_PUBLISH: OversightMapping(
        RISK_PUBLISH, OversightTier.HUMAN_IN_THE_LOOP,
        "Publish: makes content externally visible (template / brand-kit "
        "publish). A human approves before it goes live.",
        requires_approval=True,
    ),
    RISK_EXTERNAL: OversightMapping(
        RISK_EXTERNAL, OversightTier.HUMAN_IN_THE_LOOP,
        "External side-effect: acts on a third-party system (send / refund / "
        "discount / post, Shopify writes). Irreversible or customer-facing — a "
        "human must approve before it runs.",
        requires_approval=True,
    ),
    RISK_DESTRUCTIVE: OversightMapping(
        RISK_DESTRUCTIVE, OversightTier.HUMAN_IN_THE_LOOP,
        "Destructive: deletes or irreversibly mutates data (deletes, board "
        "Run-Now). A human must approve before it runs.",
        requires_approval=True,
    ),
}

# Fail-safe for an unknown / new risk class: highest oversight, never silently
# lower. Mirrors the policy document's "unknown ⇒ ask" posture.
_FALLBACK = OversightMapping(
    "unknown", OversightTier.HUMAN_IN_THE_LOOP,
    "Unclassified action: treated as requiring human approval until it is "
    "explicitly risk-classified (fail-safe).",
    requires_approval=True,
)


def oversight_for_risk(risk_class: Optional[str]) -> OversightMapping:
    """Return the oversight mapping for a policy risk class (fail-safe on unknown)."""
    if not risk_class:
        return _FALLBACK
    mapping = _OVERSIGHT_BY_RISK.get(risk_class)
    if mapping is None:
        # Preserve the actual class name in the fallback for the audit/card.
        return OversightMapping(
            risk_class, _FALLBACK.tier, _FALLBACK.rationale, _FALLBACK.requires_approval
        )
    return mapping


def classify_risk_tier(
    tool_name: str,
    *,
    permission_level: Optional[str] = None,
    is_composio: bool = False,
) -> OversightMapping:
    """Go straight from a tool call to its oversight mapping.

    Convenience for the approval-payload builder: classifies the action (pure)
    then maps the risk class to the oversight tier.
    """
    risk = classify_action(tool_name, permission_level=permission_level, is_composio=is_composio)
    return oversight_for_risk(risk)
