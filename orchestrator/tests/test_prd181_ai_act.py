"""PRD-181 S6 — EU-AI-Act autonomy-tier risk classification (scaffold).

The Annex-IV write-up is a doc scaffold (owner decision); the *machine* part that
ships is a small, pure risk-tier classification that maps the policy plane's risk
classes onto EU-AI-Act Art.14 human-oversight tiers, which the S5 approval card
reads. Pure — no DB, no config side-effects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.policy.ai_act import (  # noqa: E402
    OversightTier,
    classify_risk_tier,
    oversight_for_risk,
)
from modules.policy.policy_document import (  # noqa: E402
    RISK_DESTRUCTIVE,
    RISK_EXTERNAL,
    RISK_INTERNAL_WRITE,
    RISK_PUBLISH,
    RISK_READ,
)


def test_every_policy_risk_class_maps_to_a_tier():
    """Each of the 5 policy risk classes has an oversight tier + rationale — no
    risk class is left unclassified (fail-safe: unknown ⇒ highest oversight)."""
    for risk in (RISK_READ, RISK_INTERNAL_WRITE, RISK_PUBLISH, RISK_EXTERNAL, RISK_DESTRUCTIVE):
        mapping = oversight_for_risk(risk)
        assert mapping.tier in OversightTier
        assert mapping.rationale, "each risk must carry a human-readable oversight rationale"
        assert mapping.risk_class == risk


def test_read_is_lowest_oversight():
    m = oversight_for_risk(RISK_READ)
    assert m.tier == OversightTier.MONITOR  # no human in the loop needed


def test_destructive_and_external_require_human_in_the_loop():
    for risk in (RISK_DESTRUCTIVE, RISK_EXTERNAL, RISK_PUBLISH):
        m = oversight_for_risk(risk)
        assert m.tier == OversightTier.HUMAN_IN_THE_LOOP
        assert m.requires_approval is True


def test_unknown_risk_fails_safe_to_highest_oversight():
    m = oversight_for_risk("something_new_and_unknown")
    assert m.tier == OversightTier.HUMAN_IN_THE_LOOP
    assert m.requires_approval is True


def test_classify_risk_tier_from_tool_name():
    """The convenience classifier goes straight from a tool name to the oversight
    mapping (used by the approval payload builder)."""
    m = classify_risk_tier("platform_delete_agent", permission_level="destructive")
    assert m.tier == OversightTier.HUMAN_IN_THE_LOOP
    assert m.risk_class == RISK_DESTRUCTIVE

    m2 = classify_risk_tier("composio_send_email", is_composio=True)
    assert m2.risk_class == RISK_EXTERNAL
    assert m2.requires_approval is True


def test_mapping_is_json_serialisable():
    """The mapping goes into the approval payload → must serialise."""
    import json

    m = oversight_for_risk(RISK_EXTERNAL)
    json.dumps(m.to_dict())
    d = m.to_dict()
    assert d["tier"] == "human_in_the_loop"
    assert "rationale" in d and "risk_class" in d
