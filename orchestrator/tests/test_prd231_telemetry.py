"""PRD-231 US-006 — activation + size telemetry on the [skills] log line.

Pure, LLM-free, Postgres-free. Renders SkillsSection for a two-skill Auto
fixture (charter core + ops L1) and captures the activation log record — the one
deliverable of US-006. No dashboards, no new sinks: a week of these lines yields
the real per-turn saving and the ops-skill activation rate.
"""

import asyncio
import logging
from types import SimpleNamespace

from modules.context.sections.base import SectionContext
from modules.context.sections.skills import SkillsSection

CORE_SKILL = "platform-management"
OPS_SKILL = "platform-operations"


def _skill(name, body, description=""):
    s = SimpleNamespace()
    s.name = name
    s.prompt_template = body
    s.description = description
    s.is_active = True
    s.id = None
    s.content_hash = None
    s.tools_schema = None
    return s


def _render_and_capture(agent, caplog):
    with caplog.at_level(logging.INFO, logger="modules.context.sections.skills"):
        asyncio.run(SkillsSection().render(SectionContext(agent=agent, workspace_id="ws-006")))
    recs = [r for r in caplog.records if "[skills] activation" in r.getMessage()]
    assert recs, "the [skills] activation log line did not fire"
    return recs[0].getMessage()


def test_activation_log_carries_core_tokens_and_l1_count(caplog):
    charter_body = "CHARTER BODY " * 100
    charter = _skill(CORE_SKILL, charter_body, "charter")
    ops = _skill(OPS_SKILL, "OPS COOKBOOK " * 500, "ops cookbook trigger")
    msg = _render_and_capture(SimpleNamespace(skills=[charter, ops]), caplog)

    assert "core_tokens=" in msg
    assert "l1_count=" in msg
    # core_tokens ≈ rendered (stripped) charter body // 4; the ops body is NOT
    # counted — it paid only its one-line L1 tax this turn (the whole point).
    expected_core_tokens = len(charter_body.strip()) // 4
    assert f"core_tokens={expected_core_tokens}" in msg
    assert "l1_count=1" in msg


def test_two_skill_fixture_logs_charter_core_ops_l1(caplog):
    charter = _skill(CORE_SKILL, "CHARTER BODY", "charter")
    ops = _skill(OPS_SKILL, "OPS BODY", "ops cookbook trigger")
    msg = _render_and_capture(SimpleNamespace(skills=[charter, ops]), caplog)

    # charter is always-on (core); ops is offered at L1 — the whole point of PRD-231
    assert f"core_always_on=['{CORE_SKILL}']" in msg
    assert f"l1_offered=['{OPS_SKILL}']" in msg


def test_l1_count_is_zero_when_only_core_present(caplog):
    charter = _skill(CORE_SKILL, "CHARTER BODY " * 10, "charter")
    msg = _render_and_capture(SimpleNamespace(skills=[charter]), caplog)

    assert "l1_count=0" in msg
    assert f"l1_offered=[]" in msg
    assert "core_tokens=" in msg  # still records the always-on cost
