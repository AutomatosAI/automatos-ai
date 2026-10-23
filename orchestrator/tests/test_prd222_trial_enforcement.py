"""PRD-222 W1·S9 (US-005) — "no background burn" + credit-gate wiring guards.

The $5 trial ledger this file tested (``services.trial_ledger``: the routing
gate, the accrual, the daily spend counter) is DELETED — PRD-265 W1 replaced
it with the credit ledger (``services/credits/``). Pure behaviour tests of the
new routing/ledger/trial logic belong to the PRD-265 test files
(``test_prd265_credit_ledger.py`` / ``test_prd265_credit_ledger_realdb.py``);
they are not duplicated here.

What stays this file's own concern is the WIRING PRD-222 FR-17 ("no background
burn") and the credit gate depend on, asserted by source grep (no Postgres, no
live LLM, same as before):

- the credit gate sits at the one LLM call seam, before every provider call
  (``LLMManager._credit_gate`` -> ``services.credits.gate.before_platform_call``);
- the agent factory's own-keys-only check (``_platform_unfunded``) is defined
  once and shared by both the agent-creation and the chat-activation paths —
  no drift between them, as the old ``_resolve_trial_decision`` guaranteed;
- ``activate_agent`` re-raises the typed paused error instead of swallowing it;
- every background runner (heartbeat, scheduled tasks, playbook schedules,
  socials research) consults ``services.credits.routing.background_allowed``
  before it fires — an idle workspace on credits alone burns nothing, exactly
  as an idle trial workspace burned nothing before PRD-265.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (REPO / rel).read_text()


# --------------------------------------------------------------------------- #
# The credit gate sits at the one LLM call seam (core/llm/manager.py)
# --------------------------------------------------------------------------- #


def test_credit_gate_wired_at_the_manager_seam():
    src = _src("core/llm/manager.py")
    assert "_credit_gate" in src                 # the gate runs before every provider call
    assert "before_platform_call" in src          # services.credits.gate's check
    assert "_note_ask_cost" in src                # the call's cost is noted after
    assert "after_platform_call" in src


def test_credit_gate_raises_before_any_provider_call():
    """``generate_response`` / ``generate_response_sync`` call the gate before
    ``_ensure_provider_initialized`` — never after a call already started. (The
    async path runs it on a thread, ``asyncio.to_thread(self._credit_gate)``,
    because an ask's first check reads the database — F105.)"""
    src = _src("core/llm/manager.py")
    for method in ("async def generate_response(", "def generate_response_sync("):
        start = src.index(method)
        body = src[start:start + 1600]
        gate_at = body.index("self._credit_gate")
        provider_at = body.index("_ensure_provider_initialized()")
        assert gate_at < provider_at, f"{method} must gate before creating the provider"


# --------------------------------------------------------------------------- #
# _platform_unfunded: one check, shared by agent creation and activation
# --------------------------------------------------------------------------- #


def test_platform_unfunded_defined_once_and_shared():
    src = _src("modules/agents/factory/agent_factory.py")
    assert src.count("def _platform_unfunded(") == 1
    creation_start = src.index("async def _creation_key(")
    activation_start = src.index("async def _activation_key(")
    assert "_platform_unfunded(" in src[creation_start:creation_start + 1500], (
        "the agent-creation path must use the shared own-keys-only check"
    )
    assert "_platform_unfunded(" in src[activation_start:activation_start + 1500], (
        "the chat-activation path must use the same shared check — no drift between them"
    )


def test_activate_agent_reraises_credits_exhausted():
    """PRD-265 US-004: the paused workspace's typed error must reach the
    chat/mission surface, never swallowed into a generic 'failed to activate' None."""
    src = _src("modules/agents/factory/agent_factory.py")
    assert "from services.credits.routing import CreditsExhaustedError" in src
    assert "except CreditsExhaustedError:" in src
    assert "TrialExhaustedError" not in src  # the old typed error is gone, not just unused


# --------------------------------------------------------------------------- #
# No background burn (PRD-222 FR-17) — every scheduled runner consults
# services.credits.routing.background_allowed before it fires
# --------------------------------------------------------------------------- #


def test_no_background_burn_guards_present():
    for rel in [
        "services/heartbeat_service.py",
        "services/scheduled_task_steps.py",   # the scheduler's guard (PRD-252 3c moved it here)
        "services/playbook_scheduler.py",
        "services/socials_plan_research.py",
    ]:
        src = _src(rel)
        assert "background_allowed" in src, f"missing the credits background guard in {rel}"


def test_scheduled_task_service_checks_before_every_firing():
    assert "if skipped_for_credits(db, task, task_id):" in _src("services/scheduled_task_service.py")


def test_no_stale_trial_ledger_references():
    """The ledger this file used to test is deleted; nothing on the guarded
    seams should still name it (a leftover import would resurrect a dead
    module at collection, not at runtime)."""
    for rel in [
        "services/heartbeat_service.py",
        "services/scheduled_task_steps.py",
        "services/scheduled_task_service.py",
        "services/playbook_scheduler.py",
        "services/socials_plan_research.py",
        "modules/agents/factory/agent_factory.py",
        "core/llm/manager.py",
    ]:
        assert "trial_ledger" not in _src(rel), f"stale trial_ledger reference in {rel}"
