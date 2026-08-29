"""PRD-222 W1S5 / US-010 — trust guards + onboarding summary Deliverable.

Locks the trust defaults so a regression fails loud:

1. Static guard — no onboarding-owned source file ever SETS ``skip_verification``
   or ``auto_approve`` truthy (prohibition mentions are allowed; the guard is
   self-checked to prove it bites on a real violation).
2. Mission default — the approval policy that governs every tool-created mission
   defaults to ``awaiting_approval`` (``always_ask``), and ``full_auto`` without
   the autonomy gate fails safe to await.
3. Completed handoff — the section instructs Auto to write the onboarding summary
   via ``platform_submit_report(report_type='onboarding')`` and only then advance
   to ``completed`` — and ``onboarding`` is a VALID report type so it never
   dead-ends (schema ↔ handler truth).
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import modules.context.sections.onboarding as onb_mod
from core.services.approval_policy import evaluate_approval
from modules.context.sections.base import SectionContext
from modules.context.sections.onboarding import OnboardingSection

_ORCH_ROOT = Path(onb_mod.__file__).resolve().parents[3]  # orchestrator/

# The onboarding-owned surface: the section, the state machine, the trial ledger,
# the capability report, the new platform tools (onboarding + intake), and the
# Wave-1b (US-016) dev-reset / built-artifact wipe surface.
_ONBOARDING_OWNED = [
    "modules/context/sections/onboarding.py",
    "services/onboarding_state.py",
    "services/trial_ledger.py",
    "services/capability_report.py",
    "modules/tools/discovery/actions_onboarding.py",
    "modules/tools/discovery/handlers_onboarding.py",
    "modules/tools/discovery/actions_intake.py",
    "modules/tools/discovery/handlers_intake.py",
    # Wave-1b US-016 dev reset: reset_onboarding + its optional trial re-grant /
    # built-artifact wipe / credential wipe. The wipe REUSES the workspace_purge
    # machinery (parameterised to spare survivors) and is exposed on the existing
    # workspaces router — both are onboarding-owned and must stay trust-clean.
    "services/workspace_purge.py",
    "api/workspaces.py",
]

# Matches a TRUTHY assignment of either flag — `x=True`, `"x": True`,
# `cfg["x"] = True` — but NOT a prohibition mention (no `= True` / `: True`).
_SET_TRUE = re.compile(r"(skip_verification|auto_approve)\W{0,4}[=:]\s*(True|true|1)\b")


def _read(rel: str) -> str:
    return (_ORCH_ROOT / rel).read_text()


# --------------------------------------------------------------------------- #
# AC1 — static trust-flag guard over the onboarding-owned files
# --------------------------------------------------------------------------- #


def test_no_onboarding_file_sets_trust_flags():
    offenders = []
    for rel in _ONBOARDING_OWNED:
        for i, line in enumerate(_read(rel).splitlines(), 1):
            if _SET_TRUE.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "onboarding path sets a trust flag:\n" + "\n".join(offenders)


def test_all_owned_files_exist():
    # A renamed/moved file must not silently drop out of the guard's coverage.
    for rel in _ONBOARDING_OWNED:
        assert (_ORCH_ROOT / rel).is_file(), f"missing owned file {rel}"


def test_trust_guard_self_check_bites():
    # The guard MUST match real violations …
    assert _SET_TRUE.search("skip_verification=True")
    assert _SET_TRUE.search('config["auto_approve"] = True')
    assert _SET_TRUE.search('    "skip_verification": True,')
    # … and MUST NOT flag the section's prohibition lines.
    assert not _SET_TRUE.search(
        "- NEVER create a mission or run a tool with `skip_verification` or "
        "`auto_approve` set — every build is verified"
    )
    assert not _SET_TRUE.search("Onboarding must NEVER set ``skip_verification`` or ``auto_approve``")


# --------------------------------------------------------------------------- #
# AC2 — tool-created missions default to awaiting_approval
# --------------------------------------------------------------------------- #


def _db_with_settings(settings):
    ws = MagicMock()
    ws.settings = settings
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = ws
    return db


def test_default_workspace_policy_awaits_approval():
    # Empty settings → always_ask → a tool-created mission awaits, not auto-runs.
    decision = evaluate_approval(_db_with_settings({}), uuid4(), 0.10)
    assert decision.auto_approve is False
    assert decision.policy == "always_ask"


def test_full_auto_without_autonomy_gate_still_awaits():
    db = _db_with_settings({"approval_policy": {"policy": "full_auto"}})
    with patch("core.services.auto_autonomy.is_full_autonomy", return_value=False):
        decision = evaluate_approval(db, uuid4(), 0.10)
    assert decision.auto_approve is False  # fail-safe: gate off → await


def test_create_mission_defaults_auto_approve_false_in_source():
    # The coordinator only auto-approves on an EXPLICIT override; a missing
    # config key defaults to False (so onboarding, which never sets it, awaits).
    src = _read("services/coordinator_service.py")
    assert 'mission_config.get("auto_approve", False)' in src


# --------------------------------------------------------------------------- #
# AC3 — completed handoff writes the summary Deliverable, then advances
# --------------------------------------------------------------------------- #


def _render_powerup():
    ws = MagicMock()
    ws.onboarding = {
        "stage": "powerup",
        "stages": {},
        "segment": {},
        "trial": {"granted_usd": 5.0, "spent_usd": 1.0, "state": "active"},
    }
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = ws
    ctx = SectionContext(agent=None, workspace_id="ws-1", db_session=db, messages=[])
    return asyncio.run(OnboardingSection().render(ctx))


def test_powerup_handoff_instructs_summary_report_then_completed():
    out = _render_powerup()
    assert "platform_submit_report" in out
    assert "report_type `onboarding`" in out
    assert "advance_to `completed`" in out
    # Order: write the summary BEFORE advancing to completed.
    assert out.index("platform_submit_report") < out.index("advance_to `completed`")


def test_onboarding_is_a_valid_report_type_so_the_handoff_does_not_dead_end():
    # The instruction must be executable: report_type 'onboarding' is accepted by
    # the handler AND advertised in the schema (schema ↔ handler truth, US-011).
    handler_src = _read("modules/tools/discovery/handlers_reports.py")
    schema_src = _read("modules/tools/discovery/actions_reports.py")
    assert re.search(r"valid_types\s*=\s*\{[^}]*['\"]onboarding['\"]", handler_src)
    assert '"onboarding"' in schema_src  # present in the submit-report enum
