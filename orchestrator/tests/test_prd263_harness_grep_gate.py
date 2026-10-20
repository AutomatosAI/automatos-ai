"""PRD-263 FR-12 (first version, US-016) — HARNESS's reading and judging modules act on nothing.

The read model (services.harness_metrics), the mode vocabulary (services.harness_mode)
and the statistics (core.stats.harness_verdicts) only read and compute. None of them
may name a tool executor, or import an LLM client, so no change and no model call can
start from them. Later stories add the modules they create to GATED.

US-016 also deletes the auto-apply loop, the card-to-Done path and escalation, with the
settings that tuned them: no source outside the tests names them again, and the
weekly run files no card and sends no channel notice.

US-047 deletes the always-approving /approve path (api/harness_commands.py) and the
owner-only bypass key it set: no source outside the tests names either again.

US-048 deletes the direct apply helper, the card parsers and the F156 legacy ledger-file
importer from services.harness_service, and submits the audit report straight through
ReportService, so harness_service.py joins GATED and names none of them again.

US-049 moves PRESCRIBE onto the change-kind registry (services.harness_changes), which
joins GATED. The placeholder kinds and their constants are gone from every service:
grep ``review_needed`` in orchestrator/services returns 0 (section 11-Q9).

US-052 and US-053 add services.harness_grants (the apply rail's ledger moves, then
reconciliation and the daily grant sweep), which joins GATED: it reads and moves ledger
rows and grants, and applies nothing.

US-054 raises the owner's Approvals cards from services.harness_grants (the grants, their
ledger rows and the evidence ticket) with their text from services.harness_cards, which
joins GATED: raising a card applies nothing and calls no model.

US-056 judges each applied change against its frozen reference in
services.harness_post_change, which joins GATED: it suggests a rollback, which becomes its
own Approvals card, and applies nothing.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Tuple

_ORCH = Path(__file__).resolve().parents[1]
HARNESS_SERVICE = "services/harness_service.py"
GATED = (
    "services/harness_metrics.py",
    "services/harness_mode.py",
    "core/stats/harness_verdicts.py",
    HARNESS_SERVICE,
    "services/harness_changes.py",
    "services/harness_grants.py",
    "services/harness_cards.py",
    "services/harness_post_change.py",
)
EXECUTORS = re.compile(r"\b(PlatformActionExecutor|UnifiedToolExecutor|execute_tool)\b")
# An import of any of these modules, or of these names, is an LLM client import.
LLM_MODULES = ("core.llm", "anthropic", "openai", "litellm")
LLM_NAMES = frozenset({"LLMManager", "create_llm_manager", "get_decision_engine"})
# What US-016 deleted (grep across orchestrator/, tests excluded, returns 0).
DELETED = (
    "_phase_apply", "_apply_approved_board_tasks", "_note_needs_an_approve", "_workspace_allows_auto_apply",
    "_auto_apply_max_risk", "_get_harness_config", "_maybe_escalate", "_build_escalation_message",
    "_workspace_has_channel", "HARNESS_AUTO_APPLY_MAX_RISK_STANDARD", "HARNESS_AUTO_APPLY_MAX_RISK_FULL",
    "HARNESS_HIGH_PRIORITY_RISK",
)
# What US-047 deleted: the /approve module and its handler, and the bypass key, by constant and by value.
DELETED_APPROVE = (
    "harness_commands", "handle_harness_command", "_governed_verdict", "_find_task_by_rx",
    "OWNERS_OWN_DECISION", "owners_own_decision",
)
# What US-048 deleted from services.harness_service (grep in harness_service.py returns 0).
DELETED_APPLY = (
    "_auto_apply_prescription", "_PLACEHOLDER_PROPOSED_VALUE", "review_needed", "_snapshot_current_value",
    "_resolve_agents_by_name", "_extract_agents_list", "_parse_harness_task", "_extract_json_section",
    "_extract_text_section", "_extract_risk_score", "_read_applied_tasks", "_write_applied_tasks",
    "LEGACY_LEDGER_FILE", "UNREADABLE_LEDGER_", "_import_legacy_ledger", "_file_unreadable_ledger_card",
)
# What US-049 deleted: the placeholder emitters' kinds, their placeholder and its threshold
# (grep across orchestrator/services returns 0).
DELETED_PLACEHOLDERS = (
    "review_needed", "model_change_same_tier", "description_update", "_PLACEHOLDER_PROPOSED_VALUE",
    "_MIN_SUCCESS_RATE_FOR_COST_OPT",
)
SERVICES = "services"
# Skipped by that grep: the tests, and caches and tool state (__pycache__, dot directories).
SKIPPED_PARTS = ("tests", "__pycache__")
CONFIG_SURFACE = "reports/config-surface.json"
# How the weekly run filed a [HARNESS] card, found done ones, and sent an escalation notice.
RUN_SENDS = ("platform_create_task", "platform_list_tasks", "send_workspace_notification")


def violations(source: str) -> List[str]:
    """Every executor name and LLM client import in ``source``."""
    found = [f"names {match.group(1)}" for match in EXECUTORS.finditer(source)]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules, names = [alias.name for alias in node.names], []
        elif isinstance(node, ast.ImportFrom):
            modules, names = [node.module or ""], [alias.name for alias in node.names]
        else:
            continue
        found += [f"imports {m}" for m in modules if any(m == p or m.startswith(p + ".") for p in LLM_MODULES)]
        found += [f"imports {n}" for n in names if n in LLM_NAMES]
    return found


def test_the_gated_modules_name_no_executor_and_import_no_llm_client():
    for rel in GATED:
        source = (_ORCH / rel).read_text(encoding="utf-8")
        assert violations(source) == [], rel


def test_the_gate_reports_each_kind_of_violation():
    source = (
        "from core.llm.manager import LLMManager\n"
        "import openai\n"
        "from modules.tools.discovery.platform_executor import PlatformActionExecutor\n"
        "result = execute_tool('platform_update_agent', {})\n"
    )
    assert violations(source) == [
        "names PlatformActionExecutor", "names execute_tool",
        "imports core.llm.manager", "imports LLMManager", "imports openai",
    ]


def _sources() -> List[Path]:
    """Every Python file under orchestrator/ outside the tests, and the config surface."""
    paths = [
        path for path in _ORCH.rglob("*.py")
        if not any(part in SKIPPED_PARTS or part.startswith(".") for part in path.relative_to(_ORCH).parts)
    ]
    return paths + [_ORCH / CONFIG_SURFACE]


def _offenders(names: Tuple[str, ...]) -> List[str]:
    """Each source outside the tests that still names one of ``names``."""
    return [
        f"{path.relative_to(_ORCH)}: {name}"
        for path in _sources()
        for name in names
        if name in path.read_text(encoding="utf-8", errors="ignore")
    ]


def test_no_source_names_the_deleted_auto_apply_loop_or_escalation():
    assert _offenders(DELETED) == []


def test_no_source_names_the_deleted_approve_path_or_its_bypass_key():
    assert _offenders(DELETED_APPROVE) == []


def test_the_weekly_run_files_no_card_and_sends_no_notice():
    source = (_ORCH / HARNESS_SERVICE).read_text(encoding="utf-8")
    assert [name for name in RUN_SENDS if name in source] == []


def test_harness_service_names_none_of_the_deleted_apply_helpers():
    source = (_ORCH / HARNESS_SERVICE).read_text(encoding="utf-8")
    assert [name for name in DELETED_APPLY if name in source] == []


def test_no_service_names_the_placeholder_kinds():
    offenders = [
        f"{path.relative_to(_ORCH)}: {name}"
        for path in (_ORCH / SERVICES).rglob("*.py")
        for name in DELETED_PLACEHOLDERS
        if name in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []
