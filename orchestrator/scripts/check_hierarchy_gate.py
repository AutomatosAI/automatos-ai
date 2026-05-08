#!/usr/bin/env python3
"""CI gate — every mutating platform_* action must route through
hierarchy permission checks (PRD-140 Phase 1).

Strategy:
  1. Scan ``orchestrator/modules/tools/discovery/actions_*.py`` for
     ``ActionDefinition`` registrations with permission_level
     ``write`` or ``destructive``.
  2. Read the ``_HIERARCHY_TARGETS`` map from ``platform_executor.py``.
  3. Any name that *looks* like a per-target mutation (update/delete/
     assign/configure/add/remove) but is missing from the map fails
     the gate.

Run as part of CI / pre-push:

    python orchestrator/scripts/check_hierarchy_gate.py

Exits 0 on success, 1 on missing entries.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parents[2]
ACTIONS_DIR = ROOT / "orchestrator" / "modules" / "tools" / "discovery"
EXECUTOR = ACTIONS_DIR / "platform_executor.py"

# Actions that are intentionally NOT hierarchy-gated.
# Each entry must be justified — if you add to this list, write a comment.
ALLOW_LIST: Set[str] = {
    # Creation actions — no existing target to scope. Workspace-scoped
    # writes are protected by admin_only / rate_limit, not hierarchy.
    "platform_create_agent",
    "platform_create_task",
    "platform_create_playbook",
    "platform_create_recipe",
    "platform_create_mission",
    "platform_create_blueprint",
    "platform_create_workspace_skill",  # also enforced as TARGET_SKILL in map
    # Workspace / memory writes — not hierarchy-scoped.
    "platform_store_memory",
    "platform_delete_memory",
    "platform_delete_document",
    "platform_reprocess_document",
    # Marketplace installs — affect the workspace, not a specific agent.
    "platform_install_plugin",
    "platform_install_skill",
    "platform_install_model",
    # Auto / orchestrator-only writes — protected by admin gate.
    "platform_send_notification",
    "platform_update_auto_reporting_prefs",
    "platform_harness_trigger",
    "platform_acknowledge_report",
    "platform_link_report_to_task",
    "platform_submit_report",
    "platform_publish_blog_post",
    "platform_update_blog_post",
    "platform_update_blueprint",
    "platform_schedule_task",
    "platform_cancel_scheduled_task",
    "platform_schedule_playbook",
    "platform_execute_playbook",
    "platform_execute_recipe",
}

# Heuristic — names that suggest a mutation against a specific target.
SUSPICIOUS_VERBS = ("update_", "delete_", "assign_", "unassign_", "configure_", "add_", "remove_")


def collect_writeable_actions() -> Set[str]:
    """Return action names registered with permission_level write/destructive.

    Each block runs from one ``registry.register(ActionDefinition(`` up to
    the next, so name= and permission_level= are matched within the same
    definition without trying to balance nested brackets.
    """
    found: Set[str] = set()
    for path in sorted(ACTIONS_DIR.glob("actions_*.py")):
        text = path.read_text(encoding="utf-8")
        blocks = re.split(r"registry\.register\(ActionDefinition\(", text)[1:]
        for block in blocks:
            name_match = re.search(r'\bname\s*=\s*"([^"]+)"', block)
            perm_match = re.search(r'\bpermission_level\s*=\s*"(read|write|destructive)"', block)
            if not name_match or not perm_match:
                continue
            if perm_match.group(1) in ("write", "destructive"):
                found.add(name_match.group(1))
    return found


def collect_gated_actions() -> Set[str]:
    """Return action names listed in _HIERARCHY_TARGETS in platform_executor."""
    text = EXECUTOR.read_text(encoding="utf-8")
    match = re.search(
        r"_HIERARCHY_TARGETS[^\{]*=\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    if not match:
        sys.stderr.write("ERROR: could not find _HIERARCHY_TARGETS in platform_executor.py\n")
        sys.exit(2)
    body = match.group(1)
    return set(re.findall(r'"([^"]+)"\s*:\s*\(', body))


def main() -> int:
    writeable = collect_writeable_actions()
    gated = collect_gated_actions()

    suspicious_unaccounted = sorted(
        name
        for name in writeable
        if name not in gated
        and name not in ALLOW_LIST
        and any(verb in name for verb in SUSPICIOUS_VERBS)
    )

    if suspicious_unaccounted:
        sys.stderr.write(
            "FAIL: PRD-140 hierarchy gate — these mutating actions are "
            "neither hierarchy-gated nor on the allow-list:\n"
        )
        for name in suspicious_unaccounted:
            sys.stderr.write(f"  - {name}\n")
        sys.stderr.write(
            "\nFix by EITHER adding an entry to _HIERARCHY_TARGETS in\n"
            "  orchestrator/modules/tools/discovery/platform_executor.py\n"
            "OR — if the action is intentionally workspace-scoped — adding\n"
            "the name (with a comment explaining why) to ALLOW_LIST in\n"
            "  orchestrator/scripts/check_hierarchy_gate.py\n"
        )
        return 1

    print(
        f"OK: PRD-140 hierarchy gate — {len(gated)} actions gated, "
        f"{len(ALLOW_LIST)} on allow-list, {len(writeable)} mutating actions total."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
