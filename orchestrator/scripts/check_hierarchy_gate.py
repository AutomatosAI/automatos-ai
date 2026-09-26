#!/usr/bin/env python3
"""CI gate — every mutating platform_* action is gated (PRD-140 Phase 1, F151).

Every ActionDefinition registered in any file under
``orchestrator/modules/tools/discovery/`` (at any depth: the workspace tools in
``workspace_actions.py`` too, F179) with permission_level ``write`` or
``destructive`` must be one of:

  1. hierarchy-gated: a key of ``_HIERARCHY_TARGETS`` in platform_executor.py;
  2. flag-gated: registered with ``super_admin_only=True`` or
     ``admin_only=True``, which the executor enforces for the caller (a
     workspace tool clears the same gates, exec_workspace.clear_declared_gates);
  3. on ALLOW_LIST below, with a comment saying why it needs neither.

The audit keys on the declared type, not on the name: an ungated write whose
name has no per-target verb fails too. The verbs only suggest which fix fits.
An allow-list entry that is gated anyway, or is no longer a mutating
registration, fails as well, so the list only ever holds live exceptions.

Registrations are read with ``ast``: a flag named in a comment never counts,
and only a literal ``True`` gates. A registration the audit cannot read (a
name or permission_level that is not a literal string, an ActionDefinition
built outside ``registry.register(...)``, or a ``register(...)`` given anything
but an ``ActionDefinition(...)``) stops it rather than passing unseen.

Run as part of CI / pre-push:

    python orchestrator/scripts/check_hierarchy_gate.py

Exits 0 on success, 1 on unaccounted or stale entries, 2 when the sources
cannot be read.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Set

ROOT = Path(__file__).resolve().parents[2]
ACTIONS_DIR = ROOT / "orchestrator" / "modules" / "tools" / "discovery"
EXECUTOR = ACTIONS_DIR / "platform_executor.py"

PERMISSION_LEVELS = ("read", "write", "destructive")
MUTATING_LEVELS = ("write", "destructive")
DEFAULT_PERMISSION_LEVEL = "read"  # ActionDefinition's own default
GATE_FLAGS = ("super_admin_only", "admin_only")

# Mutating actions that are intentionally neither hierarchy- nor flag-gated.
# Each entry must be justified — if you add to this list, write a comment.
ALLOW_LIST: Set[str] = {
    # Creation actions — no existing target to scope; rate-limited.
    "platform_create_agent",
    "platform_create_task",
    "platform_create_playbook",
    "platform_create_mission",
    "platform_create_blueprint",
    # Workspace / memory writes — not hierarchy-scoped.
    "platform_store_memory",
    "platform_delete_memory",
    "platform_delete_document",
    "platform_reprocess_document",
    # Marketplace installs — affect the workspace, not a specific agent.
    "platform_install_plugin",
    "platform_install_skill",
    "platform_install_model",
    "platform_install_marketplace_agent",  # F151: REST agents:create, editor and up
    # Auto / orchestrator-only writes.
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
    # F147: workspace-scoped, no agent target.
    "platform_remove_member",              # destructive + confirmation (admin grant); never the owner
    "platform_update_widget_config",       # fail-closed whitelist of three public widget keys
    "platform_update_mission_plan",        # edits a plan still awaiting the owner's approval
    "platform_update_onboarding",          # the workspace's own onboarding state
    # F151: a mission's own lifecycle — approving needs a person under
    # always-ask (F036); the rest act on the workspace's own missions.
    "platform_approve_mission",
    "platform_cancel_mission",
    "platform_pause_mission",
    "platform_resume_mission",
    "platform_reject_mission",
    "platform_replan_mission",
    # F151: a run asking or telling its owner; each writes only its own
    # question, note or checkpoint.
    "platform_ask_human",
    "platform_ask_orchestrator",
    "platform_notify_owner",
    "platform_checkpoint_thread",
    # F151: the workspace's own watches.
    "platform_create_watch",
    "platform_cancel_watch",
    # F151: the workspace's own content and indexes.
    "platform_create_blog_post",
    "platform_create_social_post",         # PRD-251: REST POST documents:create, editor and up
    "platform_update_social_post",         # PRD-251: REST PATCH documents:update, editor and up
    "platform_submit_social_post",         # PRD-251: REST POST .../submit documents:update; asks a person, never publishes
    "platform_generate_cover_image",
    "platform_upload_document",
    "platform_scan_business_site",
    "platform_shopify_sync_catalog",
    "platform_codegraph_index",
    "platform_codegraph_reindex",
    "platform_codegraph_set_auto_reindex",
    # F151: runs only where platform_set_skill_script_execution (admin_only +
    # confirmation) enabled scripts for the skill.
    "platform_run_skill_script",
    # F179: the workspace's own sandbox (its files, shell, renders and repos);
    # no agent, ticket or member to scope.
    "workspace_write_file",
    "workspace_exec",
    "workspace_html_to_png",
    "workspace_git",
    # F179: confirmation-gated, a card on every lane but an owner's or admin's
    # own chat turn; it publishes a raster image only.
    "workspace_get_public_url",
}

# A hint only — names that suggest a mutation against a specific target, whose
# fix is usually a _HIERARCHY_TARGETS entry rather than a flag.
SUSPICIOUS_VERBS = ("update_", "delete_", "assign_", "unassign_", "configure_", "add_", "remove_")


class Registration(NamedTuple):
    permission_level: str
    flag_gated: bool


def _literal(node: Optional[ast.expr]) -> object:
    return node.value if isinstance(node, ast.Constant) else None


def _is_action_definition(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "ActionDefinition") or (
        isinstance(func, ast.Attribute) and func.attr == "ActionDefinition")


def _is_register_call(node: ast.AST) -> bool:
    """``registry.register(...)``, as every actions module writes it. Another
    object's register (a Deliverable's, PRD-251 US-117) is not a registration; an
    ActionDefinition registered any other way still fails the built-outside check."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "registry")


def _registered_definition(node: ast.AST) -> Optional[ast.Call]:
    """The ActionDefinition(...) in ``<registry>.register(ActionDefinition(...))``."""
    if _is_register_call(node) and len(node.args) == 1 and _is_action_definition(node.args[0]):
        return node.args[0]
    return None


def _unreadable(path: Path, line: int, why: str) -> None:
    sys.stderr.write(f"ERROR: {path.name}:{line}: {why}\n")
    sys.exit(2)


def _read_registration(path: Path, definition: ast.Call) -> tuple:
    kwargs = {kw.arg: kw.value for kw in definition.keywords if kw.arg}
    name = _literal(kwargs.get("name"))
    level = _literal(kwargs["permission_level"]) if "permission_level" in kwargs else DEFAULT_PERMISSION_LEVEL
    if not isinstance(name, str) or level not in PERMISSION_LEVELS:
        _unreadable(path, definition.lineno, "name and permission_level must be literal strings")
    flag_gated = any(_literal(kwargs.get(flag)) is True for flag in GATE_FLAGS)
    return name, Registration(level, flag_gated)


def collect_registrations() -> Dict[str, Registration]:
    """Map every action registered under ACTIONS_DIR to its declared level and flags."""
    found: Dict[str, Registration] = {}
    for path in sorted(ACTIONS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        registered = [d for d in map(_registered_definition, ast.walk(tree)) if d is not None]
        built = [n for n in ast.walk(tree) if _is_action_definition(n)]
        if len(built) != len(registered):
            _unreadable(path, built[0].lineno, "an ActionDefinition is built outside registry.register(...)")
        other = [n for n in ast.walk(tree) if _is_register_call(n) and _registered_definition(n) is None]
        if other:
            _unreadable(path, other[0].lineno, "register(...) is given something other than an ActionDefinition(...)")
        for definition in registered:
            name, registration = _read_registration(path, definition)
            found[name] = registration
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


def _report_unaccounted(unaccounted: list, registrations: Dict[str, Registration]) -> None:
    sys.stderr.write(
        "FAIL: hierarchy gate — these write/destructive actions are neither "
        "hierarchy-gated, admin_only / super_admin_only, nor on the allow-list:\n"
    )
    for name in unaccounted:
        hint = " (per-target verb: likely a _HIERARCHY_TARGETS entry)" if any(
            verb in name for verb in SUSPICIOUS_VERBS) else ""
        sys.stderr.write(f"  - {name} [{registrations[name].permission_level}]{hint}\n")
    sys.stderr.write(
        "\nFix by gating the action — an entry in _HIERARCHY_TARGETS in\n"
        "  orchestrator/modules/tools/discovery/platform_executor.py\n"
        "or admin_only / super_admin_only on its ActionDefinition — OR, if it\n"
        "needs no gate, adding the name (with a comment explaining why) to\n"
        "ALLOW_LIST in orchestrator/scripts/check_hierarchy_gate.py\n"
    )


def _report_stale(stale: list) -> None:
    sys.stderr.write(
        "FAIL: hierarchy gate — these ALLOW_LIST entries are gated anyway or "
        "are no longer write/destructive registrations; remove them:\n"
    )
    for name in stale:
        sys.stderr.write(f"  - {name}\n")


def main() -> int:
    registrations = collect_registrations()
    hierarchy_gated = collect_gated_actions()
    mutating = {name for name, reg in registrations.items() if reg.permission_level in MUTATING_LEVELS}
    flag_gated = {name for name in mutating if registrations[name].flag_gated}
    gated = (hierarchy_gated & mutating) | flag_gated

    unaccounted = sorted(mutating - gated - ALLOW_LIST)
    stale = sorted(name for name in ALLOW_LIST if name not in mutating or name in gated)
    if unaccounted:
        _report_unaccounted(unaccounted, registrations)
    if stale:
        _report_stale(stale)
    if unaccounted or stale:
        return 1

    print(
        f"OK: hierarchy gate — {len(mutating)} write/destructive actions: "
        f"{len(hierarchy_gated & mutating)} hierarchy-gated, "
        f"{len(flag_gated - hierarchy_gated)} admin_only / super_admin_only, "
        f"{len(ALLOW_LIST)} on the allow-list."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
