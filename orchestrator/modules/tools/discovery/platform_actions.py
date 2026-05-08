"""
Platform Action Definitions (PRD-64)
=====================================

Curated set of platform actions that Auto can execute.
These are the operations Auto can perform on the Automatos platform itself.

ActionDefinitions are split into domain-specific files.
This module re-exports register_all_actions() as the single entry point.
"""

from .action_registry import ActionRegistry

from .actions_agents import register_agents_actions
from .actions_playbooks import register_playbooks_actions
from .actions_analytics import register_analytics_actions
from .actions_documents import register_documents_actions
from .actions_workspace import register_workspace_actions_defs
from .actions_monitoring import register_monitoring_actions
from .actions_search import register_search_actions
from .actions_tools_llms import register_tools_llms_actions
from .actions_marketplace import register_marketplace_actions
from .actions_skills import register_skills_actions
from .actions_assignments import register_assignments_actions
from .actions_board_tasks import register_board_task_actions
from .actions_scheduling import register_scheduling_actions
from .actions_reports import register_report_actions
from .actions_field import register_field_actions
from .actions_blog import register_blog_actions
from .actions_missions import register_mission_actions
from .actions_analytics_enhanced import register_analytics_enhanced_actions
from .actions_governance import register_governance_actions
from .actions_harness import register_harness_actions
from .actions_graph import register_graph_actions


def register_all_actions(registry: ActionRegistry) -> None:
    """Register all platform actions with the registry."""
    register_agents_actions(registry)
    register_playbooks_actions(registry)
    register_analytics_actions(registry)
    register_documents_actions(registry)
    register_workspace_actions_defs(registry)
    register_monitoring_actions(registry)
    register_search_actions(registry)
    register_tools_llms_actions(registry)
    register_marketplace_actions(registry)
    register_skills_actions(registry)
    register_assignments_actions(registry)
    register_board_task_actions(registry)
    register_scheduling_actions(registry)
    register_report_actions(registry)
    register_field_actions(registry)
    register_blog_actions(registry)
    register_mission_actions(registry)
    register_analytics_enhanced_actions(registry)
    register_governance_actions(registry)
    register_harness_actions(registry)
    register_graph_actions(registry)

    # Workspace tools (file I/O, grep, exec, git)
    from .workspace_actions import register_workspace_actions
    register_workspace_actions(registry)
