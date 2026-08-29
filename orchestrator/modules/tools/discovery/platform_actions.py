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
from .actions_asks import register_asks_actions  # PRD-225: platform_ask_human
from .actions_clarify import register_clarify_actions  # PRD-229: ask_orchestrator
from .actions_harness import register_harness_actions
from .actions_graph import register_graph_actions
from .actions_codegraph import register_codegraph_actions  # PRD-165 S4
from .actions_shopify import register_shopify_actions  # PRD-183 S3 (F088)
from .actions_auto_reporting import register_auto_reporting_actions
from .actions_notifications import register_notifications_actions
from .actions_routing import register_routing_actions  # PRD-142 Wave 4 (W4-S6)
from .actions_channels import register_channels_actions  # PRD-143 S10
from .actions_widgets import register_widgets_actions  # PRD-143 S10
from .actions_members import register_members_actions  # PRD-143 S11
from .actions_api_keys import register_api_keys_actions  # PRD-143 S11
from .actions_power import register_power_actions  # PRD-142 Wave 4 (W4-S5)
from .actions_autonomy import register_autonomy_actions
from .actions_deliverables import register_deliverables_actions  # PRD-164 S3
from .actions_watches import register_watch_actions  # PRD-204 S9
from .actions_fleet import register_fleet_actions  # PRD-228: platform_fleet_status
from .actions_capabilities import register_capabilities_actions  # tool-surface PR-B
from .actions_onboarding import register_onboarding_actions  # PRD-222 W1S3
from .actions_intake import register_intake_actions  # PRD-222 W1S8
from .actions_packages import register_package_actions  # PRD-230 US-006


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
    register_asks_actions(registry)  # PRD-225: platform_ask_human
    register_clarify_actions(registry)  # PRD-229: ask_orchestrator (mid-run clarification)
    register_harness_actions(registry)
    register_graph_actions(registry)
    register_codegraph_actions(registry)  # PRD-165 S4: codegraph as an agent capability
    register_shopify_actions(registry)  # PRD-183 S3 (F088): Shopify sync + freshness tools
    register_auto_reporting_actions(registry)
    register_notifications_actions(registry)
    register_routing_actions(registry)  # PRD-142 Wave 4 (W4-S6): routing-rule tool
    register_channels_actions(registry)  # PRD-143 S10: channel connect/configure surface
    register_widgets_actions(registry)  # PRD-143 S10: widget-config surface
    register_members_actions(registry)  # PRD-143 S11: member administration surface
    register_api_keys_actions(registry)  # PRD-143 S11: SDK API-key administration surface
    register_power_actions(registry)  # PRD-142 Wave 4 (W4-S5): power-mode tool
    register_autonomy_actions(registry)
    register_deliverables_actions(registry)  # PRD-164 S3: deliverable list/get tools
    register_watch_actions(registry)  # PRD-204 S9: watch create/list/get/cancel
    register_fleet_actions(registry)  # PRD-228: platform_fleet_status (live floor read)
    register_capabilities_actions(registry)  # PR-B: platform_find_tools discovery seam
    register_onboarding_actions(registry)  # PRD-222 W1S3: platform_update_onboarding
    register_intake_actions(registry)  # PRD-222 W1S8: platform_scan_business_site + status
    register_package_actions(registry)  # PRD-230 US-006: package search/install tools

    # Workspace tools (file I/O, grep, exec, git)
    from .workspace_actions import register_workspace_actions
    register_workspace_actions(registry)
