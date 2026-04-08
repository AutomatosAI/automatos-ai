"""
Context sections package.

Exports all concrete section classes and a ``SECTION_REGISTRY`` mapping
section name strings to their classes.  Used by ``ContextService`` to
instantiate sections from ``ModeConfig.sections`` lists.
"""

from modules.context.sections.agent_roster import AgentRosterSection
from modules.context.sections.base import BaseSection, SectionContext
from modules.context.sections.composio import ComposioSection
from modules.context.sections.conversation import ConversationSection
from modules.context.sections.custom import CustomSection
from modules.context.sections.datetime_context import DatetimeContextSection
from modules.context.sections.graph_context import GraphSection
from modules.context.sections.identity import IdentitySection
from modules.context.sections.memory import MemorySection
from modules.context.sections.mission_context import MissionContextSection
from modules.context.sections.onboarding import OnboardingSection
from modules.context.sections.platform_actions import PlatformActionsSection
from modules.context.sections.plugins import PluginsSection
from modules.context.sections.playbook_context import PlaybookContextSection
from modules.context.sections.skills import SkillsSection
from modules.context.sections.task_context import TaskContextSection
from modules.context.sections.tools import ToolsSection

# Maps section name strings (as used in ModeConfig.sections) to classes.
SECTION_REGISTRY: dict[str, type[BaseSection]] = {
    "identity": IdentitySection,
    "skills": SkillsSection,
    "composio": ComposioSection,
    "plugins": PluginsSection,
    "platform_actions": PlatformActionsSection,
    "memory": MemorySection,
    "mission_context": MissionContextSection,
    "onboarding": OnboardingSection,
    "agent_roster": AgentRosterSection,
    "tools": ToolsSection,
    "task_context": TaskContextSection,
    "playbook_context": PlaybookContextSection,
    "datetime_context": DatetimeContextSection,
    "business_graph": GraphSection,
    "conversation": ConversationSection,
    "custom": CustomSection,
}

__all__ = [
    "AgentRosterSection",
    "BaseSection",
    "SectionContext",
    "ComposioSection",
    "ConversationSection",
    "CustomSection",
    "DatetimeContextSection",
    "GraphSection",
    "IdentitySection",
    "MemorySection",
    "MissionContextSection",
    "OnboardingSection",
    "PlatformActionsSection",
    "PluginsSection",
    "PlaybookContextSection",
    "SkillsSection",
    "TaskContextSection",
    "ToolsSection",
    "SECTION_REGISTRY",
]
