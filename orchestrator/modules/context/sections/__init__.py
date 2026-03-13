"""
Context sections package.

Exports all concrete section classes and a ``SECTION_REGISTRY`` mapping
section name strings to their classes.  Used by ``ContextService`` to
instantiate sections from ``ModeConfig.sections`` lists.
"""

from modules.context.sections.base import BaseSection, SectionContext
from modules.context.sections.conversation import ConversationSection
from modules.context.sections.custom import CustomSection
from modules.context.sections.datetime_context import DatetimeContextSection
from modules.context.sections.identity import IdentitySection
from modules.context.sections.memory import MemorySection
from modules.context.sections.platform_actions import PlatformActionsSection
from modules.context.sections.recipe_context import RecipeContextSection
from modules.context.sections.skills import SkillsSection
from modules.context.sections.task_context import TaskContextSection
from modules.context.sections.tools import ToolsSection

# Maps section name strings (as used in ModeConfig.sections) to classes.
SECTION_REGISTRY: dict[str, type[BaseSection]] = {
    "identity": IdentitySection,
    "skills": SkillsSection,
    "platform_actions": PlatformActionsSection,
    "memory": MemorySection,
    "tools": ToolsSection,
    "task_context": TaskContextSection,
    "recipe_context": RecipeContextSection,
    "datetime_context": DatetimeContextSection,
    "conversation": ConversationSection,
    "custom": CustomSection,
}

__all__ = [
    "BaseSection",
    "SectionContext",
    "ConversationSection",
    "CustomSection",
    "DatetimeContextSection",
    "IdentitySection",
    "MemorySection",
    "PlatformActionsSection",
    "RecipeContextSection",
    "SkillsSection",
    "TaskContextSection",
    "ToolsSection",
    "SECTION_REGISTRY",
]
