"""
Unified Context Service — single entry point for building LLM context.

Usage:
    from modules.context import ContextService, ContextResult, ContextMode

    context = await ContextService(db).build_context(
        mode=ContextMode.TASK_EXECUTION,
        agent=agent,
        workspace_id=workspace_id,
    )
"""

from modules.context.modes import MODE_CONFIGS, ContextMode, ModeConfig
from modules.context.planning import PlanningContextPack
from modules.context.result import ContextResult
from modules.context.service import ContextService

__all__ = [
    "ContextService",
    "ContextResult",
    "ContextMode",
    "ModeConfig",
    "MODE_CONFIGS",
    "PlanningContextPack",
]
