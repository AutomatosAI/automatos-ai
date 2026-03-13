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

from modules.context.result import ContextResult
from modules.context.modes import ContextMode, ModeConfig, MODE_CONFIGS

__all__ = [
    "ContextResult",
    "ContextMode",
    "ModeConfig",
    "MODE_CONFIGS",
]
