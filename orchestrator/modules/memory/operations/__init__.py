"""Memory Operations — contradiction-based consolidation (PRD-159 S4).

The retired manager's satellites that used to live here (augmentation,
access_patterns, consolidation, execution_history, prompt_injection) were
deleted with that stack in PRD-187 S5 — ``contradiction`` is the one live
operation, driven by ``UnifiedMemoryService.run_sleep_time_consolidation``.
"""
from .contradiction import plan_consolidation

__all__ = ["plan_consolidation"]
