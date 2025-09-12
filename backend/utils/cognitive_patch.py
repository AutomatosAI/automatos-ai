"""
Legacy cognitive patch methods (safe no-op).

This module previously contained class-bound methods intended to be
manually inserted into `EnhancedTwoTierOrchestrator`. It caused import-time
syntax errors. It is now a safe, importable placeholder to avoid breaking
tooling or application startup. If you need these utilities, wire them
explicitly in the orchestrator and write proper tests.
"""

from __future__ import annotations

from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


async def cognitive_task_breakdown(task_prompt: str) -> List[Dict[str, Any]]:
    """Return an empty task list by default.

    This placeholder avoids accidental side-effects at import time.
    Replace with a concrete implementation and explicit wiring where needed.
    """
    try:
        logger.info("cognitive_task_breakdown invoked (no-op placeholder)")
        return []
    except Exception as exc:  # pragma: no cover
        logger.error("cognitive_task_breakdown failed: %s", exc)
        return []


async def cognitive_content_generation(task: Dict[str, Any], project_path: str) -> str:
    """Return an empty string by default.

    This placeholder avoids accidental side-effects at import time.
    Replace with a concrete implementation and explicit wiring where needed.
    """
    try:
        logger.info("cognitive_content_generation invoked (no-op placeholder)")
        return ""
    except Exception as exc:  # pragma: no cover
        logger.error("cognitive_content_generation failed: %s", exc)
        return ""
