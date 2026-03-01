"""
Task Runner Factory
===================
PRD-56: Infrastructure Scaling & Ephemeral Agent Compute

Returns the correct TaskRunner implementation based on environment config.
Singleton per process — one runner shared across all request handlers.

Configuration:
  TASK_RUNNER_BACKEND=local       → LocalTaskRunner (default, Phase 1)
  TASK_RUNNER_BACKEND=queued      → QueuedTaskRunner (Phase 2)
  TASK_RUNNER_BACKEND=kubernetes  → KubernetesTaskRunner (Phase 3)
"""

from __future__ import annotations

import logging
from typing import Optional

from config import config

from .base import TaskRunner
from .local import LocalTaskRunner

logger = logging.getLogger(__name__)

# Module-level singleton
_runner_instance: Optional[TaskRunner] = None


def get_task_runner() -> TaskRunner:
    """Get or create the singleton TaskRunner.

    The backend is selected via the TASK_RUNNER_BACKEND env var.
    Defaults to 'local' (in-process asyncio execution).

    Returns:
        The configured TaskRunner instance.

    Raises:
        ValueError: If TASK_RUNNER_BACKEND is set to an unknown value.
    """
    global _runner_instance

    if _runner_instance is not None:
        return _runner_instance

    backend = (config.TASK_RUNNER_BACKEND or "local").lower().strip()

    if backend == "local":
        _runner_instance = LocalTaskRunner()
        logger.info("TaskRunner initialized: LocalTaskRunner (in-process asyncio)")

    elif backend == "queued":
        # Phase 2: Redis queue + worker containers
        try:
            from .queued import QueuedTaskRunner
            _runner_instance = QueuedTaskRunner()
            logger.info("TaskRunner initialized: QueuedTaskRunner (Redis + workers)")
        except ImportError:
            logger.warning(
                "TASK_RUNNER_BACKEND=queued but QueuedTaskRunner not available. "
                "Falling back to LocalTaskRunner. Install Phase 2 dependencies."
            )
            _runner_instance = LocalTaskRunner()

    elif backend == "kubernetes":
        # Phase 3: K8s Jobs with ephemeral pods
        try:
            from .kubernetes import KubernetesTaskRunner
            _runner_instance = KubernetesTaskRunner()
            logger.info("TaskRunner initialized: KubernetesTaskRunner (K8s Jobs)")
        except ImportError:
            logger.warning(
                "TASK_RUNNER_BACKEND=kubernetes but KubernetesTaskRunner not available. "
                "Falling back to LocalTaskRunner. Install Phase 3 dependencies."
            )
            _runner_instance = LocalTaskRunner()

    else:
        raise ValueError(
            f"Unknown TASK_RUNNER_BACKEND: '{backend}'. "
            f"Valid options: local, queued, kubernetes"
        )

    return _runner_instance


def reset_task_runner() -> None:
    """Reset the singleton. Useful for testing."""
    global _runner_instance
    _runner_instance = None
