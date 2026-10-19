"""The agent-opt worker's own settings (PRD-262 US-006).

The worker is a separate container, so it reads its environment here, as the
workspace worker does in ``worker_config.py`` (AGENTS.md: each service keeps its
settings in its own config module).
"""
from __future__ import annotations

import os

# The shared secret the API sends as X-Internal-Token. The API reads the same name
# (orchestrator/config.py AGENT_OPT_WORKER_TOKEN). The worker refuses to start
# without it: every route but /health and /metrics spends FutureAGI or OpenAI quota.
TOKEN_ENV = "AGENT_OPT_WORKER_TOKEN"

MAX_JOBS_ENV = "AGENT_OPT_MAX_CONCURRENT_JOBS"
DEFAULT_MAX_CONCURRENT_JOBS = 2

# What the platform sends today (orchestrator/core/services/futureagi_service.py,
# optimize_prompt): gpt-4o-mini as the teacher, and at most 10 dataset examples.
ALLOWED_TEACHER_MODELS = frozenset({"gpt-4o-mini"})
MAX_DATASET_ITEMS = 10


def internal_token() -> str:
    """The token every protected route requires. Unset or blank stops the start-up."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            f"{TOKEN_ENV} is not set: agent-opt-worker refuses to start without the token "
            "the API sends as X-Internal-Token (PRD-262 US-006)."
        )
    return token


def max_concurrent_jobs() -> int:
    """How many /optimize jobs may run at once (at least 1)."""
    raw = os.environ.get(MAX_JOBS_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_CONCURRENT_JOBS
    value = int(raw)
    if value < 1:
        raise RuntimeError(f"{MAX_JOBS_ENV} must be at least 1, not {value}")
    return value
