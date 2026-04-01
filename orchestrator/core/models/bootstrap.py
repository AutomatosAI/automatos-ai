"""
Bootstrap Models — PRD-123 Patterns #2 and #10
===============================================

Dataclasses for two-phase trust-gated startup and named bootstrap stages
with timing instrumentation.

Pattern #2: Trust-Gated Init — DeferredInitResult tracks extension health.
Pattern #10: Named Bootstrap Stages — BootstrapStage enum + StageResult + BootstrapReport.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern #10: Named Bootstrap Stages
# ---------------------------------------------------------------------------


class BootstrapStage(str, Enum):
    """Named stages of the application bootstrap sequence."""

    CONFIG_LOAD = "config_load"
    DATABASE_INIT = "database_init"
    SCHEMA_MIGRATION = "schema_migration"
    SEED_SYSTEM_DATA = "seed_system_data"
    TRUST_GATE = "trust_gate"
    SKILL_SOURCES = "skill_sources"
    TOOL_SYNC = "tool_sync"
    SEMANTIC_EMBEDDINGS = "semantic_embeddings"
    DASHBOARD_INIT = "dashboard_init"
    SCHEDULER_INIT = "scheduler_init"
    CHANNEL_CONNECT = "channel_connect"
    READY = "ready"


@dataclass(frozen=True)
class StageResult:
    """Immutable record of a single bootstrap stage execution."""

    stage: BootstrapStage
    status: str  # 'success' | 'failed' | 'skipped'
    duration_ms: int
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class BootstrapReport:
    """Mutable collection of stage results built up during startup."""

    stages: list[StageResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    ready_at: Optional[datetime] = None

    @property
    def total_duration_ms(self) -> int:
        if self.started_at is None:
            return 0
        end = self.ready_at or datetime.now(timezone.utc)
        return int((end - self.started_at).total_seconds() * 1000)

    @property
    def failed_stages(self) -> list[StageResult]:
        return [s for s in self.stages if s.status == "failed"]

    def as_dict(self) -> dict:
        return {
            "stages": [s.as_dict() for s in self.stages],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ready_at": self.ready_at.isoformat() if self.ready_at else None,
            "total_duration_ms": self.total_duration_ms,
            "failed_stages": [s.stage.value for s in self.failed_stages],
        }


async def run_stage(
    report: BootstrapReport,
    stage: BootstrapStage,
    func: Callable,
    *,
    skip_condition: bool = False,
) -> StageResult:
    """
    Execute a bootstrap stage with timing and error capture.

    Args:
        report: The BootstrapReport to append the result to.
        stage: Which BootstrapStage this is.
        func: Async or sync callable to execute.
        skip_condition: If True, skip execution and record as 'skipped'.

    Returns:
        The StageResult for this stage.
    """
    if skip_condition:
        result = StageResult(stage=stage, status="skipped", duration_ms=0)
        report.stages.append(result)
        logger.info("Bootstrap [%s] skipped", stage.value)
        return result

    start = time.monotonic()
    try:
        import asyncio

        if asyncio.iscoroutinefunction(func):
            await func()
        else:
            func()
        elapsed_ms = int((time.monotonic() - start) * 1000)
        result = StageResult(stage=stage, status="success", duration_ms=elapsed_ms)
        logger.info("Bootstrap [%s] completed in %dms", stage.value, elapsed_ms)
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        result = StageResult(
            stage=stage, status="failed", duration_ms=elapsed_ms, error=str(exc)
        )
        logger.error(
            "Bootstrap [%s] failed after %dms: %s", stage.value, elapsed_ms, exc
        )

    report.stages.append(result)
    return result


# ---------------------------------------------------------------------------
# Pattern #2: Trust-Gated Init — Deferred extension results
# ---------------------------------------------------------------------------


@dataclass
class DeferredInitResult:
    """Tracks which extensions loaded successfully in Phase 2."""

    skills_loaded: bool = False
    tools_synced: bool = False
    channels_connected: bool = False
    scheduler_started: bool = False
    dashboard_initialized: bool = False

    @property
    def all_healthy(self) -> bool:
        return all([
            self.skills_loaded,
            self.tools_synced,
            self.channels_connected,
            self.scheduler_started,
            self.dashboard_initialized,
        ])

    def as_dict(self) -> dict:
        return {
            "skills_loaded": self.skills_loaded,
            "tools_synced": self.tools_synced,
            "channels_connected": self.channels_connected,
            "scheduler_started": self.scheduler_started,
            "dashboard_initialized": self.dashboard_initialized,
            "all_healthy": self.all_healthy,
        }
