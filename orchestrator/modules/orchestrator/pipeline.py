"""
Workflow Pipeline — Composable Stage Executor
==============================================

PRD-59: Executes a dynamic subset of stages determined by PhaseSelector.
Replaces the monolithic execute_workflow_with_progress() with a composable
pipeline that can run any combination of phases.

Each stage is a callable with a standard interface:
  async def stage_fn(ctx: WorkflowContext) -> StageResult

The pipeline:
  1. Receives a list of PhaseSpec from PhaseSelector
  2. Iterates through phases and their stages
  3. Calls each stage function, passing accumulated context
  4. Emits SSE events for progress tracking
  5. Handles errors (skip or abort)
"""

import logging
import asyncio
from typing import (
    Any, Callable, Awaitable, Dict, List, Optional, Protocol
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from modules.orchestrator.phase_selector import PhaseSpec, Phase, StageSpec

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ErrorStrategy(Enum):
    """What to do when a stage fails"""
    ABORT = "abort"           # Stop the pipeline
    SKIP = "skip"             # Skip this stage, continue
    RETRY = "retry"           # Retry once, then skip


@dataclass
class StageResult:
    """Result from a single stage execution"""
    stage_name: str
    status: StageStatus
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: int = 0
    tokens_used: int = 0


@dataclass
class WorkflowContext:
    """
    Shared context passed through the pipeline.
    Each stage reads from and writes to this context.
    """
    # Workflow metadata
    workflow_id: Optional[int] = None
    execution_id: Optional[int] = None
    workspace_id: Optional[int] = None
    task_description: str = ""

    # Stage results accumulate here
    decomposition: Optional[Dict[str, Any]] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    agent_assignments: Optional[Dict[str, Any]] = None
    context_enhancements: Optional[Dict[str, Any]] = None
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    aggregated_results: Optional[Any] = None
    quality_assessment: Optional[Any] = None
    learning_updates: Optional[Dict[str, Any]] = None
    memory_results: Optional[Dict[str, Any]] = None
    final_response: Optional[Dict[str, Any]] = None

    # Infrastructure references
    db: Optional[Any] = None
    execution: Optional[Any] = None  # WorkflowExecution ORM object
    mem0_client: Optional[Any] = None
    stage_tracker: Optional[Any] = None

    # Tracking
    stage_results: List[StageResult] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access for backward compatibility"""
        return getattr(self, key, default)

    def set(self, key: str, value: Any) -> None:
        """Dict-like setter"""
        setattr(self, key, value)


# Type for stage functions
StageFunction = Callable[[WorkflowContext], Awaitable[StageResult]]


class StageProgressCallback(Protocol):
    """Protocol for SSE progress updates"""
    async def on_phase_start(self, phase: Phase, total_phases: int, current_index: int) -> None: ...
    async def on_stage_start(self, stage: StageSpec, phase: Phase) -> None: ...
    async def on_stage_complete(self, stage: StageSpec, result: StageResult) -> None: ...
    async def on_stage_skip(self, stage: StageSpec, reason: str) -> None: ...
    async def on_phase_complete(self, phase: Phase) -> None: ...


class DefaultProgressCallback:
    """Default progress callback that just logs"""
    async def on_phase_start(self, phase: Phase, total_phases: int, current_index: int) -> None:
        logger.info(f"▶️ Phase {current_index + 1}/{total_phases}: {phase.value}")

    async def on_stage_start(self, stage: StageSpec, phase: Phase) -> None:
        logger.info(f"  ▶️ Stage {stage.stage_number}: {stage.stage_name}")

    async def on_stage_complete(self, stage: StageSpec, result: StageResult) -> None:
        logger.info(f"  ✅ Stage {stage.stage_number}: {result.status.value} ({result.duration_ms}ms)")

    async def on_stage_skip(self, stage: StageSpec, reason: str) -> None:
        logger.info(f"  ⏭️ Stage {stage.stage_number}: skipped ({reason})")

    async def on_phase_complete(self, phase: Phase) -> None:
        logger.info(f"✅ Phase {phase.value} complete")


class WorkflowPipeline:
    """
    Composable pipeline that executes phases and stages in order.

    Usage:
        pipeline = WorkflowPipeline()
        pipeline.register_stage("Task Decomposition", decompose_fn)
        pipeline.register_stage("Agent Selection", select_fn)
        ...
        results = await pipeline.execute(phases, context)
    """

    def __init__(
        self,
        error_strategy: ErrorStrategy = ErrorStrategy.SKIP,
        progress_callback: Optional[StageProgressCallback] = None,
    ):
        self.error_strategy = error_strategy
        self.progress = progress_callback or DefaultProgressCallback()
        self._stage_registry: Dict[str, StageFunction] = {}

    def register_stage(self, stage_name: str, fn: StageFunction) -> None:
        """Register a stage function by name"""
        self._stage_registry[stage_name] = fn

    async def execute(
        self,
        phases: List[PhaseSpec],
        context: WorkflowContext,
        dry_run: bool = False,
    ) -> WorkflowContext:
        """
        Execute the pipeline with the given phases.

        Args:
            phases: Ordered list of PhaseSpec from PhaseSelector
            context: Shared workflow context
            dry_run: If True, log planned stages without executing

        Returns:
            Updated WorkflowContext with all stage results
        """
        total_phases = len(phases)
        total_stages = sum(p.stage_count for p in phases)

        logger.info(
            f"🚀 Pipeline starting: {total_phases} phases, {total_stages} stages"
            f"{' (DRY RUN)' if dry_run else ''}"
        )

        for phase_idx, phase_spec in enumerate(phases):
            await self.progress.on_phase_start(
                phase_spec.phase, total_phases, phase_idx
            )

            for stage in phase_spec.stages:
                if dry_run:
                    logger.info(f"  [DRY RUN] Would execute: {stage.stage_name}")
                    continue

                # Look up stage function
                stage_fn = self._stage_registry.get(stage.stage_name)
                if stage_fn is None:
                    if stage.is_optional:
                        await self.progress.on_stage_skip(
                            stage, "not registered (optional)"
                        )
                        continue
                    else:
                        logger.warning(
                            f"Stage '{stage.stage_name}' not registered, skipping"
                        )
                        await self.progress.on_stage_skip(
                            stage, "not registered"
                        )
                        continue

                # Execute stage
                await self.progress.on_stage_start(stage, phase_spec.phase)
                start_time = datetime.now()

                try:
                    result = await stage_fn(context)
                    result.duration_ms = int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    )
                    context.stage_results.append(result)
                    await self.progress.on_stage_complete(stage, result)

                except Exception as e:
                    duration = int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    )
                    error_result = StageResult(
                        stage_name=stage.stage_name,
                        status=StageStatus.FAILED,
                        error=str(e),
                        duration_ms=duration,
                    )
                    context.stage_results.append(error_result)

                    logger.error(
                        f"Stage '{stage.stage_name}' failed: {e}",
                        exc_info=True,
                    )

                    if self.error_strategy == ErrorStrategy.ABORT:
                        raise
                    elif self.error_strategy == ErrorStrategy.RETRY:
                        # One retry attempt
                        try:
                            logger.info(f"  🔄 Retrying {stage.stage_name}...")
                            result = await stage_fn(context)
                            result.duration_ms = int(
                                (datetime.now() - start_time).total_seconds() * 1000
                            )
                            context.stage_results.append(result)
                            await self.progress.on_stage_complete(stage, result)
                        except Exception as retry_err:
                            logger.error(f"Retry failed: {retry_err}")
                            await self.progress.on_stage_skip(
                                stage, f"failed after retry: {retry_err}"
                            )
                    else:
                        # SKIP strategy
                        await self.progress.on_stage_skip(
                            stage, f"failed: {e}"
                        )

            await self.progress.on_phase_complete(phase_spec.phase)

        completed = sum(
            1 for r in context.stage_results if r.status == StageStatus.COMPLETED
        )
        failed = sum(
            1 for r in context.stage_results if r.status == StageStatus.FAILED
        )
        total_ms = sum(r.duration_ms for r in context.stage_results)

        logger.info(
            f"🏁 Pipeline complete: {completed}/{total_stages} stages completed, "
            f"{failed} failed, {total_ms}ms total"
        )

        return context

    def get_execution_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Generate execution summary from context"""
        return {
            "total_stages": len(context.stage_results),
            "completed": sum(
                1 for r in context.stage_results if r.status == StageStatus.COMPLETED
            ),
            "failed": sum(
                1 for r in context.stage_results if r.status == StageStatus.FAILED
            ),
            "skipped": sum(
                1 for r in context.stage_results if r.status == StageStatus.SKIPPED
            ),
            "total_duration_ms": sum(r.duration_ms for r in context.stage_results),
            "total_tokens": sum(r.tokens_used for r in context.stage_results),
            "stages": [
                {
                    "name": r.stage_name,
                    "status": r.status.value,
                    "duration_ms": r.duration_ms,
                    "tokens_used": r.tokens_used,
                    "error": r.error,
                }
                for r in context.stage_results
            ],
        }
