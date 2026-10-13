"""
P256-FIX-N14-1 (F416): the nightly EdgeBuilder runs in a child process.

Night 14 (12–13 Oct, F416): EdgeBuilderScheduler._run_edge_builder started at 03:00
UTC, never logged completion, and the backend container was OOMKilled at 03:50
while chats still flowed. N13-2 had moved the work off the event loop, but not out
of the backend's memory.

build_edges_in_worker runs one build_edges in a fresh (spawned) child process:

* the child caps its own address space at config.EDGE_BUILDER_WORKER_MEMORY_MB
  (resource.setrlimit RLIMIT_AS; 0 = no cap), so running past it raises
  MemoryError inside the child;
* the child raises its own oom_score_adj to the maximum, so if the container's
  memory fills first the kernel ends the child, not the backend;
* a child that raises, or dies, leaves the caller alive: one error is logged and a
  failed EdgeBuildSummary comes back. A fresh child runs each night, so the next
  night runs as usual.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import timedelta
from typing import Callable, Optional

from core.services.edge_builder import EdgeBuildSummary, build_edges

logger = logging.getLogger(__name__)

_DEFAULT_WORKER_MEMORY_MB = 4096  # config.EDGE_BUILDER_WORKER_MEMORY_MB overrides
_BYTES_PER_MB = 1024 * 1024
_OOM_SCORE_ADJ_PATH = "/proc/self/oom_score_adj"
_OOM_SCORE_ADJ_FIRST_VICTIM = "1000"  # the kernel's first pick when memory runs out
_START_METHOD = "spawn"  # a fresh interpreter: never fork the threaded backend

# The child's entry: (window in seconds, workspace id) -> EdgeBuildSummary.
WorkerTarget = Callable[[float, Optional[str]], EdgeBuildSummary]


def worker_memory_bytes() -> int:
    """The child's address-space ceiling in bytes, from config (0 = no ceiling)."""
    from config import config

    mb = int(getattr(config, "EDGE_BUILDER_WORKER_MEMORY_MB", _DEFAULT_WORKER_MEMORY_MB))
    return max(0, mb) * _BYTES_PER_MB


def limit_worker_memory(ceiling_bytes: int) -> None:
    """The child's initializer: make it the first OOM victim and cap its memory."""
    _prefer_as_oom_victim()
    if ceiling_bytes > 0:
        _cap_address_space(ceiling_bytes)


def _prefer_as_oom_victim() -> None:
    try:
        with open(_OOM_SCORE_ADJ_PATH, "w", encoding="ascii") as handle:
            handle.write(_OOM_SCORE_ADJ_FIRST_VICTIM)
    except OSError:
        logger.warning(
            "EdgeBuilder worker: could not raise its oom_score_adj (not Linux?); "
            "the memory ceiling alone bounds it", exc_info=True,
        )


def _cap_address_space(ceiling_bytes: int) -> None:
    try:
        import resource

        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            ceiling_bytes = min(ceiling_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (ceiling_bytes, hard))
    except (ImportError, ValueError, OSError):
        logger.warning(
            "EdgeBuilder worker: could not cap its address space at %d bytes; "
            "running uncapped", ceiling_bytes, exc_info=True,
        )


def run_build_edges(window_seconds: float, workspace_id: Optional[str]) -> EdgeBuildSummary:
    """The child's entry: one build_edges run on the child's own event loop."""
    window = timedelta(seconds=window_seconds)
    return asyncio.run(build_edges(window=window, workspace_id=workspace_id))


async def build_edges_in_worker(
    window: timedelta,
    workspace_id: Optional[str] = None,
    target: WorkerTarget = run_build_edges,
) -> EdgeBuildSummary:
    """Run ``target`` (build_edges by default) in a memory-capped child process.

    The caller's loop only awaits the child. A child that raises (MemoryError
    included) or dies (BrokenProcessPool: the kernel killed it) ends the run
    alone: one error is logged and a failed summary returned.
    """
    pool = ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context(_START_METHOD),
        initializer=limit_worker_memory,
        initargs=(worker_memory_bytes(),),
    )
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(pool, target, window.total_seconds(), workspace_id)
    except (MemoryError, BrokenProcessPool) as exc:
        logger.error(
            "[EdgeBuilder] Nightly run ended in its worker (%s: out of memory, or "
            "the worker was killed); the backend is unaffected and the next night "
            "runs as usual",
            type(exc).__name__,
        )
        return EdgeBuildSummary(failed=True, error=type(exc).__name__)
    except Exception as exc:
        logger.exception("[EdgeBuilder] Nightly run failed in its worker")
        return EdgeBuildSummary(failed=True, error=type(exc).__name__)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
