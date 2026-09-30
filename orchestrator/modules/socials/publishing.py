"""PRD-251 D8 (US-301, S3.3): the publisher — ONE engine for every channel.

A post the person approved (D6) is claimed for publishing by ``publisher.py`` (a
compare-and-set approved | scheduled → publishing), and this runs in the background
(``launch_guarded``):

1. :func:`publish_records.load_work`: nothing runs unless the post is still
   publishing the version it was claimed for and its approval still matches.
2. Each target not yet published, in turn: its steps through Composio
   (``publish_steps.py``), up to ``SOCIALS_MAX_TARGET_ATTEMPTS`` attempts. A
   transient failure (a timeout, a 5xx, a 429, a connection error) tries again after
   ``SOCIALS_PUBLISH_RETRY_BACKOFF_SECONDS`` times the attempt, from the step that
   failed; any other failure ends the target with the platform's message. Every
   attempt is counted on the target (``attempts``), and a published target keeps its
   receipt: remote id, permalink, when.
3. :func:`publish_records.finish`: the post ends published (every target),
   partially_published (some) or failed (none), and the workspace is told
   (``social_post_published`` / ``social_post_failed``).

The whole run lives at most ``SOCIALS_PUBLISH_RUN_MAX_SECONDS``, under the boot
reaper's stale cutoff: a target still running then fails saying so, and one never
tried fails saying that. A run that cannot start still ends the post. A post whose
process died mid-publish is ended by the boot reaper (``core/boot/reaper.py``) or,
sooner, by the leader's reconcile tick (``schedule_jobs.end_lost_publishes``).
The channels differ only in the adapter DATA (``channel_adapters.py``): the engine
never branches on a channel.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Set
from uuid import UUID

from config import config
from modules.socials import notify
from modules.socials.publish_records import (
    PublishJob,
    TargetWork,
    begin_attempt,
    fail_attempt,
    finish,
    load_work,
    record_published,
)
from modules.socials.publish_steps import Runtime, Stager, StepFailure, TargetProgress, run_steps

logger = logging.getLogger(__name__)

TIMED_OUT = "The publish ran out of time before this channel finished. Check the channel, then retry."
UNEXPECTED = "The publish stopped unexpectedly before this channel finished. Check the channel, then retry."


async def publish_target(work: TargetWork, rt: Runtime, factory: Callable[[], Any], held: Set[UUID]) -> None:
    """One target, attempt by attempt, each attempt claimed and recorded; ``held``
    gains the target once this run has claimed it."""
    progress = TargetProgress()
    attempts = max(1, config.SOCIALS_MAX_TARGET_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        if not await asyncio.to_thread(begin_attempt, factory, work.target_id):
            return  # published already, or another run holds it
        held.add(work.target_id)
        if work.error:  # nothing can run for it (a stale plan): it fails, and nothing is called
            await asyncio.to_thread(fail_attempt, factory, work.target_id, work.error, ())
            return
        try:
            remote_id, permalink = await run_steps(work.steps, work.context, rt, progress)
        except StepFailure as failure:
            await asyncio.to_thread(fail_attempt, factory, work.target_id, failure.message, progress.notes)
            if not failure.transient or attempt == attempts:
                return
            logger.info("[Socials] target %s: transient failure, attempt %s of %s", work.target_id, attempt, attempts)
            await rt.sleep(config.SOCIALS_PUBLISH_RETRY_BACKOFF_SECONDS * attempt)
            continue
        await asyncio.to_thread(record_published, factory, work.target_id, remote_id, permalink, progress.notes)
        return


async def _run_targets(work: List[TargetWork], rt: Runtime, factory: Callable[[], Any], held: Set[UUID]) -> None:
    for target in work:
        await publish_target(target, rt, factory, held)


def _default_session_factory() -> Callable[[], Any]:
    from core.database.database import SessionLocal

    return SessionLocal


class _SessionPerCall:
    """The Composio executor with its own session for each call, closed when the call
    ends: a publish talks to the platforms for minutes, and no session (or open
    transaction) is held across the whole run."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory

    async def execute_with_uploads(self, action: str, params: Any, **kwargs: Any) -> Any:
        from core.composio.tool_executor import ComposioToolExecutor

        db = self._factory()
        try:
            return await ComposioToolExecutor(db).execute_with_uploads(action, params, **kwargs)
        finally:
            db.close()


async def _run(job: PublishJob, work: List[TargetWork], rt: Runtime, factory: Callable[[], Any]) -> Optional[str]:
    held: Set[UUID] = set()  # the targets this run claimed: the only ones it may fail
    try:
        await asyncio.wait_for(_run_targets(work, rt, factory, held), timeout=config.SOCIALS_PUBLISH_RUN_MAX_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("[Socials] the publish of post %s ran out of time", job.post_id)
        return await asyncio.to_thread(finish, factory, job, TIMED_OUT, held)
    except Exception:
        logger.exception("[Socials] the publish of post %s failed unexpectedly", job.post_id)
        await asyncio.to_thread(finish, factory, job, UNEXPECTED, held)
        raise
    return await asyncio.to_thread(finish, factory, job, UNEXPECTED, held)


async def run_publish(
    job: PublishJob,
    *,
    executor: Any = None,
    session_factory: Optional[Callable[[], Any]] = None,
    runtime: Optional[Callable[[Any, Stager], Runtime]] = None,
) -> Optional[str]:
    """Publish the post's targets in the background: the status the post ended in,
    or ``None`` when it was not publishing its approved version (nothing ran)."""
    factory = session_factory or _default_session_factory()
    try:
        work = await asyncio.to_thread(load_work, factory, job)
    except Exception:
        # Nothing ran: the run still ends the post, each untried target failing.
        logger.exception("[Socials] the publish of post %s could not start", job.post_id)
        work = []
    if work is None:
        return None
    with tempfile.TemporaryDirectory(prefix="socials-publish-") as workdir:
        stager = Stager(Path(workdir))
        chosen = executor or _SessionPerCall(factory)
        rt = runtime(chosen, stager) if runtime else Runtime(executor=chosen, stager=stager)
        ended = await _run(job, work, rt, factory)
    if ended is not None:
        await notify.dispatch_publish_outcome(job.workspace_id, job.post_id, job.title, ended, session_factory=factory)
    return ended
