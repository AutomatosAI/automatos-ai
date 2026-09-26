"""Render jobs: their states, their report, their files, and when they expire.

    preparing -> rejected                     the check found errors; nothing renders (422)
    preparing -> queued -> rendering -> done  (or failed)

A job is an immutable snapshot; every change stores a new one. Its directory
holds the staged composition (``project/``), the spoken lines (``audio/``) and
the render (``out/``). The first two go as soon as the render ends; ``out/``
stays until the job expires (``MEDIA_RENDER_JOB_TTL_SECONDS`` after it
finished), long enough for the orchestrator to copy the file into storage.
"""

from __future__ import annotations

import re
import shutil
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .bundle import Bundle

PREPARING = "preparing"
REJECTED = "rejected"
QUEUED = "queued"
RENDERING = "rendering"
DONE = "done"
FAILED = "failed"
TERMINAL = frozenset({REJECTED, DONE, FAILED})

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
PROJECT_DIR = "project"
AUDIO_DIR = "audio"
OUTPUT_DIR = "out"


@dataclass(frozen=True)
class Job:
    id: str
    bundle: Bundle
    dir: Path
    status: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    report: Mapping[str, Any] = field(default_factory=dict)
    outputs: Tuple[Mapping[str, Any], ...] = ()
    error: Optional[Mapping[str, Any]] = None

    @property
    def workspace_id(self) -> str:
        return self.bundle.workspace_id

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def project_dir(self) -> Path:
        return self.dir / PROJECT_DIR

    @property
    def audio_dir(self) -> Path:
        return self.dir / AUDIO_DIR

    @property
    def output_dir(self) -> Path:
        return self.dir / OUTPUT_DIR


def _iso(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def job_json(job: Job, *, queue_position: Optional[int] = None) -> Dict[str, Any]:
    composition = job.bundle.composition
    body: Dict[str, Any] = {
        "id": job.id,
        "workspace_id": job.workspace_id,
        "reference": job.bundle.reference,
        "status": job.status,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
        "composition": {
            "duration": composition.duration,
            "width": composition.width,
            "height": composition.height,
            "aspect": composition.aspect,
        },
        "outputs": [dict(output) for output in job.outputs],
        "report": dict(job.report),
        "error": dict(job.error) if job.error else None,
    }
    if queue_position is not None:
        body["queue_position"] = queue_position
    return body


def reset_work_dir(work_dir: Path) -> None:
    """Clear what a previous process left behind: only directories named like a job."""
    work_dir.mkdir(parents=True, exist_ok=True)
    for child in work_dir.iterdir():
        if child.is_dir() and JOB_ID.match(child.name):
            shutil.rmtree(child, ignore_errors=True)


class JobStore:
    def __init__(self, work_dir: Path, clock: Callable[[], float] = time.time) -> None:
        self._work_dir = work_dir
        self._clock = clock
        self._jobs: Dict[str, Job] = {}

    def now(self) -> float:
        return self._clock()

    def create(self, bundle: Bundle) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, bundle=bundle, dir=self._work_dir / job_id, status=PREPARING, created_at=self._clock())
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> Job:
        job = replace(self._jobs[job_id], **changes)
        self._jobs[job_id] = job
        return job

    def finish(self, job_id: str, status: str, **changes: Any) -> Job:
        if status not in TERMINAL:
            raise ValueError(f"{status} is not a final state")
        return self.update(job_id, status=status, finished_at=self._clock(), **changes)

    def active_count(self) -> int:
        return sum(1 for job in self._jobs.values() if not job.terminal)

    def expire(self, ttl_seconds: int) -> List[Job]:
        """Forget finished jobs older than the TTL; the caller deletes their files."""
        cutoff = self._clock() - ttl_seconds
        expired = [job for job in self._jobs.values() if job.terminal and (job.finished_at or 0) <= cutoff]
        gone = {job.id for job in expired}
        self._jobs = {job_id: job for job_id, job in self._jobs.items() if job_id not in gone}
        return expired


def delete_files(jobs: List[Job]) -> None:
    for job in jobs:
        shutil.rmtree(job.dir, ignore_errors=True)


def remove_working_files(job: Job, *, keep_outputs: bool) -> None:
    """Drop the staged composition and the spoken lines; the outputs too unless kept."""
    if not keep_outputs:
        shutil.rmtree(job.dir, ignore_errors=True)
        return
    for directory in (job.project_dir, job.audio_dir):
        shutil.rmtree(directory, ignore_errors=True)
