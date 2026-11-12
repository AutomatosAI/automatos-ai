"""The durable job queue (PRD-276): one way to say "do this, later or now, and don't lose it".

``enqueue`` (in the caller's transaction) and ``@job_handler`` are the only way in; handlers never
touch the backend. See ``registry`` (kinds), ``queue`` (the database operations), ``worker`` (the
loop) and ``erasure`` (the purge's barrier).
"""
from core.jobs.queue import Job, complete, enqueue, enqueue_now, fence
from core.jobs.registry import job_handler

__all__ = ["Job", "complete", "enqueue", "enqueue_now", "fence", "job_handler"]
