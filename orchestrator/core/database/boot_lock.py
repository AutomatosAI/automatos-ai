"""
Single-worker bootstrap guard using PostgreSQL advisory locks.

On startup, 4 uvicorn workers all race to run seed operations.
Only the worker that acquires the advisory lock runs seeds;
the rest skip immediately.

The lock is session-scoped — it auto-releases when the connection closes.
"""

import logging
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Unique lock ID — must not collide with other advisory locks in the system.
# 0xB007 = "BOOT" in hex-speak.
BOOT_LOCK_ID = 47111  # arbitrary fixed integer


@contextmanager
def boot_leader_lock(engine: Engine):
    """Context manager that yields True if this worker is the boot leader.

    Usage:
        with boot_leader_lock(engine) as is_leader:
            if is_leader:
                run_seeds()
            else:
                logger.info("Another worker is seeding, skipping")
    """
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT pg_try_advisory_lock(:lock_id)"),
            {"lock_id": BOOT_LOCK_ID},
        )
        acquired = result.scalar()

        if acquired:
            logger.info("Boot lock acquired — this worker will run seeds")
        else:
            logger.info("Boot lock held by another worker — skipping seeds")

        try:
            yield bool(acquired)
        finally:
            if acquired:
                conn.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"),
                    {"lock_id": BOOT_LOCK_ID},
                )
