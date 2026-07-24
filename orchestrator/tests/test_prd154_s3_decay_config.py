"""PRD-154 S3 — L2 decay is retuned to week-scale via config.

Root cause (reports/PLATFORM_DEEP_REVIEW_2026-06.md §2.2): MEMORY_DECAY_RATE=0.1/hr
with archive threshold 0.3 archived an importance-0.8 memory in ~15h. Pilot
memories evaporated overnight.

This guards the REAL config default (the mock-config decay tests in
test_unified_memory.py cannot — they hardcode the rate). The decay formula is
   retention = exp(-rate*hours) * (1 + 0.5*importance + 0.1*min(access,10))
archived when retention < threshold.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402


def _retention(hours: float, importance: float, access_count: int, rate: float) -> float:
    return math.exp(-rate * hours) * (1.0 + 0.5 * importance + 0.1 * min(access_count, 10))


def test_importance_08_survives_seven_days_under_real_config():
    rate = config.MEMORY_DECAY_RATE
    threshold = config.MEMORY_DECAY_ARCHIVE_THRESHOLD
    one_week_hours = 7 * 24

    survived = _retention(one_week_hours, importance=0.8, access_count=0, rate=rate)
    assert survived >= threshold, (
        f"importance-0.8 archived before 7 days under MEMORY_DECAY_RATE={rate}"
    )

    # Regression proof: the old 0.1/hr rate archived it long before a week.
    old = _retention(one_week_hours, importance=0.8, access_count=0, rate=0.1)
    assert old < threshold
