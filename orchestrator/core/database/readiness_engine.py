"""The readiness check's own connection to the database (PRD-282 US-201, #1651).

GET /health/ready reads a SELECT 1 that a background watch makes (core/health_readiness.py).
On the main engine that SELECT 1 had no libpq timeouts: after a failover or a partition that
sends no TCP reset, the check's thread waited on the kernel's own timeouts (about 2 min to
connect, about 15 min on an open connection), and as the watch runs one check thread at a
time, readiness stayed 503 long after the database was back.

So the check runs on an engine of its own:

* **The main engine's address and TLS.** ``database.DATABASE_URL`` (production's default
  ``sslmode=require`` included) through ``create_app_engine``, which adds PRD-278's
  ``DATABASE_SSLMODE`` / ``DATABASE_SSLROOTCERT`` and, for a mode that checks a certificate,
  ``gssencmode=disable``, exactly as for the main engine.
* **No pool** (``NullPool``): every check opens a fresh connection, so a connection the
  network silently dropped is never reused, and the check never takes a slot in the app's pool.
* **libpq timeouts set from the check's time limit** (``readiness_connect_args``):
  ``connect_timeout`` ends a connect that gets no answer, ``tcp_user_timeout`` a query whose
  bytes are never acknowledged, and keepalives an open connection that has gone silent. So a
  hung check's thread ends about when its time limit does, and the next tick checks again.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict

from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from config import config

MS_PER_SECOND = 1000
# libpq takes connect_timeout and the keepalive settings in whole seconds.
MIN_WHOLE_SECONDS = 1
# Keepalive probes start after this much silence, one this often: the count, set from the
# time limit, ends a silent connection about when the check's time limit does.
KEEPALIVE_IDLE_S = 1
KEEPALIVE_INTERVAL_S = 1


def readiness_connect_args(limit_s: float) -> Dict[str, int]:
    """The libpq keywords that end a hung readiness check near ``limit_s`` seconds."""
    whole_s = max(MIN_WHOLE_SECONDS, math.ceil(limit_s))
    return {
        "connect_timeout": whole_s,
        "tcp_user_timeout": max(1, round(limit_s * MS_PER_SECOND)),  # milliseconds
        "keepalives": 1,
        "keepalives_idle": KEEPALIVE_IDLE_S,
        "keepalives_interval": KEEPALIVE_INTERVAL_S,
        "keepalives_count": whole_s,
    }


@lru_cache(maxsize=1)
def readiness_engine() -> Engine:
    """The readiness check's engine, built on first use (it connects only when a check runs)."""
    from core.database.database import DATABASE_URL
    from core.database.tls import create_app_engine

    return create_app_engine(DATABASE_URL, poolclass=NullPool,
                             connect_args=readiness_connect_args(config.HEALTH_READY_CHECK_TIMEOUT_SECONDS))
