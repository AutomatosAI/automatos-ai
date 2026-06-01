"""PRD-142 Wave 1 · WS-C · W1-S7 — observable startup tasks.

The two boot-time background seeds — agent semantic embedding (PRD-64) and the
shared ``field_memory`` collection bootstrap (PRD-108) — used to swallow their
failures with only a ``logger.warning``. They were launched fire-and-forget
from ``main.py`` and reached their handler only when a real dependency blew up,
so a failed boot seed died silently and never showed up anywhere.

They now also fire ``record_error(subsystem="startup")`` so a failed seed lights
up the ERRORS-by-subsystem dashboard tile (the WS-A sink). These tests prove the
contract:

  - on failure, ``record_error`` fires with ``subsystem="startup"`` and the
    right ``operation`` — and the task itself never raises (boot proceeds);
  - on success, ``record_error`` does NOT fire.
"""
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Importing core.boot.startup_tasks pulls in core.utils.exception_telemetry,
# which imports SessionLocal and thus builds the SQLAlchemy engine — it refuses
# to build without POSTGRES_* creds. These tests never touch a real DB; the
# setdefault means a real .env (when present) still wins.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)


# ---------------------------------------------------------------------------
# embed_all_agents_on_startup (subsystem="startup", operation="embed_all_agents")
# ---------------------------------------------------------------------------

def test_embed_startup_failure_emits_startup_error(monkeypatch):
    import core.boot.startup_tasks as st
    import core.database.database as db_mod

    rec = MagicMock()
    monkeypatch.setattr(st, "record_error", rec, raising=False)

    # SessionLocal() is the first call inside the try → make it raise to drive
    # straight to the terminal except. A distinctive message proves it was THIS
    # failure that was recorded (not an incidental import error).
    def _boom():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(db_mod, "SessionLocal", _boom)

    # Must NOT raise — boot proceeds even when the seed blows up.
    asyncio.run(st.embed_all_agents_on_startup())

    rec.assert_called_once()
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "startup"
    assert kw["operation"] == "embed_all_agents"
    assert "db unavailable" in str(kw["error"])


# ---------------------------------------------------------------------------
# ensure_field_memory_collection (subsystem="startup",
#                                 operation="ensure_field_memory_collection")
# ---------------------------------------------------------------------------

def test_field_memory_startup_failure_emits_startup_error(monkeypatch):
    import core.boot.startup_tasks as st
    import modules.context.factory as factory_mod

    rec = MagicMock()
    monkeypatch.setattr(st, "record_error", rec, raising=False)

    def _boom():
        raise RuntimeError("context factory down")

    monkeypatch.setattr(factory_mod, "get_shared_context", _boom)

    asyncio.run(st.ensure_field_memory_collection())

    rec.assert_called_once()
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "startup"
    assert kw["operation"] == "ensure_field_memory_collection"
    assert "context factory down" in str(kw["error"])


def test_field_memory_success_does_not_emit(monkeypatch):
    """Guardrail: a clean bootstrap must NOT record an error."""
    import core.boot.startup_tasks as st
    import modules.context.factory as factory_mod

    rec = MagicMock()
    monkeypatch.setattr(st, "record_error", rec, raising=False)

    # ctx._inner is a plain MagicMock → the isinstance(VectorFieldSharedContext)
    # guard is False → the body is a clean no-op, no error path taken.
    monkeypatch.setattr(factory_mod, "get_shared_context", lambda: MagicMock())

    asyncio.run(st.ensure_field_memory_collection())

    rec.assert_not_called()
