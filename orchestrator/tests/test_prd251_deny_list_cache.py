"""PRD-251 S0.6 — the Composio deny list is cached, and never read on the event loop.

Before this, every Composio call read ``composio.denied_actions`` with a
synchronous ``SessionLocal`` query, inline in ``async def execute``: on an
exhausted pool the checkout froze the event loop (the F105 shape) and then
refused every Composio action. These tests hold the fix to its effects:

* a warm cache answers from memory, with no database read;
* a failed refresh keeps the cached decision (WARNING); only a cold cache with an
  unreadable setting fails closed (ERROR);
* the agent executor's read never runs on the event loop's thread, and a stale
  cache answers at once while it refreshes in the background;
* the async form gives exactly the sync form's decision.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import config as config_module  # noqa: E402
import core.composio.deny_list as deny_list  # noqa: E402
from core.composio.client import ComposioClient  # noqa: E402
from core.composio.tool_executor import ComposioToolExecutor  # noqa: E402

BILLING = "HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE"
ALLOWED = "SLACK_SEND_MESSAGE"
BLOCKED = "This action is blocked in Automatos: "
READ_FAILED = BLOCKED + deny_list.READ_FAILED_REASON
UNREADABLE = BLOCKED + deny_list.UNREADABLE_REASON


class Reader:
    """Stands in for the strict setting read: counts calls, records the thread
    each ran on, and returns a value or raises."""

    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.threads = []

    @property
    def calls(self):
        return len(self.threads)

    def __call__(self):
        self.threads.append(threading.get_ident())
        if self.error is not None:
            raise self.error
        return self.value


@pytest.fixture
def reader(monkeypatch):
    stub = Reader(value=json.dumps([BILLING]))
    monkeypatch.setattr(deny_list, "_read_denied_actions", stub)
    return stub


@pytest.fixture
def clock(monkeypatch):
    now = [1_000.0]
    monkeypatch.setattr(deny_list, "_now", lambda: now[0], raising=False)
    return now


@pytest.fixture
def inline_refresh(monkeypatch):
    """A stale cache refreshes in the calling thread instead of a daemon thread."""
    monkeypatch.setattr(
        deny_list, "_start_background_refresh", getattr(deny_list, "_refresh", lambda: None), raising=False,
    )


@pytest.fixture
def log(monkeypatch):
    logger = MagicMock(name="logger")
    monkeypatch.setattr(deny_list, "logger", logger)
    return logger


def _ttl():
    return config_module.config.COMPOSIO_DENY_LIST_CACHE_TTL_SECONDS


def _executor():
    sdk = MagicMock(name="ComposioSDK")
    sdk.tools.execute.return_value = {"successful": True, "data": {"ok": True}}
    client = ComposioClient(api_key="test-key")
    client._composio = sdk
    return ComposioToolExecutor(db=MagicMock(name="db"), client=client), sdk


# ---------------------------------------------------------------------------
# Warm cache: memory, no database
# ---------------------------------------------------------------------------


def test_the_ttl_is_config_between_30_and_60_seconds():
    assert 30 <= _ttl() <= 60


def test_a_warm_cache_answers_from_memory_with_no_database_read(reader, clock):
    assert deny_list.composio_action_denial(BILLING).startswith(BLOCKED)
    assert reader.calls == 1

    clock[0] += _ttl() - 1  # still inside the TTL
    assert deny_list.composio_action_denial(BILLING).startswith(BLOCKED)
    assert deny_list.composio_action_denial(ALLOWED) is None
    assert reader.calls == 1  # both answered from memory


# ---------------------------------------------------------------------------
# Refresh failures: keep what was read; fail closed only when nothing was
# ---------------------------------------------------------------------------


def test_a_failed_refresh_keeps_the_cached_decision_and_logs_a_warning(reader, clock, inline_refresh, log):
    assert deny_list.composio_action_denial(BILLING).startswith(BLOCKED)  # warm: BILLING denied

    reader.error = TimeoutError("QueuePool limit reached")  # the pool is exhausted
    clock[0] += _ttl() + 1  # stale: this call refreshes

    denial = deny_list.composio_action_denial(BILLING)
    assert denial.startswith(BLOCKED) and BILLING in denial  # the cached decision, not a blanket refusal
    assert denial != READ_FAILED
    assert deny_list.composio_action_denial(ALLOWED) is None  # nothing else is refused
    assert reader.calls == 2  # one refresh attempt, then the retry waits a TTL
    log.warning.assert_any_call(
        "[ComposioDenyList] composio.denied_actions could not be refreshed; keeping the cached list", exc_info=True,
    )
    log.error.assert_not_called()


def test_a_cold_cache_with_an_unreadable_setting_fails_closed(reader, log):
    reader.error = TimeoutError("QueuePool limit reached")

    assert deny_list.composio_action_denial(ALLOWED) == READ_FAILED
    assert deny_list.composio_action_denial(BILLING) == READ_FAILED
    assert reader.calls == 2  # a failed read is never cached: each call tries again
    assert log.error.call_count == 2
    assert "could not be read" in log.error.call_args.args[0]
    assert log.error.call_args.kwargs["exc_info"] is True


def test_a_malformed_value_is_cached_as_a_refusal_until_it_is_fixed(reader, clock, inline_refresh):
    reader.value = "not json"
    assert deny_list.composio_action_denial(ALLOWED) == UNREADABLE
    assert deny_list.composio_action_denial(ALLOWED) == UNREADABLE
    assert reader.calls == 1

    reader.value = json.dumps([BILLING])  # a super-admin fixes it
    clock[0] += _ttl() + 1
    assert deny_list.composio_action_denial(ALLOWED) is None


# ---------------------------------------------------------------------------
# The event loop never waits on the database
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_agent_executors_read_never_runs_on_the_event_loop_thread(reader):
    loop_thread = threading.get_ident()
    executor, sdk = _executor()

    result = await executor.execute(action=BILLING, params={}, agent_id=1, workspace_id=uuid.uuid4())

    assert result["success"] is False and result["error_type"] == "action_denied"
    sdk.tools.execute.assert_not_called()
    assert reader.calls == 1
    assert reader.threads[0] != loop_thread  # read in a worker thread, not on the loop

    again = await executor.execute(action=BILLING, params={}, agent_id=1, workspace_id=uuid.uuid4())
    assert again["error_type"] == "action_denied"
    assert reader.calls == 1  # warm: answered from memory


@pytest.mark.asyncio
async def test_a_stale_cache_answers_at_once_and_refreshes_off_the_loop(reader, clock):
    loop_thread = threading.get_ident()
    assert (await deny_list.composio_action_denial_async(BILLING)).startswith(BLOCKED)  # warm

    release = threading.Event()
    reader.value = json.dumps([])  # the super-admin removed BILLING

    def slow_read():
        reader.threads.append(threading.get_ident())
        release.wait(5)  # a slow database: the caller must not wait for it
        return reader.value

    deny_list._read_denied_actions = slow_read  # restored by the reader fixture's monkeypatch
    clock[0] += _ttl() + 1

    started = time.monotonic()
    denial = await deny_list.composio_action_denial_async(BILLING)
    assert time.monotonic() - started < 1  # answered from the stale list at once
    assert denial.startswith(BLOCKED)

    release.set()
    for _ in range(200):  # the background refresh lands
        if reader.calls == 2 and not deny_list._refresh_lock.locked():
            break
        await asyncio.sleep(0.01)
    assert reader.calls == 2 and reader.threads[1] != loop_thread
    assert await deny_list.composio_action_denial_async(BILLING) is None  # the refreshed list


# ---------------------------------------------------------------------------
# One decision, two forms
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stored, error, slug", [
    (json.dumps([BILLING]), None, BILLING),
    (json.dumps([BILLING]), None, ALLOWED),
    ("not json", None, ALLOWED),
    (None, None, BILLING),
    (None, TimeoutError("QueuePool limit reached"), ALLOWED),
])
async def test_the_async_form_gives_the_sync_forms_decision(monkeypatch, stored, error, slug):
    monkeypatch.setattr(deny_list, "_read_denied_actions", Reader(value=stored, error=error))
    sync_decision = deny_list.composio_action_denial(slug)
    deny_list.reset_cache()
    async_decision = await deny_list.composio_action_denial_async(slug)
    assert async_decision == sync_decision


def test_reset_drops_a_refresh_that_started_before_it(monkeypatch):
    def read_then_reset():
        deny_list.reset_cache()  # a reset lands while this read is in flight
        return json.dumps([BILLING])

    monkeypatch.setattr(deny_list, "_read_denied_actions", read_then_reset)
    assert deny_list.composio_action_denial(BILLING).startswith(BLOCKED)  # this call still gets its answer
    assert deny_list._cache is None  # but the pre-reset read is not cached
