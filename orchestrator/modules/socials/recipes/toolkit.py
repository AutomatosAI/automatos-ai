"""PRD-251 D12, D13: what every recipe shares about a workspace's Composio toolkit.

* **Calling it.** A recipe calls an action the media capability registry offers
  (``modules/socials/capabilities.py``) through ``ComposioToolExecutor.execute``
  on the workspace's own connection, which checks the Wave 0 deny list again
  (:func:`call`). The platform calls for a person's render, not an agent: the
  admin pattern (``api/bug_reports.py``), agent id 0. Its parameters are built
  from the action's cached schema (:func:`params_for`), and its own output is
  read out of Composio's envelope (:func:`output_of`).
* **Its account (D13).** A toolkit that bills credit is read before and after a
  job (:func:`balance_of`), and the difference is booked on the media lane. Two
  readings of one account never overlap: one credit window per workspace and
  toolkit runs at a time, in this process and across every worker process
  (:func:`credit_window`, a Postgres advisory lock), so a render never books
  another render's spend.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncIterator, Callable, Dict, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy import text

from config import config
from modules.socials.capabilities import OfferedAction

# The admin pattern (api/bug_reports.py): the platform calls the workspace's own
# connection for a person's render, not an agent. The registry already checked
# the toolkit is connected, allowlisted, cached and not denied; the executor
# checks the deny list again.
PLATFORM_AGENT_ID = 0
ERROR_CHARS = 300
# How deep an answer is searched for a value.
MAX_DEPTH = 6


class ToolkitSchemaChanged(Exception):
    """The action's cached schema no longer takes a parameter the recipe needs."""


# ── calling an action ───────────────────────────────────────────────────────
def params_for(action: OfferedAction, wanted: Mapping[str, Any], required: Sequence[str]) -> Dict[str, Any]:
    """``wanted`` as the action's cached schema takes it: a parameter the schema
    does not list is left out, and a required one it does not list refuses the
    call (:class:`ToolkitSchemaChanged`). A schema the sync has not filled yet
    (``{}``) takes them all."""
    properties = action.parameters.get("properties") if isinstance(action.parameters, Mapping) else None
    if not isinstance(properties, Mapping) or not properties:
        return dict(wanted)
    missing = [name for name in required if name not in properties]
    if missing:
        raise ToolkitSchemaChanged(
            f"{action.slug} takes no {', '.join(missing)} (its cached schema changed): sync the toolkit's actions"
        )
    return {name: value for name, value in wanted.items() if name in properties}


async def call(executor: Any, workspace_id: UUID, action: OfferedAction, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Run ``action`` on the workspace's own connection; the executor's result as it answers."""
    return await executor.execute(
        action=action.slug,
        params=dict(params),
        agent_id=PLATFORM_AGENT_ID,
        workspace_id=workspace_id,
        app_name=action.toolkit.upper(),
        skip_validation=True,
    )


def error_of(result: Mapping[str, Any]) -> str:
    return str(result.get("error") or "no reason given").strip()[:ERROR_CHARS]


def output_of(result: Mapping[str, Any]) -> Any:
    """The tool's own output in an executor result: ``data``, out of Composio's
    ``{"data", "successful", "error"}`` envelope, and parsed when the tool
    answered JSON as text."""
    data = result.get("data") if isinstance(result, Mapping) else None
    if isinstance(data, Mapping) and "data" in data and ("successful" in data or "error" in data):
        data = data["data"]
    if isinstance(data, str) and data.strip()[:1] in ("{", "["):
        try:
            return json.loads(data)
        except ValueError:
            return data
    return data


def children(item: Any) -> list:
    """What an answer holds one level down: an object's values, a list's items."""
    if isinstance(item, Mapping):
        return list(item.values())
    if isinstance(item, (list, tuple)):
        return list(item)
    return []


# ── the account (D13) ───────────────────────────────────────────────────────
def decimal_of(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def balance_of(response: Any, keys: Sequence[str]) -> Optional[Decimal]:
    """The credit a balance action reports under ``keys``, shallowest first; a
    bare number when the action answers with the number alone."""
    bare = decimal_of(response) if isinstance(response, (int, float, str)) else None
    if bare is not None:
        return bare
    frontier = [response]
    for _ in range(MAX_DEPTH):
        objects = [item for item in frontier if isinstance(item, Mapping)]
        for key in keys:
            for obj in objects:
                number = decimal_of(obj.get(key))
                if number is not None:
                    return number
        frontier = [value for item in frontier for value in children(item)]
        if not frontier:
            break
    return None


# One lock's key space in the database: a namespace per kind of window keeps its
# keys apart from every other advisory lock ('socv', the credit window).
CREDIT_LOCK_NAMESPACE = 0x736F6376
_TRY_LOCK = text("SELECT pg_try_advisory_xact_lock(:namespace, hashtext(:key))")
# Within one process, a window is also an asyncio lock: renders on this event
# loop queue here without holding a database connection each.
_LOCKS: Dict[Tuple[int, str], asyncio.Lock] = {}


def process_lock(namespace: int, key: str) -> asyncio.Lock:
    return _LOCKS.setdefault((namespace, key), asyncio.Lock())


def account_lock(workspace_id: UUID, toolkit: str) -> asyncio.Lock:
    return process_lock(CREDIT_LOCK_NAMESPACE, credit_lock_key(workspace_id, toolkit))


def credit_lock_key(workspace_id: UUID, toolkit: str) -> str:
    return f"{workspace_id}:{toolkit}"


@asynccontextmanager
async def advisory_lock(session_factory: Callable[[], Any], namespace: int, key: str) -> AsyncIterator[None]:
    """Hold ``(namespace, key)`` across every worker process.

    Production runs several uvicorn workers, and a render runs in whichever took
    its request. The lock is transaction-scoped and held on its own connection
    for the whole window, so it is released when the window ends, and with the
    connection if anything fails; waiting for it polls every
    ``SOCIALS_RENDER_POLL_SECONDS`` within the render's own deadline. A database
    without advisory locks (SQLite, in the unit tests) has one process: the
    asyncio lock is the window there."""
    db = session_factory()
    try:
        if db.get_bind().dialect.name == "postgresql":
            params = {"namespace": namespace, "key": key}
            while not await asyncio.to_thread(lambda: bool(db.execute(_TRY_LOCK, params).scalar())):
                await asyncio.sleep(config.SOCIALS_RENDER_POLL_SECONDS)
        yield
    finally:
        await asyncio.to_thread(db.close)  # ends the transaction, and with it the lock


@asynccontextmanager
async def window(session_factory: Callable[[], Any], namespace: int, key: str) -> AsyncIterator[None]:
    """One holder of ``(namespace, key)`` at a time: in this process, then across processes."""
    async with process_lock(namespace, key):
        async with advisory_lock(session_factory, namespace, key):
            yield


def credit_window(session_factory: Callable[[], Any], workspace_id: UUID, toolkit: str):
    """One credit window per workspace and toolkit at a time, across every worker process."""
    return window(session_factory, CREDIT_LOCK_NAMESPACE, credit_lock_key(workspace_id, toolkit))
