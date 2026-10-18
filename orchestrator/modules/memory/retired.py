"""P256-FIX-N15-9 (night 15, F429): a standing rule the owner replaces is retired.

Night 15: the owner replaced their 'cafe minimum 12 kg' with '10 kg'. Auto stored the new rule,
the old row stayed in the memory store, and the Memory page listed both while the rules said 10.
Now the row a new rule replaces is retired (``platform_store_memory``'s
``rule_replaces.retires_what_it_replaces`` picks it): its metadata says when, and by what
(``retired_at``, ``replaced_by``); nothing is deleted. A delete is ``platform_delete_memory``,
which an owner or admin's own chat instruction runs with no card (``platform_executor``'s
human-directed rule, Gerard 2026-08-06): it is no owner-only action (``owner_only``), so a rule
is replaced by retiring it, never by deleting it first (P256-FIX-N17-4).

A retired row is left out of every read of the durable store (``leaves_out_retired`` on
``DurableMemoryStore.search`` and ``_scroll``): recall, the standing rules, the Memory page and
Auto's browse and search. The GDPR export still holds it. The owner saying it again in the same
words brings it back (``DurableMemoryStore._restated_now``, through ``unretired``).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

RETIRED_AT_KEY = "retired_at"
REPLACED_BY_KEY = "replaced_by"
RETIREMENT_KEYS = (RETIRED_AT_KEY, REPLACED_BY_KEY)

Read = Callable[..., Awaitable[List[Dict[str, Any]]]]


def is_retired(mem: Any) -> bool:
    """Whether a memory row (or a stored point's payload) was retired by a rule that replaced it."""
    meta = mem.get("metadata") if isinstance(mem, dict) else None
    return isinstance(meta, dict) and bool(meta.get(RETIRED_AT_KEY))


def retired_metadata(metadata: Optional[Dict[str, Any]], replaced_by: str, at: str) -> Dict[str, Any]:
    """``metadata`` marked retired at ``at`` by the rule ``replaced_by`` (a new dict)."""
    return {**(metadata or {}), RETIRED_AT_KEY: at, REPLACED_BY_KEY: replaced_by}


def unretired(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``metadata`` with its retirement taken off (a new dict)."""
    return {k: v for k, v in (metadata or {}).items() if k not in RETIREMENT_KEYS}


def leaves_out_retired(read: Read) -> Read:
    """Wrap a durable-store read: the rows it returns, retired ones left out."""
    @functools.wraps(read)
    async def wrapped(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return [row for row in await read(*args, **kwargs) if not is_retired(row)]
    return wrapped


async def owners_rows(workspace_id: Any, owner: str) -> List[Dict[str, Any]]:
    """The memories ``owner`` owns in the workspace's own namespace (retired ones left out)."""
    from config import config
    from modules.memory.unified_memory_service import get_unified_memory_service

    service = get_unified_memory_service()
    limit = config.STANDING_RULES_SCAN_LIMIT
    rows = await service._durable.get_where_any(service.namespace(str(workspace_id)).resolve(None), {},
                                                limit=limit, where={"metadata.owner": owner})
    if len(rows) >= limit:
        logger.warning("[N15-9] %d rows of %s read in workspace %s, the cap (STANDING_RULES_SCAN_LIMIT): a rule "
                       "past it is not weighed for replacement", limit, owner, workspace_id)
    return rows


async def retire(workspace_id: Any, rows: List[Dict[str, Any]], replaced_by: str) -> List[str]:
    """Retire ``rows`` (the workspace namespace's, from ``owners_rows``) as replaced by the rule
    ``replaced_by``; the ids retired. A store that fails raises (the caller logs it)."""
    from modules.memory.unified_memory_service import get_unified_memory_service

    if not rows:
        return []
    service = get_unified_memory_service()
    namespace = service.namespace(str(workspace_id)).resolve(None)
    retired = await service._durable.retire(namespace, rows, replaced_by)
    if retired:
        await service._invalidate_search_cache(str(workspace_id))   # a cached recall must not bring it back
        logger.info("[N15-9] retired %d replaced memor(ies) in workspace %s", len(retired), workspace_id)
    return retired


__all__ = ["REPLACED_BY_KEY", "RETIRED_AT_KEY", "is_retired", "leaves_out_retired", "owners_rows", "retire",
           "retired_metadata", "unretired"]
