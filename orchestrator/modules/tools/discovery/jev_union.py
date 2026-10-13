"""P256-FIX-N14-8 — Jev's tool shortlist, an experiment behind a flag.

TESTER's read of 2,722 v2 ``tool_rerank`` shadow rows (13 Oct, night 14): on the
turns where Auto used a platform tool successfully, that tool was in Jev's top 5
51% of the time against 46% for the platform's embedding top 15 (22% for its
top 5), at a rerank p50 of 428 ms.

``TOOL_SHORTLIST_JEV_UNION`` (off by default) makes Auto's offered shortlist the
union of the two: Jev's kept top few (``TOOL_SHORTLIST_JEV_TOP``) first, then the
platform's embedding top few (``TOOL_SHORTLIST_PLATFORM_TOP``), each name once.
Jev gets ``TOOL_SHORTLIST_JEV_TIMEOUT_S``; late, silent or failing, the
platform's list is offered unchanged. Every turn writes one ``tool_rerank``
shadow row that names the list offered (``offered``) and why (``offered_reason``).

Off, the router runs the ``tool_rerank_mode`` dial exactly as before.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from config import config
from modules.tools.discovery.decision_rerank import MODE_UNION, judged_fields, shadow_row, write_shadow

logger = logging.getLogger(__name__)

# What the shadow row's ``offered`` names.
OFFERED_UNION = "jev_union"
OFFERED_PLATFORM = "platform"

# Why the platform's list was offered.
REASON_TIMEOUT = "timeout"
REASON_NO_ANSWER = "no_answer"
REASON_ERROR = "error"

RerankTurn = Callable[..., Awaitable[Tuple[Any, Any, List[str], str]]]


def offered_mode(dial_mode: str) -> str:
    """The mode the router runs: ``union`` while the experiment is on, else the dial."""
    return MODE_UNION if bool(config.TOOL_SHORTLIST_JEV_UNION) else dial_mode


def union_of(jev: Sequence[str], platform: Sequence[str]) -> List[str]:
    """Jev's names first, then the platform's, each name once, in that order."""
    return list(dict.fromkeys(name for name in [*jev, *platform] if name))


def _jev_union(allowed: Sequence[str], kept: Sequence[str]) -> List[str]:
    jev_top = max(0, int(config.TOOL_SHORTLIST_JEV_TOP))
    platform_top = max(0, int(config.TOOL_SHORTLIST_PLATFORM_TOP))
    return union_of(list(kept)[:jev_top], list(allowed)[:platform_top])


async def _judge(
    query: str, allowed: List[str], rerank: RerankTurn, workspace_id: Any, rerank_kwargs: Dict[str, Any]
) -> Tuple[Optional[List[str]], Dict[str, Any], Optional[str]]:
    """(the union or None, the judgement's shadow fields, why the platform's list stands)."""
    timeout_s = float(config.TOOL_SHORTLIST_JEV_TIMEOUT_S)
    try:
        cut, result, wide, source = await asyncio.wait_for(
            rerank(query=query, workspace_id=workspace_id, **rerank_kwargs), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        logger.info("[tool-shortlist] Jev missed %.2fs — the platform's list offered", timeout_s)
        return None, {"error": REASON_TIMEOUT}, REASON_TIMEOUT
    except Exception as exc:  # noqa: BLE001 — fail-open to the platform's list, logged
        logger.warning("[tool-shortlist] Jev rerank failed — the platform's list offered", exc_info=True)
        return None, {"error": f"{exc!r}"[:200]}, REASON_ERROR
    fields = judged_fields(allowed, cut, result, wide, source)
    if cut is None or not cut.kept:
        return None, fields, REASON_NO_ANSWER
    return _jev_union(allowed, cut.kept), fields, None


async def union_shortlist(
    *,
    query: str,
    allowed: List[str],
    rerank: RerankTurn,
    record_shadow: Callable[[Mapping[str, Any]], None],
    workspace_id: Any,
    rerank_kwargs: Dict[str, Any],
) -> List[str]:
    """The list Auto is offered this turn: Jev's union when Jev answered in time,
    else ``allowed`` unchanged. Never raises; one shadow row either way."""
    started = time.monotonic()
    union, fields, reason = await _judge(query, allowed, rerank, workspace_id, rerank_kwargs)
    offered = union if union is not None else list(allowed)
    row = {
        **shadow_row(query, allowed, workspace_id),
        **fields,
        "mode": MODE_UNION,
        "offered": OFFERED_UNION if union is not None else OFFERED_PLATFORM,
        "offered_reason": reason,
        "offered_list": offered,
        "shadow_ms": int((time.monotonic() - started) * 1000),
    }
    write_shadow(record_shadow, row)
    logger.info(
        "[tool-shortlist] offered %s (%d actions)",
        row["offered"], len(offered),
        extra={"workspace_id": workspace_id, "offered_reason": reason, "latency_ms": row.get("latency_ms")},
    )
    return offered
