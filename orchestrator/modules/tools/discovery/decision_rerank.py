"""PRD-248 S4 — the tool-surface rerank in production (everything injected).

``narrow_with_decisions`` applies the ``tool_rerank_mode`` dial to the ranked
allow-list the embedding index produced for a turn:

* off    — unchanged, byte-identical to today;
* shadow — a fire-and-forget comparison: the engine judges a wider candidate
           list and the cut is written to the shadow log beside today's, the
           surface itself unchanged;
* live   — the reranked cut replaces the embedding top-K; on any miss (no
           answer, too few answers, an error) the embedding cut is kept.

The judgement itself is the pure ``core.llm.decisions.rerank``. The index,
registry, engine and shadow writer are passed in, so this runs in tests with
none of the router's module graph loaded.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from core.llm.decisions.questions import DecisionResult
from core.llm.decisions.rerank import (
    DEFAULT_CANDIDATES,
    DEFAULT_MIN_KEEP,
    DEFAULT_MIN_PROBABILITY,
    PURPOSE,
    Decide,
    RerankCut,
    compare_surfaces,
    query_digest,
    rerank_candidates,
)

logger = logging.getLogger(__name__)

PREVIEW_CHARS = 120

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_LIVE = "live"

RankWide = Callable[[int], Awaitable[Sequence[Tuple[str, float]]]]
Describe = Callable[[str], str]

# Shadow rerank tasks in flight: asyncio holds only a weak reference to a
# running task, so a fire-and-forget shadow must be held until it is done.
_SHADOW_TASKS: set = set()


async def _rerank_turn(
    *,
    query: str,
    rank_wide: RankWide,
    describe: Describe,
    decide: Decide,
    candidates_n: int,
    top_k: int,
    min_probability: float,
    min_keep: int,
    workspace_id: Any,
) -> Tuple[Optional[RerankCut], Optional[DecisionResult], List[str]]:
    wide = await rank_wide(max(int(candidates_n), int(top_k)))
    names = [name for name, _score in wide if name]
    candidates = [(name, describe(name) or "") for name in names]
    cut, result = await rerank_candidates(
        query=query,
        candidates=candidates,
        decide=decide,
        top_k=top_k,
        min_probability=min_probability,
        min_keep=min_keep,
        workspace_id=workspace_id,
    )
    return cut, result, names


async def _shadow_turn(
    *,
    query: str,
    allowed: Sequence[str],
    record_shadow: Callable[[Mapping[str, Any]], None],
    workspace_id: Any,
    **rerank_kwargs: Any,
) -> None:
    started = time.monotonic()
    row: Dict[str, Any] = {
        "purpose": PURPOSE,
        "workspace_id": workspace_id,
        "query_sha": query_digest(query),
        "query_preview": (query or "")[:PREVIEW_CHARS],
        "embedding_top": list(allowed),
    }
    try:
        cut, result, wide = await _rerank_turn(query=query, workspace_id=workspace_id, **rerank_kwargs)
        row["wide"] = wide
        if result is None:
            row["error"] = "no_result"
        else:
            row.update(
                provider=result.provider,
                model=result.model,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
            )
            if cut is None:
                row["error"] = "too_few_answers"
            else:
                row.update(cut.to_dict())
                row["compare"] = compare_surfaces(allowed, cut.kept)
    except Exception as exc:  # noqa: BLE001 — a shadow must never surface
        row["error"] = f"{exc!r}"[:200]
    row["shadow_ms"] = int((time.monotonic() - started) * 1000)
    try:
        record_shadow(row)
    except Exception:  # noqa: BLE001
        logger.debug("[tool-rerank] shadow row not written", exc_info=True)
    logger.info(
        "[tool-rerank] shadow",
        extra={
            "workspace_id": workspace_id,
            "error": row.get("error"),
            "latency_ms": row.get("latency_ms"),
            "compare": row.get("compare"),
            "nothing_fits": row.get("nothing_fits"),
        },
    )


async def narrow_with_decisions(
    *,
    query: Optional[str],
    allowed: Optional[List[str]],
    mode: str,
    rank_wide: RankWide,
    describe: Describe,
    decide: Decide,
    record_shadow: Callable[[Mapping[str, Any]], None],
    top_k: int,
    candidates_n: int = DEFAULT_CANDIDATES,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    min_keep: int = DEFAULT_MIN_KEEP,
    workspace_id: Any = None,
) -> Optional[List[str]]:
    """Apply the dial to a ranked allow-list. Never raises; never turns a real
    list into None."""
    if mode == MODE_OFF or not query or not allowed:
        return allowed
    rerank_kwargs = dict(
        rank_wide=rank_wide,
        describe=describe,
        decide=decide,
        candidates_n=candidates_n,
        top_k=top_k,
        min_probability=min_probability,
        min_keep=min_keep,
    )
    if mode == MODE_SHADOW:
        try:
            task = asyncio.get_running_loop().create_task(
                _shadow_turn(
                    query=query,
                    allowed=allowed,
                    record_shadow=record_shadow,
                    workspace_id=workspace_id,
                    **rerank_kwargs,
                )
            )
        except RuntimeError:
            return allowed
        _SHADOW_TASKS.add(task)
        task.add_done_callback(_SHADOW_TASKS.discard)
        return allowed
    if mode == MODE_LIVE:
        try:
            cut, _result, _wide = await _rerank_turn(query=query, workspace_id=workspace_id, **rerank_kwargs)
        except Exception as exc:  # noqa: BLE001 — fail-open to the embedding cut
            logger.warning("[tool-rerank] live rerank failed — embedding cut kept: %s", exc)
            return allowed
        if cut is None or not cut.kept:
            logger.info("[tool-rerank] live rerank returned no cut — embedding cut kept")
            return allowed
        logger.info(
            "[tool-rerank] live kept %d of %d (nothing_fits=%s)",
            len(cut.kept), cut.answered, cut.nothing_fits,
            extra={"workspace_id": workspace_id},
        )
        # Canonical order, not probability order: the tool block sits at the head
        # of Auto's cached prefix, so the same SET of actions must produce the
        # same bytes turn after turn or a live rerank pays for its smaller
        # surface in prompt-cache misses (night-1 data: first calls cached only
        # 2.4–3.5k of a ~34k prefix because the block changed per query).
        return sorted(cut.kept)
    return allowed
