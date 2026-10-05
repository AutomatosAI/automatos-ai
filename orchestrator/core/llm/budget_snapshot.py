"""#836: what output budgets read from the database, held in memory.

Two facts: the ``llm_output_budget`` rows in system_settings (an operator's
override of a purpose's budget), and which models think (OpenRouter's catalogue,
``openrouter_models_cache.supports_reasoning``, with the model's output ceiling).

A manager reads its budgets when it is built, often on the event loop, and a
database read there would stall it (F105). Before #836 a read on the loop
answered from a per-purpose cache, or with nothing, and refreshed in the
background, so the first manager after a restart or after a new row used the old
budget. Now each worker loads both facts whole, on a thread: at boot before it
serves (warm), then every LLM_OUTPUT_BUDGET_REFRESH_SECONDS (keep_fresh). The
loop only reads memory. Off the loop (a thread, a script) a cold or stale
snapshot is loaded in place.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional, Set

from config import config

logger = logging.getLogger(__name__)

SETTINGS_CATEGORY = "llm_output_budget"
# A snapshot older than this many refresh intervals means the refresher stopped;
# a read on the loop then starts one load in the background.
STALE_AFTER_INTERVALS = 3


@dataclass(frozen=True)
class ModelFacts:
    """What the catalogue says about one model: whether it thinks (its reasoning
    tokens count toward max_tokens), and its own output maximum when known."""

    thinks: bool = False
    ceiling: Optional[int] = None


NOT_THINKING = ModelFacts()


@dataclass(frozen=True)
class Snapshot:
    budgets: Mapping[str, int] = field(default_factory=dict)
    models: Mapping[str, ModelFacts] = field(default_factory=dict)
    loaded_at: float = 0.0


_snapshot: Optional[Snapshot] = None
_background: Set[asyncio.Task] = set()


def _interval() -> float:
    return float(config.LLM_OUTPUT_BUDGET_REFRESH_SECONDS or 0)


def _load_budgets() -> Dict[str, int]:
    """Every llm_output_budget row as {purpose: tokens}; a row that is not a
    positive whole number is logged and left out (the table answers)."""
    from core.database.database import SessionLocal
    from core.models.system_settings import SystemSetting

    db = SessionLocal()
    try:
        rows = db.query(SystemSetting.key, SystemSetting.value).filter(
            SystemSetting.category == SETTINGS_CATEGORY).all()
    finally:
        db.close()
    budgets: Dict[str, int] = {}
    for key, value in rows:
        try:
            tokens = int(str(value).strip())
        except (TypeError, ValueError):
            logger.warning("[#836] llm_output_budget.%s = %r is not a whole number; the table's budget stands", key, value)
            continue
        if tokens > 0:
            budgets[key] = tokens
    return budgets


def _load_thinking_models() -> Dict[str, ModelFacts]:
    """The catalogue's active models that take a reasoning budget, with their ceilings."""
    from core.database.database import SessionLocal
    from core.models.openrouter_cache import OpenRouterModelCache as Cached

    db = SessionLocal()
    try:
        rows = db.query(Cached.model_id, Cached.max_completion_tokens).filter(
            Cached.supports_reasoning.is_(True), Cached.status == "active").all()
    finally:
        db.close()
    return {model_id: ModelFacts(thinks=True, ceiling=int(ceiling) if ceiling else None)
            for model_id, ceiling in rows}


def refresh() -> Snapshot:
    """Load both facts (two database reads; never call it on the event loop) and
    make them the worker's snapshot. A failed read is logged, and the snapshot
    before it (or none: the table's budgets) stands until the next interval."""
    global _snapshot
    kept = _snapshot or Snapshot()
    try:
        _snapshot = Snapshot(budgets=_load_budgets(), models=_load_thinking_models(), loaded_at=time.monotonic())
    except Exception:  # noqa: BLE001 — logged; the last good snapshot answers until the next try
        logger.exception("[#836] output budgets: could not read the settings rows or the model catalogue; "
                         "the last snapshot (or the table) stands")
        _snapshot = replace(kept, loaded_at=time.monotonic())
    return _snapshot


async def warm() -> Snapshot:
    """Load the snapshot on a thread (asyncio.to_thread carries the context)."""
    return await asyncio.to_thread(refresh)


async def keep_fresh() -> None:
    """Reload the snapshot every LLM_OUTPUT_BUDGET_REFRESH_SECONDS until cancelled
    (0: never; the boot load stands)."""
    while _interval() > 0:
        await asyncio.sleep(_interval())
        await warm()


def _is_fresh(snapshot: Optional[Snapshot]) -> bool:
    if snapshot is None:
        return False
    if _interval() <= 0:
        return True
    return time.monotonic() - snapshot.loaded_at < _interval() * STALE_AFTER_INTERVALS


def _load_in_background(loop: asyncio.AbstractEventLoop) -> None:
    """Start one load on ``loop``, unless one is already running there."""
    if any(task.get_loop() is loop and not task.done() for task in _background):
        return
    task = loop.create_task(warm())
    _background.add(task)
    task.add_done_callback(_background_done)


def _background_done(task: asyncio.Task) -> None:
    _background.discard(task)


def current() -> Snapshot:
    """The worker's snapshot. On the event loop it is never loaded in place: a
    cold or stale one answers (empty: the table's budgets) and one load starts in
    the background. Off the loop a cold or stale snapshot is loaded first."""
    snapshot = _snapshot
    if _is_fresh(snapshot):
        return snapshot
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return refresh()
    _load_in_background(loop)
    return snapshot or Snapshot()


def stored_budget(purpose: str) -> Optional[int]:
    """The llm_output_budget row for ``purpose``, or None."""
    return current().budgets.get(purpose)


def model_facts(model: Optional[str]) -> ModelFacts:
    """What the catalogue says about ``model``; a model it doesn't list does not think."""
    if not model:
        return NOT_THINKING
    return current().models.get(model, NOT_THINKING)
