"""Report metrics: one source (``llm_usage``) and one wording for every report.

F321 (night 9b, build 15): report f6d590ab ("Playbook: Monday green stock", run
exec-45d8ac862a79, card #0102) said "LLM calls: 0 … Tokens 0 / 0 / 35294 … Cost
$0.0000". The run's 19 model calls (llm_usage rows 102294–102320: 307,554 in,
10,318 out, $0.3928) were booked under the chat that started the run
(``chat:dd0b6649…``), so the rollup by the run's id found none, and the playbook
report printed the executor's own figure beside zero calls and zero dollars. That
figure was each step's LAST call only (13,406 + 21,888). 20 of the 78 playbook
reports of 2–4 Oct had zero calls beside non-zero tokens; all 25 runs Auto started
from chat were among them. The task and heartbeat reports each wrote their own
copy of the block.

Now every report reads calls, tokens and cost from ``llm_usage`` through
``compute_execution_metrics`` and says them through ``metrics_lines``: a number
that is not known is said in words, never printed as 0 beside one that is, and a
subscription session's cost is "plan usage", not "$0.0000".
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.database.session_health import rollback_if_aborted

logger = logging.getLogger(__name__)

SUBSCRIPTION_TIER = "subscription"
SUCCESS_STATUS = "success"
# A call's row is written off the event loop; a report written the moment its run
# ends waits this long at most for the run's last rows to land.
USAGE_SETTLE_SECONDS = 2.0
# A run's calls read back to split its tokens by step (a playbook step makes tens).
RUN_CALLS_READ_CAP = 5000

NO_CALLS = "none recorded for this run"
NOT_RECORDED = "not known: no model call was recorded for this run"
UNREADABLE = "not known: the usage records could not be read"
SUBSCRIPTION_COST = "plan usage (subscription), no dollar figure"

_SCOPE_PARAMS = ("execution_id", "agent_id", "w_start", "w_end")

_BY_MODEL_SQL = text("""
    SELECT model_id,
           COUNT(*) AS calls,
           COALESCE(SUM(input_tokens), 0) AS in_tok,
           COALESCE(SUM(output_tokens), 0) AS out_tok,
           COALESCE(SUM(total_tokens), 0) AS tot_tok,
           COALESCE(SUM(total_cost), 0) AS cost,
           COUNT(*) FILTER (WHERE tier = :subscription) AS sub_calls,
           COUNT(*) FILTER (WHERE status <> :success) AS errors
    FROM llm_usage
    WHERE workspace_id = CAST(:workspace_id AS uuid)
      AND (CAST(:execution_id AS text) IS NULL OR execution_id = CAST(:execution_id AS text))
      AND (CAST(:agent_id AS integer) IS NULL OR agent_id = CAST(:agent_id AS integer))
      AND (CAST(:w_start AS timestamp) IS NULL OR created_at >= CAST(:w_start AS timestamp))
      AND (CAST(:w_end AS timestamp) IS NULL OR created_at <= CAST(:w_end AS timestamp))
    GROUP BY model_id
""")

_RUN_CALLS_SQL = text("""
    SELECT created_at, total_tokens
    FROM llm_usage
    WHERE workspace_id = CAST(:workspace_id AS uuid) AND execution_id = :execution_id
    ORDER BY created_at
    LIMIT :cap
""")


def _naive_utc(value: Any) -> Optional[datetime]:
    """A datetime or ISO text as naive UTC, the way ``llm_usage.created_at`` is stored."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def _empty_metrics(started_at: Optional[datetime], completed_at: Optional[datetime]) -> Dict[str, Any]:
    """The standard keys, with no usage read yet."""
    duration = int((completed_at - started_at).total_seconds() * 1000) if started_at and completed_at else None
    return {
        "model": None, "models_used": [], "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
        "tokens_used": 0, "cost_usd": 0.0, "subscription_calls": 0, "llm_errors": 0,
        "usage_recorded": False, "duration_ms": duration,
        "started_at": started_at.isoformat() if started_at else None,
        "completed_at": completed_at.isoformat() if completed_at else None,
    }


def _usage_scope(
    agent_id: Optional[int],
    execution_id: Optional[str],
    started_at: Optional[datetime],
    completed_at: Optional[datetime],
) -> Optional[Dict[str, Any]]:
    """Which rows are this report's: the run's own id when it has one, else the
    agent's calls over the run's window. None when neither is known."""
    if execution_id:
        return {"execution_id": str(execution_id), "agent_id": None, "w_start": None, "w_end": None}
    if agent_id is None:
        return None
    windowed = bool(started_at and completed_at)
    return {"execution_id": None, "agent_id": agent_id,
            "w_start": _naive_utc(started_at) if windowed else None,
            "w_end": _naive_utc(completed_at) if windowed else None}


def _rolled_up(metrics: Mapping[str, Any], rows: Sequence[Any]) -> Dict[str, Any]:
    """``metrics`` with the per-model rows summed in. A new dict."""
    if not rows:
        return dict(metrics)
    ranked = sorted(rows, key=lambda r: int(r.tot_tok or 0), reverse=True)
    return {
        **metrics,
        "model": ranked[0].model_id,
        "models_used": [r.model_id for r in ranked],
        "llm_calls": sum(int(r.calls or 0) for r in rows),
        "input_tokens": sum(int(r.in_tok or 0) for r in rows),
        "output_tokens": sum(int(r.out_tok or 0) for r in rows),
        "tokens_used": sum(int(r.tot_tok or 0) for r in rows),
        "cost_usd": float(sum(float(r.cost or 0) for r in rows)),
        "subscription_calls": sum(int(r.sub_calls or 0) for r in rows),
        "llm_errors": sum(int(r.errors or 0) for r in rows),
        "usage_recorded": True,
    }


def _read_usage(db: Session, workspace_id: Any, metrics: Mapping[str, Any], scope: Mapping[str, Any]) -> Dict[str, Any]:
    """The rollup of ``scope``'s rows, or ``metrics`` marked unreadable when the read fails."""
    params = {"workspace_id": str(workspace_id), "subscription": SUBSCRIPTION_TIER,
              "success": SUCCESS_STATUS, **{k: scope.get(k) for k in _SCOPE_PARAMS}}
    try:
        rows = db.execute(_BY_MODEL_SQL, params).fetchall()
    except Exception:
        logger.exception("[report-metrics] llm_usage rollup failed for ws=%s scope=%s", workspace_id, dict(scope))
        rollback_if_aborted(db, "the report metrics rollup")
        return {**metrics, "usage_read_failed": True}
    return _rolled_up(metrics, rows)


def compute_execution_metrics(
    db: Session,
    workspace_id: Any,
    *,
    agent_id: Optional[int] = None,
    execution_id: Optional[str] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Model, calls, tokens, cost and duration of one run, from ``llm_usage``.

    ``execution_id`` (preferred) matches the run's own rows; otherwise
    ``agent_id`` over ``started_at``–``completed_at``. Always the standard keys;
    ``usage_recorded`` says whether any row was found, ``subscription_calls`` how
    many ran on a plan with no dollar figure. ``extra`` is merged last.
    """
    metrics = _empty_metrics(started_at, completed_at)
    scope = _usage_scope(agent_id, execution_id, started_at, completed_at)
    if scope is not None:
        metrics = _read_usage(db, workspace_id, metrics, scope)
    return {**metrics, **(extra or {})}


async def settled_execution_metrics(db: Session, workspace_id: Any, **kwargs: Any) -> Dict[str, Any]:
    """``compute_execution_metrics`` once the rows handed to the write threads
    have landed (at most ``USAGE_SETTLE_SECONDS``): a report written as its run
    ends would otherwise miss the run's last call. Read through
    ``services.report_service``, the name the report writers always used."""
    from core.best_effort import drain
    from services import report_service

    await asyncio.to_thread(drain, USAGE_SETTLE_SECONDS)
    return report_service.compute_execution_metrics(db, workspace_id, **kwargs)


def _step_starts(steps: Iterable[Mapping[str, Any]]) -> List[tuple]:
    """(start, order) for each step that has a start, earliest first."""
    starts = [(_naive_utc(s.get("started_at")), s.get("order")) for s in steps]
    return sorted((start, order) for start, order in starts if start is not None)


def step_tokens(db: Session, workspace_id: Any, execution_id: str, steps: Sequence[Mapping[str, Any]]) -> Dict[Any, int]:
    """Each step's tokens from the run's ``llm_usage`` rows: a call belongs to
    the latest step that had started when it was booked. Empty when the rows
    cannot be read (logged); a step with no calls is absent."""
    try:
        rows = db.execute(_RUN_CALLS_SQL, {"workspace_id": str(workspace_id), "execution_id": str(execution_id),
                                           "cap": RUN_CALLS_READ_CAP}).fetchall()
    except Exception:
        logger.exception("[report-metrics] step tokens unreadable for %s", execution_id)
        rollback_if_aborted(db, "the report step-tokens read")
        return {}
    starts, totals = _step_starts(steps), {}
    for row in rows:
        booked = _naive_utc(row.created_at)
        owners = [order for start, order in starts if booked is not None and start <= booked]
        if owners:
            totals[owners[-1]] = totals.get(owners[-1], 0) + int(row.total_tokens or 0)
    return totals


def _all_on_a_plan(metrics: Mapping[str, Any], subscription: bool) -> bool:
    calls = int(metrics.get("llm_calls") or 0)
    return subscription or (calls > 0 and int(metrics.get("subscription_calls") or 0) >= calls)


def cost_text(metrics: Mapping[str, Any], *, subscription: bool = False) -> str:
    """The run's cost in words: dollars, plan usage, a split of the two, or why
    it is not known."""
    if _all_on_a_plan(metrics, subscription):
        return SUBSCRIPTION_COST
    if metrics.get("usage_read_failed"):
        return UNREADABLE
    calls = int(metrics.get("llm_calls") or 0)
    if not calls:
        return NOT_RECORDED
    dollars = f"${float(metrics.get('cost_usd') or 0):.4f}"
    on_plan = int(metrics.get("subscription_calls") or 0)
    if not on_plan:
        return dollars
    return f"{dollars} for {calls - on_plan} metered calls; {on_plan} ran on a subscription plan, no dollar figure"


def tokens_text(metrics: Mapping[str, Any]) -> str:
    """Tokens in / out / total, or why they are not known (and what the run
    itself reported, when it did)."""
    if metrics.get("usage_read_failed"):
        return UNREADABLE
    if not int(metrics.get("llm_calls") or 0):
        reported = int(metrics.get("reported_tokens") or 0)
        return f"{NOT_RECORDED} (the run reported {reported} in total)" if reported else NOT_RECORDED
    return (f"{int(metrics.get('input_tokens') or 0)} / {int(metrics.get('output_tokens') or 0)} / "
            f"{int(metrics.get('tokens_used') or 0)}")


def metrics_lines(metrics: Mapping[str, Any], *, subscription: bool = False, model_label: str = "Model") -> List[str]:
    """The "## Execution Metrics" lines every report kind shares (model, calls,
    tokens, cost, duration)."""
    calls = int(metrics.get("llm_calls") or 0)
    lines = [
        "## Execution Metrics",
        f"- {model_label}: {metrics.get('model') or 'unknown'}",
        f"- LLM calls: {calls if calls else (UNREADABLE if metrics.get('usage_read_failed') else NO_CALLS)}",
        f"- Tokens (in/out/total): {tokens_text(metrics)}",
        f"- Cost: {cost_text(metrics, subscription=subscription)}",
    ]
    if metrics.get("duration_ms") is not None:
        lines.append(f"- Duration: {metrics['duration_ms']} ms")
    return lines


def cost_summary(metrics: Mapping[str, Any], *, subscription: bool = False) -> str:
    """The cost as a report summary's short segment."""
    if _all_on_a_plan(metrics, subscription):
        return "plan usage"
    if metrics.get("usage_read_failed") or not int(metrics.get("llm_calls") or 0):
        return "cost not recorded"
    return f"${float(metrics.get('cost_usd') or 0):.4f}"


__all__ = [
    "compute_execution_metrics",
    "cost_summary",
    "cost_text",
    "metrics_lines",
    "settled_execution_metrics",
    "step_tokens",
    "tokens_text",
]
