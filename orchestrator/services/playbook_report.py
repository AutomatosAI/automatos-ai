"""The report a Playbook run files when it ends (moved out of api/recipe_executor).

F321 (night 9b, build 15): report f6d590ab for run exec-45d8ac862a79 (card #0102)
said "LLM calls: 0 … Tokens 0 / 0 / 35294 … Cost $0.0000" and listed its steps at
13,406 and 21,888 tokens: the executor's figure, each step's LAST model call. The
run made 19 calls, 317,872 tokens, $0.3928. The numbers now come from the run's
own ``llm_usage`` rows (``services.report_metrics``), per run and per step, in
the words every report kind uses.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

FINAL_OUTPUT_PREVIEW_CHARS = 500
SUMMARY_PREVIEW_CHARS = 100
FAILED_STATUSES = ("failed", "error")


def _report_status(success: bool, step_results: List[dict]) -> str:
    """ok, warning (a step failed, or the run failed after some work) or critical."""
    if not success:
        return "warning" if step_results else "critical"
    return "warning" if any(s.get("status") in FAILED_STATUSES for s in step_results) else "ok"


def _step_line(step: Mapping[str, Any], tokens: Optional[int]) -> str:
    """One step: its number, agent, status, tokens (from llm_usage) and duration."""
    order = step.get("order") or step.get("step_order") or "?"
    name = step.get("name") or step.get("agent_name") or "(unnamed)"
    duration = step.get("duration_ms")
    spent = f"{tokens} tokens" if tokens else "tokens not recorded"
    took = f"{duration} ms" if duration is not None else "n/a"
    return f"- **Step {order}: {name}** — {step.get('status', '?')} · {spent} · {took}"


def _final_output_lines(final_output: Any) -> List[str]:
    """The run's answer, its first characters, as the report shows it."""
    if not final_output:
        return []
    whole = str(final_output)
    preview = whole[:FINAL_OUTPUT_PREVIEW_CHARS] + ("…" if len(whole) > FINAL_OUTPUT_PREVIEW_CHARS else "")
    return ["", "## Final Output (preview)", "```", preview, "```"]


def playbook_report_content(
    name: str,
    execution_id: str,
    success: bool,
    metrics: Mapping[str, Any],
    step_results: List[dict],
    per_step: Mapping[Any, int],
    final_output: Any,
) -> str:
    """The report's markdown: header, metrics, steps, models, final output."""
    from services.report_metrics import metrics_lines

    lines = [f"# {name} — Playbook Report", f"**Execution:** {execution_id}",
             f"**Status:** {'completed' if success else 'failed'}", ""]
    lines += metrics_lines(metrics, model_label="Primary model")
    lines += [f"- Steps: {len(step_results)}", "", "## Steps"]
    lines += [_step_line(step, per_step.get(step.get("order"))) for step in step_results]
    if metrics.get("models_used"):
        lines += ["", "## Models Used"] + [f"- {model}" for model in metrics["models_used"]]
    return "\n".join(lines + _final_output_lines(final_output))


def _summary(metrics: Mapping[str, Any], step_results: List[dict]) -> str:
    """'2 steps · $0.3928 · 117721 ms · <the first step's answer>'."""
    from services.report_metrics import cost_summary

    summary = f"{len(step_results)} steps · {cost_summary(metrics)} · {metrics.get('duration_ms', 0)} ms"
    first = next((s.get("output_preview") for s in step_results if s.get("output_preview")), None)
    return f"{summary} · {str(first)[:SUMMARY_PREVIEW_CHARS]}" if first else summary


async def _run_metrics(db: Session, workspace_id: Any, recipe: Any, name: str, execution_id: str,
                       execution: Any, steps_count: int, duration_ms: int) -> Dict[str, Any]:
    """The run's numbers from its own llm_usage rows (after the last rows land)."""
    from services.report_metrics import settled_execution_metrics

    metrics = await settled_execution_metrics(
        db, workspace_id, execution_id=execution_id,
        started_at=getattr(execution, "started_at", None), completed_at=getattr(execution, "completed_at", None),
        extra={"recipe_id": getattr(recipe, "id", None), "recipe_name": name, "recipe_execution_id": execution_id,
               "steps_count": steps_count, "trigger": "playbook"},
    )
    return metrics if metrics.get("duration_ms") is not None else {**metrics, "duration_ms": duration_ms}


async def auto_create_playbook_report(
    *,
    db: Session,
    workspace_id: Any,
    recipe: Any,
    recipe_execution_id: str,
    execution: Any,
    step_results: List[dict],
    total_duration_ms: int,
    total_tokens: int,
    final_output: Any,
    success: bool,
) -> None:
    """File an agent_reports row summarising a Playbook run; never raises.

    ``total_tokens`` (the executor's sum of each step's last call) is not used:
    F321 — it is what made report f6d590ab contradict itself.
    """
    from services.report_metrics import step_tokens
    from services.report_service import ReportService

    try:
        name = getattr(recipe, "name", None) or f"playbook-{recipe.id}"
        metrics = await _run_metrics(db, workspace_id, recipe, name, recipe_execution_id, execution,
                                     len(step_results), total_duration_ms)
        per_step = step_tokens(db, workspace_id, recipe_execution_id, step_results) if metrics.get("llm_calls") else {}
        filed = await ReportService(db, workspace_id).create_report(
            agent_id=None, agent_name=f"playbook-{name}", title=f"Playbook: {name}",
            content=playbook_report_content(name, recipe_execution_id, success, metrics, step_results,
                                            per_step, final_output),
            report_type="summary", status=_report_status(success, step_results),
            summary=_summary(metrics, step_results), metrics=metrics,
        )
        if not filed.get("success"):
            logger.warning("[recipe_direct] Playbook auto-report DB insert failed for %s: %s",
                           recipe_execution_id, filed.get("error"))
    except Exception:
        logger.exception("[recipe_direct] the Playbook report for %s was not filed", recipe_execution_id)


__all__ = ["auto_create_playbook_report", "playbook_report_content"]
