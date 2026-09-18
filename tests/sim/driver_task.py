"""The *task* scenario: file a ticket for an agent and watch it to the end.

Files ``POST /api/v1/tasks`` assigned to a seeded agent. On the local edition
creating an assigned ticket is the operator's consent (``board_consent``), so
the dispatcher claims it within ``BOARD_DISPATCH_POLL_SECONDS``; if it has not
after the grace period the runner sends ``run-now`` once and records that as
a finding rather than a fix. Every poll also answers pending questions.

Terminal states are the board's own: done, review, failed, blocked, cancelled.
What the agent produced is read back the way a customer would find it —
deliverables linked to the task, reports by the agent since the run began.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from .answerer import answer_pending
from .api import Api, ApiError, items_of
from .packs import Scenario
from .results import (Check, RunContext, ScenarioResult, effect_checks, must_contain_checks, now_iso,
                      parse_iso, text_of)

TERMINAL = ("done", "review", "failed", "blocked", "cancelled")
RESULT_CAP = 40_000


def _create(ctx: RunContext, sc: Scenario, agent: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "title": sc.title, "description": sc.description or None, "raw_prompt": sc.description or None,
        "assigned_agent_id": agent["id"], "priority": sc.priority, "review_mode": sc.review_mode,
        "source_type": "user", "tags": ["sim", sc.id, *sc.tags],
    }
    return ctx.api.post("/api/v1/tasks", body, label=f"{sc.id}:create")


def _denials(task: Mapping[str, Any]) -> tuple[str, ...]:
    ref = task.get("runtime_ref")
    if not isinstance(ref, Mapping):
        return ()
    notes = []
    for key in ("session_denials", "pending_permissions", "session_asks", "denials"):
        value = ref.get(key)
        if isinstance(value, list) and value:
            notes.append(f"runtime_ref.{key}: {len(value)} entries")
    return tuple(notes)


def _watch(ctx: RunContext, sc: Scenario, task_id: int) -> tuple[dict[str, Any], list[dict], list[dict], list[str], str]:
    """Poll until terminal or timeout. Returns (task, timeline, asks, notes, outcome)."""
    settings = ctx.settings
    deadline = time.monotonic() + (sc.timeout_s or settings.task_timeout_s)
    started = time.monotonic()
    timeline: list[dict[str, Any]] = []
    asks: list[dict[str, Any]] = []
    notes: list[str] = []
    nudged = False
    last_status = None
    task: dict[str, Any] = {}
    while True:
        task = ctx.api.get(f"/api/v1/tasks/{task_id}", label=f"{sc.id}:poll") or {}
        status = str(task.get("status") or "")
        if status != last_status:
            timeline.append({"t": now_iso(), "status": status, "attempts": task.get("attempts"),
                             "elapsed_s": round(time.monotonic() - started, 1)})
            last_status = status
        asks.extend(answer_pending(ctx.api, ctx.persona, label=f"{sc.id}:ask"))
        if status in TERMINAL:
            return task, timeline, asks, notes, status
        if status == "assigned" and not nudged and time.monotonic() - started > settings.dispatch_grace_s:
            ctx.api.post(f"/api/v1/tasks/{task_id}/run-now", {}, label=f"{sc.id}:run-now")
            nudged = True
            notes.append(f"dispatcher had not claimed the ticket after {settings.dispatch_grace_s}s; run-now sent")
        if time.monotonic() > deadline:
            notes.append(f"still '{status}' at the {int(deadline - started)}s timeout")
            return task, timeline, asks, notes, "timeout"
        time.sleep(settings.poll_s)


def _deliverables(api: Api, task_id: int, agent_id: int, since: str, label: str) -> list[dict[str, Any]]:
    found = items_of(api.get("/api/deliverables", params={"source_type": "task", "source_id": str(task_id), "limit": 50},
                             label=f"{label}:deliverables"), "deliverables")
    if not found:
        listed = items_of(api.get("/api/deliverables", params={"agent_id": agent_id, "date_from": since, "limit": 50},
                                  label=f"{label}:deliverables-by-agent"), "deliverables")
        found = [d for d in listed if isinstance(d, Mapping)]
    detailed = []
    for item in found[:10]:
        try:
            detailed.append(api.get(f"/api/deliverables/{item['id']}", label=f"{label}:deliverable") or item)
        except (ApiError, KeyError):
            detailed.append(item)
    return detailed


def _reports(api: Api, agent_id: int, since: str, label: str) -> list[dict[str, Any]]:
    listed = items_of(api.get("/api/reports", params={"agent_id": agent_id, "period": "1d", "limit": 50},
                              label=f"{label}:reports"), "reports")
    floor = parse_iso(since)
    fresh = [r for r in listed if isinstance(r, Mapping) and (parse_iso(r.get("created_at")) or floor) >= floor]
    detailed = []
    for item in fresh[:10]:
        try:
            detailed.append(api.get(f"/api/reports/{item['id']}", label=f"{label}:report") or item)
        except (ApiError, KeyError):
            detailed.append(item)
    return detailed


def run_task(ctx: RunContext, sc: Scenario) -> ScenarioResult:
    started = now_iso()
    agent = ctx.agents[sc.agent or ""]
    task = _create(ctx, sc, agent)
    task_id = int(task["id"])
    final, timeline, asks, notes, outcome = _watch(ctx, sc, task_id)
    deliverables = _deliverables(ctx.api, task_id, int(agent["id"]), started, sc.id)
    reports = _reports(ctx.api, int(agent["id"]), started, sc.id)
    evidence = "\n\n".join(p for p in (
        text_of(final.get("result")), *(text_of(d) for d in deliverables), *(text_of(r) for r in reports)) if p)
    observed = {
        "status": outcome, "deliverables_min": len(deliverables), "reports_min": len(reports),
        "errors": 1 if final.get("error_message") else 0, "asks": len(asks),
    }
    expected_status = str(sc.expect_effects.get("status") or "done")
    checks = (
        Check("outcome", outcome == expected_status, f"ended '{outcome}', pack expects '{expected_status}'"),
        *effect_checks({k: v for k, v in sc.expect_effects.items() if k != "status"}, observed),
        *must_contain_checks(sc.must_contain, evidence),
    )
    errors = tuple(str(final.get("error_message")) for _ in [0] if final.get("error_message"))
    slim_task = {k: (v if k != "result" else text_of(v, RESULT_CAP)) for k, v in final.items() if k != "runtime_ref"}
    return ScenarioResult(
        id=sc.id, kind="task", started_at=started, ended_at=now_iso(), outcome=outcome,
        ok=all(c.ok for c in checks), task=slim_task, timeline=tuple(timeline),
        artifacts={"deliverables": deliverables, "reports": reports, "review_feedback": final.get("review_feedback")},
        effects=observed, asks=tuple(asks), checks=checks, errors=errors,
        execution_ids=(f"board_task:{task_id}",), notes=tuple(notes) + _denials(final),
        evidence_text=evidence, expect=sc.expect, brief=f"{sc.title}\n\n{sc.description}",
    )
