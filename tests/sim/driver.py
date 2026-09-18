"""Scenario dispatch (PRD-247 S0.2) and the *crud* scenario.

``run_scenario`` never raises: a scenario that blows up becomes a result with
``outcome="error"`` carrying the exception, so one bad scenario cannot take
the night down. The task and chat drivers live in their own modules.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

from .api import ApiError, items_of
from .driver_chat import run_chat
from .driver_task import run_task
from .packs import Scenario
from .results import Check, RunContext, ScenarioResult, now_iso, slug

CRUD_AGENT_TYPE = "custom"


def _timed(fn: Callable[[], Any]) -> tuple[Any, int]:
    started = time.monotonic()
    value = fn()
    return value, int((time.monotonic() - started) * 1000)


def _run_step(label: str, fn: Callable[[], Any], verify: Callable[[Any], tuple[bool, str]],
              steps: list[dict[str, Any]], checks: list[Check], errors: list[str]) -> Any:
    """Time one API step, record it, and turn a refusal into a failed check rather than an exception."""
    try:
        value, ms = _timed(fn)
        ok, detail = verify(value)
        steps.append({"step": label, "ms": ms, "ok": ok, "detail": detail})
        checks.append(Check(f"crud:{label}", ok, detail))
        return value
    except ApiError as exc:
        steps.append({"step": label, "ms": None, "ok": False, "detail": f"HTTP {exc.status}: {exc.body[:200]}"})
        checks.append(Check(f"crud:{label}", False, f"HTTP {exc.status}"))
        errors.append(f"{label}: HTTP {exc.status} {exc.body[:200]}")
        return None


def _verify_gone(api: Any, sc: Scenario, agent_id: Any, steps: list[dict[str, Any]], checks: list[Check]) -> None:
    gone = api.request("GET", f"/api/agents/{agent_id}", label=f"{sc.id}:get-after-delete")
    checks.append(Check("crud:gone", gone.status == 404, f"GET after delete returned {gone.status}"))
    steps.append({"step": "gone", "ms": gone.ms, "ok": gone.status == 404, "detail": f"HTTP {gone.status}"})


def run_crud(ctx: RunContext, sc: Scenario) -> ScenarioResult:
    """Create → update → get → list → delete an agent; each step is timed and checked."""
    started = now_iso()
    api = ctx.api
    name = f"SIM crud {slug(sc.id)}"
    steps: list[dict[str, Any]] = []
    checks: list[Check] = []
    errors: list[str] = []
    agent_id: Any = None

    def step(label: str, fn: Callable[[], Any], verify: Callable[[Any], tuple[bool, str]]) -> Any:
        return _run_step(label, fn, verify, steps, checks, errors)

    for name_of_step in sc.steps:
        if name_of_step == "create":
            created = step("create", lambda: api.post("/api/agents/", {
                "name": name, "description": sc.description or "created by the simulation", "agent_type": CRUD_AGENT_TYPE,
                "configuration": {}, "tags": ["sim", sc.id]}, label=f"{sc.id}:create"),
                lambda v: (isinstance(v, dict) and "id" in v, "created with an id" if isinstance(v, dict) and "id" in v else "no id in response"))
            agent_id = created.get("id") if isinstance(created, dict) else None
        elif name_of_step == "update" and agent_id is not None:
            step("update", lambda: api.patch(f"/api/agents/{agent_id}", {"description": "updated by the simulation"},
                                             label=f"{sc.id}:update"),
                 lambda v: (isinstance(v, dict) and v.get("description") == "updated by the simulation",
                            "description echoed back" if isinstance(v, dict) and v.get("description") == "updated by the simulation" else f"description not echoed: {str(v)[:120]}"))
        elif name_of_step == "get" and agent_id is not None:
            step("get", lambda: api.get(f"/api/agents/{agent_id}", label=f"{sc.id}:get"),
                 lambda v: (isinstance(v, dict) and v.get("id") == agent_id, "found by id"))
        elif name_of_step == "list" and agent_id is not None:
            step("list", lambda: api.get("/api/agents/", label=f"{sc.id}:list"),
                 lambda v: (any(isinstance(a, dict) and a.get("id") == agent_id for a in items_of(v, "agents")),
                            "listed" if any(isinstance(a, dict) and a.get("id") == agent_id for a in items_of(v, "agents")) else "created agent missing from the list"))
        elif name_of_step == "delete" and agent_id is not None:
            step("delete", lambda: api.delete(f"/api/agents/{agent_id}", label=f"{sc.id}:delete"), lambda v: (True, "deleted"))
            _verify_gone(api, sc, agent_id, steps, checks)
        else:
            checks.append(Check(f"crud:{name_of_step}", False, "skipped: no agent id from create"))
    ok = all(c.ok for c in checks) and not errors
    return ScenarioResult(
        id=sc.id, kind="crud", started_at=started, ended_at=now_iso(), outcome="ok" if ok else "error", ok=ok,
        steps=tuple(steps), checks=tuple(checks), errors=tuple(errors),
        effects={"steps": len(steps), "errors": len(errors), "ms_total": sum(s["ms"] or 0 for s in steps)},
        brief=f"agent CRUD: {', '.join(sc.steps)}",
    )


RUNNERS = {"task": run_task, "chat": run_chat, "crud": run_crud}


def run_scenario(ctx: RunContext, sc: Scenario) -> ScenarioResult:
    started = now_iso()
    try:
        return RUNNERS[sc.kind](ctx, sc)
    except Exception as exc:  # noqa: BLE001 — one scenario must not end the night
        return ScenarioResult(
            id=sc.id, kind=sc.kind, started_at=started, ended_at=now_iso(), outcome="error", ok=False,
            errors=(f"{type(exc).__name__}: {exc}",), notes=(traceback.format_exc()[-2000:],),
            brief=sc.title or "\n".join(sc.turns), expect=sc.expect,
        )
