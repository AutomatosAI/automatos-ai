"""Playbook CRUD + execution handlers for PlatformActionExecutor."""

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# F182: how much of a run's stored inputs execute_playbook repeats back.
INPUTS_ECHO_CHARS = 400


def next_run_note(db, playbook_id) -> str:
    """F134: the edit reaches the next run when one is in flight (imported lazily,
    like the engine itself below)."""
    from services.playbook_engine import next_run_note as note

    return note(db, playbook_id)


def _widget_turn() -> bool:
    from core.security.surface import widget_turn

    return widget_turn()


def _playbook_visitor_view(playbook: Any) -> Dict[str, Any]:
    """F155: what a public widget turn sees of a playbook (playbooks:read):
    what it is for. Never its steps (prompts, agents, error handling, outputs),
    tags or how often it ran."""
    created_at = getattr(playbook, "created_at", None)
    return {"id": playbook.id, "name": playbook.name, "description": (playbook.description or "")[:200],
            "step_count": len(playbook.steps or []), "created_at": created_at.isoformat() if created_at else None}


def _execution_visitor_view(execution: Any) -> Dict[str, Any]:
    """F155: what a public widget turn sees of a playbook run: where it stands.
    Never its inputs, step outputs or errors."""
    return {"execution_id": execution.execution_id, "playbook_id": execution.recipe_id, "status": execution.status,
            "current_step": execution.current_step,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None}


async def list_playbooks(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate

    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )

    status_filter = params.get("status_filter", "all")
    if status_filter != "all" and hasattr(WorkflowTemplate, "status"):
        query = query.filter(WorkflowTemplate.status == status_filter)

    playbooks = query.order_by(WorkflowTemplate.id).all()
    if _widget_turn():
        return {"success": True, "playbooks": [_playbook_visitor_view(r) for r in playbooks],
                "count": len(playbooks)}

    return {
        "success": True,
        "playbooks": [
            {
                "id": r.id,
                "name": r.name,
                "template_id": r.template_id,
                "description": (r.description or "")[:200],
                "tags": r.tags or [],
                "created_at": r.created_at.isoformat() if hasattr(r, "created_at") and r.created_at else None,
            }
            for r in playbooks
        ],
        "count": len(playbooks),
    }


async def get_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate

    playbook_id = params.get("playbook_id")
    playbook_name = params.get("playbook_name")

    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )
    if playbook_id:
        query = query.filter(WorkflowTemplate.id == playbook_id)
    elif playbook_name:
        query = query.filter(WorkflowTemplate.name.ilike(f"%{playbook_name}%"))
    else:
        return {"success": False, "error": "Provide playbook_name or playbook_id"}

    playbook = query.first()
    if not playbook:
        return {"success": False, "error": "Playbook not found"}
    if _widget_turn():
        return {"success": True, "playbook": _playbook_visitor_view(playbook)}

    # Count executions
    exec_count = 0
    try:
        from core.models.core import RecipeExecution
        exec_count = (
            db.query(RecipeExecution)
            .filter(RecipeExecution.recipe_id == playbook.id)
            .count()
        )
    except Exception:
        pass

    steps = playbook.steps or []
    from core.services.playbook_inputs import contract_of

    result = {
        "success": True,
        "playbook": {
            "id": playbook.id,
            "name": playbook.name,
            "template_id": playbook.template_id,
            "description": playbook.description,
            "tags": playbook.tags or [],
            # F182: what each run needs (declared, else read from the steps)
            "inputs": contract_of(playbook),
            "step_count": len(steps),
            "steps": [
                {
                    "index": i,
                    "prompt_preview": (s.get("prompt_template", "") or "")[:120],
                    "agent_id": s.get("agent_id"),
                    "error_handling": s.get("error_handling", "stop"),
                    "output_key": s.get("output_key"),
                }
                for i, s in enumerate(steps[:10])
                if isinstance(s, dict)
            ],
            "total_executions": exec_count,
        },
    }
    namesakes = _namesakes_note(db, workspace_id, playbook, sought="the one you want")
    if namesakes:
        result["namesakes"] = namesakes
    return result


def _playbooks_called(db: Session, workspace_id: UUID, name: Any) -> List[Any]:
    """This workspace's playbooks carrying ``name`` (trimmed, any case), oldest first."""
    from core.models.core import WorkflowTemplate

    return (
        db.query(WorkflowTemplate.id, WorkflowTemplate.name)
        .filter(
            WorkflowTemplate.workspace_id == workspace_id,
            func.lower(func.trim(WorkflowTemplate.name)) == str(name).strip().lower(),
        )
        .order_by(WorkflowTemplate.id)
        .all()
    )


def _namesakes_note(db: Session, workspace_id: UUID, playbook: Any, *, sought: str) -> Optional[str]:
    """F203 (night 6): 'New Cafe Onboarding' was two playbooks, 102 and 103, and the
    record-card step was 103's; an edit of 102 found no third step and Auto said it
    had put one back. Wherever a playbook is read or a step is not there, its
    namesakes are named. None when it has none."""
    namesakes = _playbooks_called(db, workspace_id, playbook.name)
    if len(namesakes) < 2:
        return None
    ids = ", ".join(str(namesake.id) for namesake in namesakes)
    others = ", ".join(str(namesake.id) for namesake in namesakes if namesake.id != playbook.id)
    return f"{len(namesakes)} playbooks are named '{playbook.name}' (ids {ids}); {sought} may be on {others}."


def _step_out_of_range(db: Session, workspace_id: UUID, playbook: Any, step_index: int, steps: List[Any]) -> str:
    refusal = f"step_index {step_index} out of range (0-{len(steps)-1})"
    note = _namesakes_note(db, workspace_id, playbook, sought="the step you want")
    return f"{refusal}. {note}" if note else refusal


def _playbook_namesakes_refusal(namesakes: List[Any]) -> str:
    name = namesakes[0].name
    if len(namesakes) == 1:
        return (f"A playbook is already called '{name}' (id {namesakes[0].id}). Run it with "
                "platform_execute_playbook, change it with platform_update_playbook, or give the "
                "new one a different name.")
    ids = ", ".join(str(playbook.id) for playbook in namesakes)
    return (f"{len(namesakes)} playbooks are already called '{name}' (ids {ids}). Run or change one "
            "of them, or give the new one a different name.")


async def create_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    import uuid

    name = params.get("name")
    description = params.get("description")
    if not name or not description:
        return {"success": False, "error": "Missing required: name and description"}

    # F185 (night 6): asked to run the playbook it had just made, Auto made
    # another of the same name. F144's namesake rule, for playbooks.
    namesakes = _playbooks_called(db, workspace_id, name)
    if namesakes:
        return {
            "success": False,
            "existing_playbook_id": namesakes[0].id,
            "existing_playbook_ids": [playbook.id for playbook in namesakes],
            "error": _playbook_namesakes_refusal(namesakes),
        }

    tags = params.get("tags", [])
    inputs = params.get("inputs")
    if inputs is not None:  # F182: what each run needs
        from core.services.playbook_inputs import inputs_problem

        problem = inputs_problem(inputs)
        if problem:
            return {"success": False, "error": problem}
    template_id = f"custom-{uuid.uuid4().hex[:8]}"

    # F133: the person the call is made for is the playbook's creator; its later
    # edits are checked against them. Injected by the executor, never the model.
    creator = params.get("_driving_user_id")
    playbook = WorkflowTemplate(
        name=name,
        template_id=template_id,
        description=description,
        workspace_id=workspace_id,
        owner_type="workspace",
        owner_id=str(workspace_id),
        created_by="platform",
        created_by_user_id=creator if isinstance(creator, int) and not isinstance(creator, bool) else None,
        tags=tags,
        inputs=inputs,
        template_definition={"steps": [], "agents": [], "config": {}, "variables": []},
    )
    db.add(playbook)
    db.flush()

    logger.info(f"[PlatformExecutor] Created playbook '{name}' (id={playbook.id}) in workspace {workspace_id}")

    return {
        "success": True,
        "playbook": {
            "id": playbook.id,
            "name": playbook.name,
            "template_id": playbook.template_id,
            "description": playbook.description,
            "inputs": inputs,
        },
        "message": f"Playbook '{name}' created successfully. Add steps via the playbook editor.",
    }


async def update_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate

    playbook_id = params.get("playbook_id")
    if not playbook_id:
        return {"success": False, "error": "Missing required parameter: playbook_id"}

    playbook = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == playbook_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not playbook:
        return {"success": False, "error": "Playbook not found"}
    if params.get("inputs") is not None:  # checked before anything changes
        from core.services.playbook_inputs import inputs_problem

        problem = inputs_problem(params["inputs"])
        if problem:
            return {"success": False, "error": problem}

    changes = []
    if params.get("name"):
        playbook.name = params["name"]
        changes.append(f"name -> '{params['name']}'")
    if params.get("description") is not None:
        playbook.description = params["description"]
        changes.append("description updated")
    if params.get("tags") is not None:
        playbook.tags = params["tags"]
        changes.append(f"tags -> {params['tags']}")
    if params.get("execution_config") is not None:
        playbook.execution_config = params["execution_config"]
        changes.append("execution_config updated")
    if params.get("inputs") is not None:  # F182: what each run needs
        playbook.inputs = params["inputs"]
        changes.append(f"inputs -> {sorted(params['inputs'])}")
    schedule_note = None
    if params.get("schedule_config") is not None:
        from services.playbook_scheduler import SERVER_ZONE, cron_trigger, is_live_cron, with_explicit_zone

        schedule_config = with_explicit_zone(params["schedule_config"], db, workspace_id)
        if is_live_cron(schedule_config):
            try:
                cron_trigger(schedule_config["cron_expression"], schedule_config.get("timezone") or SERVER_ZONE)
            except ValueError as exc:
                return {"success": False, "error": f"scheduling failed: {exc}"}
        playbook.schedule_config = schedule_config
        changes.append("schedule_config updated")

    if not changes:
        from modules.tools.discovery.action_registry import nothing_changed

        return {"success": False, "error": nothing_changed("platform_update_playbook", "playbook_id"),
                "playbook_id": playbook.id}

    db.flush()
    if params.get("schedule_config") is not None:
        failed, schedule_note = _sync_schedule(playbook)
        if failed:
            return failed
    logger.info(f"[PlatformExecutor] Updated playbook {playbook.id}: {', '.join(changes)}")

    return {
        "success": True,
        "playbook_id": playbook.id,
        "changes": changes,
        "message": f"Playbook '{playbook.name}' updated: {', '.join(changes)}"
                   + (f". Schedule: {_schedule_text(playbook.schedule_config)} {schedule_note}" if schedule_note else "")
                   + next_run_note(db, playbook.id),
    }


async def _validate_agent_id(db: Session, workspace_id: UUID, agent_id) -> tuple:
    """Validate agent_id exists in workspace. Returns (valid_id: int | None, error: str | None)."""
    if agent_id is None:
        return None, None
    from core.models import Agent
    try:
        aid = int(agent_id)
    except (ValueError, TypeError):
        return None, f"agent_id must be an integer, got: {agent_id!r}"
    agent = db.query(Agent).filter(Agent.id == aid, Agent.workspace_id == workspace_id).first()
    # F135 (B67, B87): a switched-off agent was accepted here and then ran the step.
    if not agent or (agent.status or "active") != "active":
        valid = db.query(Agent.id, Agent.name).filter(Agent.workspace_id == workspace_id, Agent.status == "active").all()
        agent_list = ", ".join(f"{a.id}={a.name}" for a in valid[:20])
        if agent:
            return None, (f"agent_id {aid} ({agent.name}) is switched off ({agent.status}). "
                          f"Switch it on, or pick an active agent: [{agent_list}]")
        return None, f"agent_id {aid} does not exist in this workspace. Valid agents: [{agent_list}]"
    return aid, None


async def add_playbook_step(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    from sqlalchemy.orm.attributes import flag_modified
    import uuid

    playbook_id = params.get("playbook_id")
    prompt_template = params.get("prompt_template")
    if not playbook_id or not prompt_template:
        return {"success": False, "error": "Missing required: playbook_id and prompt_template"}

    agent_id, err = await _validate_agent_id(db, workspace_id, params.get("agent_id"))
    if err:
        return {"success": False, "error": err}

    playbook = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == playbook_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    steps = list(playbook.steps or [])
    order = params.get("order", len(steps))

    step = {
        "step_id": uuid.uuid4().hex[:12],
        "step_number": order + 1,
        "prompt_template": prompt_template,
        "agent_id": agent_id,
        "error_handling": params.get("error_handling", "stop"),
        "output_key": params.get("output_key"),
    }

    if order >= len(steps):
        steps.append(step)
    else:
        steps.insert(order, step)

    # Re-number all steps
    for i, s in enumerate(steps):
        s["step_number"] = i + 1

    playbook.steps = steps
    flag_modified(playbook, "steps")
    db.flush()

    logger.info(f"[PlatformExecutor] Added step to playbook {playbook.id} (now {len(steps)} steps)")

    return {
        "success": True,
        "playbook_id": playbook.id,
        "step_index": order if order < len(steps) else len(steps) - 1,
        "total_steps": len(steps),
        "message": f"Step added to playbook '{playbook.name}' (now {len(steps)} steps)." + next_run_note(db, playbook.id),
    }


async def update_playbook_step(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    from sqlalchemy.orm.attributes import flag_modified

    playbook_id = params.get("playbook_id")
    step_index = params.get("step_index")
    if playbook_id is None or step_index is None:
        return {"success": False, "error": "Missing required: playbook_id and step_index"}

    playbook = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == playbook_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    steps = list(playbook.steps or [])
    if step_index < 0 or step_index >= len(steps):
        return {"success": False, "error": _step_out_of_range(db, workspace_id, playbook, step_index, steps)}

    if "agent_id" in params and params["agent_id"] is not None:
        valid_id, err = await _validate_agent_id(db, workspace_id, params["agent_id"])
        if err:
            return {"success": False, "error": err}
        params["agent_id"] = valid_id

    step = dict(steps[step_index])  # a new dict: the stored steps are never edited in place
    changes = []

    edit, refusal = _prompt_edit(step.get("prompt_template") or "", params, step_index)
    if refusal:
        return {"success": False, "error": refusal}
    if edit is not None:
        new_prompt, said = edit
        step["prompt_template"] = new_prompt
        changes.append(said)

    for field in ("agent_id", "order", "error_handling", "output_key"):
        if field in params and params[field] is not None:
            step[field] = params[field]
            changes.append(f"{field} updated")

    if not changes:
        from modules.tools.discovery.action_registry import nothing_changed

        return {"success": False, "playbook_id": playbook.id,
                "error": nothing_changed("platform_update_playbook_step", "playbook_id", "step_index")}

    steps[step_index] = step
    playbook.steps = steps
    flag_modified(playbook, "steps")
    db.flush()

    logger.info(f"[PlatformExecutor] Updated step {step_index} of playbook {playbook.id}: {', '.join(changes)}")

    return {
        "success": True,
        "playbook_id": playbook.id,
        "step_index": step_index,
        "changes": changes,
        "message": f"Step {step_index} of '{playbook.name}' updated: {', '.join(changes)}." + next_run_note(db, playbook.id),
    }


def _prompt_edit(current: str, params: Dict[str, Any], step_index: int):
    """The step prompt's new text and what the reply says about it, or a refusal.

    F134 (night 4): "update this step" re-sent the whole prompt from memory and
    dropped its safety lines (B79, B84). find/replace changes one passage and keeps
    the rest; a whole-prompt overwrite still works, and its reply names every
    line it dropped, so nothing goes silently.
    Returns ((new_text, change_note) or None, refusal or None).
    """
    find, replace, whole = params.get("find"), params.get("replace"), params.get("prompt_template")
    if find is None:
        if whole is None:
            return None, None
        dropped = _dropped_lines(current, whole)
        if not dropped:
            return (whole, "prompt_template replaced"), None
        shown = "; ".join(repr(line[:120]) for line in dropped[:10])
        more = f" (+{len(dropped) - 10} more)" if len(dropped) > 10 else ""
        return (whole, f"prompt_template replaced, and it dropped {len(dropped)} "
                       f"line{'s' if len(dropped) != 1 else ''}: {shown}{more}"), None
    if whole is not None:
        return None, "Pass find/replace or prompt_template, not both. Nothing changed."
    if replace is None:
        return None, "find needs replace (an empty string removes the text). Nothing changed."
    found = current.count(find) if find else 0
    if found != 1:
        where = "is not in" if found == 0 else f"appears {found} times in"
        return None, (f"The find text {where} step {step_index}'s prompt, so nothing changed. "
                      f"The prompt reads: {current[:300]!r}")
    return (current.replace(find, replace, 1), "prompt_template: one passage replaced"), None


def _dropped_lines(before: str, after: str) -> List[str]:
    kept = {line.strip() for line in after.splitlines()}
    return [line.strip() for line in before.splitlines() if line.strip() and line.strip() not in kept]


async def delete_playbook_step(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import WorkflowTemplate
    from sqlalchemy.orm.attributes import flag_modified

    playbook_id = params.get("playbook_id")
    step_index = params.get("step_index")
    if playbook_id is None or step_index is None:
        return {"success": False, "error": "Missing required: playbook_id and step_index"}

    playbook = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == playbook_id,
            WorkflowTemplate.workspace_id == workspace_id,
        )
        .first()
    )
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    steps = list(playbook.steps or [])
    if step_index < 0 or step_index >= len(steps):
        return {"success": False, "error": _step_out_of_range(db, workspace_id, playbook, step_index, steps)}

    removed = steps.pop(step_index)

    # Re-number remaining steps
    for i, s in enumerate(steps):
        s["step_number"] = i + 1

    playbook.steps = steps
    flag_modified(playbook, "steps")
    db.flush()

    logger.info(f"[PlatformExecutor] Deleted step {step_index} from playbook {playbook.id} (now {len(steps)} steps)")

    return {
        "success": True,
        "playbook_id": playbook.id,
        "deleted_step_index": step_index,
        "remaining_steps": len(steps),
        "message": (f"Step {step_index} removed from '{playbook.name}' ({len(steps)} steps remaining)."
                    + next_run_note(db, playbook.id)),
    }


async def schedule_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Set a cron schedule on a playbook so it runs automatically."""
    from core.models.core import WorkflowTemplate

    playbook_id = params.get("playbook_id")
    playbook_name = params.get("playbook_name")
    cron_expression = params.get("cron_expression")

    if not cron_expression:
        return {"success": False, "error": "Missing required parameter: cron_expression"}

    # Validate cron expression
    parts = cron_expression.strip().split()
    if len(parts) != 5:
        return {"success": False, "error": f"Invalid cron expression: expected 5 fields, got {len(parts)}. Format: minute hour day_of_month month day_of_week"}

    # Resolve playbook
    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )
    if playbook_id:
        query = query.filter(WorkflowTemplate.id == playbook_id)
    elif playbook_name:
        query = query.filter(WorkflowTemplate.name.ilike(f"%{playbook_name}%"))
    else:
        return {"success": False, "error": "Provide playbook_id or playbook_name"}

    playbook = query.first()
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    from services.playbook_scheduler import cron_trigger, default_schedule_zone

    timezone = params.get("timezone") or default_schedule_zone(db, workspace_id)
    enabled = params.get("enabled", True)
    try:
        cron_trigger(cron_expression, timezone)
    except ValueError as exc:
        return {"success": False, "error": f"scheduling failed: {exc}"}

    schedule_config = {
        "type": "cron",
        "cron_expression": cron_expression,
        "timezone": timezone,
        "enabled": enabled,
    }
    playbook.schedule_config = schedule_config
    db.flush()

    # F132: this called a fresh, never-started scheduler with the wrong arguments and
    # swallowed the TypeError, so Auto's schedules reached no scheduler at all.
    failed, schedule_note = _sync_schedule(playbook)
    if failed:
        return failed

    logger.info(
        "[PlatformExecutor] Scheduled playbook '%s' (id=%d) with cron '%s' tz=%s enabled=%s",
        playbook.name, playbook.id, cron_expression, timezone, enabled,
    )

    # F132 (night 6): "7am Monday" from a UK owner was saved in UTC — no zone was
    # passed, and a new workspace has none. The reply says what was assumed.
    zone_note = "" if params.get("timezone") else ZONE_ASSUMED.format(zone=timezone)
    return {
        "success": True,
        "playbook_id": playbook.id,
        "playbook_name": playbook.name,
        "schedule_config": schedule_config,
        "timezone_given": bool(params.get("timezone")),
        "message": f"Playbook '{playbook.name}' scheduled: {cron_expression} in {timezone}. {schedule_note}{zone_note}",
    }


ZONE_ASSUMED = (
    " No timezone was given, so it fires in {zone}, the workspace's default. Tell the owner the "
    "time is {zone}; if that is not where they are, ask for their zone and schedule it again "
    "with timezone set."
)


def _inputs_text(input_data: Dict[str, Any]) -> str:
    """A run's stored inputs as the caller is told them: ``none`` when empty."""
    if not input_data:
        return "none"
    text = json.dumps(input_data, default=str, ensure_ascii=False)
    return text if len(text) <= INPUTS_ECHO_CHARS else text[:INPUTS_ECHO_CHARS] + "…"


def _schedule_text(schedule_config) -> str:
    sc = schedule_config or {}
    if sc.get("type") != "cron":
        return f"{sc.get('type') or 'none'}."
    return f"{sc.get('cron_expression')} in {sc.get('timezone')}."


def _sync_schedule(playbook) -> tuple:
    """Register the playbook's saved schedule with the scheduler (F132).

    Returns (failure_reply, None) when the scheduler refused it — the reply says
    "scheduling failed", never "scheduled" — else (None, what the owner is told).
    """
    from services.playbook_scheduler import (
        SYNC_DEFERRED, SYNC_OFF, SYNC_REMOVED, sync_playbook_schedule,
    )
    from services.scheduled_task_service import RECONCILE_INTERVAL_SECONDS

    try:
        outcome = sync_playbook_schedule(playbook)
    except Exception as exc:  # noqa: BLE001 — reported to the owner, never swallowed
        logger.error("[PlatformExecutor] Scheduler sync failed for playbook %d: %s", playbook.id, exc, exc_info=True)
        return {"success": False, "playbook_id": playbook.id, "error": f"scheduling failed: {exc}"}, None
    sc = playbook.schedule_config or {}
    removed = ("Paused — set enabled=true to activate." if sc.get("type") == "cron"
               else "It runs only when started.")
    notes = {
        SYNC_REMOVED: removed,
        SYNC_DEFERRED: f"Active: the scheduler picks it up within {RECONCILE_INTERVAL_SECONDS} s.",
        SYNC_OFF: "Saved, but scheduled runs are switched off on this server (RECIPE_SCHEDULER_ENABLED).",
    }
    return None, notes.get(outcome, "Active now.")


async def execute_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger a playbook run asynchronously. Returns execution_id immediately."""
    from core.models.core import WorkflowTemplate, RecipeExecution
    import uuid

    playbook_id = params.get("playbook_id")
    playbook_name = params.get("playbook_name")
    # F113: a string (or an `inputs` / `input` alias) is read as key-value pairs;
    # a value that cannot be one is refused here, not crashed on in the run.
    from core.services.playbook_inputs import playbook_inputs

    raw_input = params.get("input_data")
    if raw_input is None:
        raw_input = params.get("inputs", params.get("input"))
    input_data, input_problem = playbook_inputs(raw_input)
    if input_problem:
        return {"success": False, "error": input_problem}

    # Resolve playbook
    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )
    if playbook_id:
        query = query.filter(WorkflowTemplate.id == playbook_id)
    elif playbook_name:
        query = query.filter(WorkflowTemplate.name.ilike(f"%{playbook_name}%"))
    else:
        return {"success": False, "error": "Provide playbook_id or playbook_name"}

    playbook = query.first()
    if not playbook:
        return {"success": False, "error": "Playbook not found"}

    # F182 (night 6): a run the call would start without an input it needs is
    # not started. The caller is told which, and asks the owner in its chat
    # (the run itself would stop and ask through Questions).
    from core.services.playbook_inputs import contract_of, inputs_needed_error, missing_inputs, with_defaults

    contract = contract_of(playbook)
    needed = missing_inputs(contract, with_defaults(contract, input_data))
    if needed:
        return {"success": False, "error": inputs_needed_error(playbook.name, needed, contract)}

    # Concurrency guard -- return error to agent if workspace is at capacity
    from services.concurrency_guard import check_concurrency
    concurrency = await check_concurrency(workspace_id, db)
    if not concurrency.allowed:
        logger.warning(
            "[PlatformExecutor] Concurrency limit reached for workspace %s: %s",
            workspace_id, concurrency.reason,
        )
        return {"status": "error", "error": concurrency.refusal}

    # Create execution record. F155: a run a widget turn starts records the
    # turn's origin, and its steps run under it (api.recipe_executor).
    from core.security.surface import stamp_origin

    execution_id = f"exec-{uuid.uuid4().hex[:12]}"
    execution = RecipeExecution(
        execution_id=execution_id,
        recipe_id=playbook.id,
        workspace_id=workspace_id,
        status="pending",
        input_data=input_data,
        triggered_by="platform_action",
        execution_metadata=stamp_origin(None) or None,
    )
    db.add(execution)
    db.commit()  # Must commit before async task (it opens its own session)

    # PRD-204 S9 (Q1): Auto-launched playbook runs get a run_and_report
    # watch by default (workspace setting watch_auto_create, default ON);
    # criteria seeded from the request context. Fail-soft by contract --
    # committed separately AFTER the execution row so a watch problem can
    # never poison the launch transaction.
    try:
        from modules.tools.discovery.handlers_watches import (
            _origin_chat_id,
            auto_create_watch,
        )

        criteria = f"Playbook '{playbook.name}' completes and delivers its expected output."
        if input_data:
            import json as _json

            try:
                criteria += f" Inputs: {_json.dumps(input_data, default=str)[:400]}"
            except Exception:
                pass
        watch = auto_create_watch(
            db,
            workspace_id,
            target_type="playbook_execution",
            target_id=execution_id,
            title=f"Watch: {playbook.name[:80]}",
            origin_chat_id=_origin_chat_id(params),
            success_criteria=criteria,
            created_by=(str(params.get("_created_by")) if params.get("_created_by") else None),
            owner_agent_id=(
                int(params["_agent_id"])
                if str(params.get("_agent_id") or "").isdigit()
                else None
            ),
        )
        if watch is not None:
            db.commit()
    except Exception:
        db.rollback()
        logger.warning(
            "[PlatformExecutor] watch auto-create commit failed for %s -- "
            "launch unaffected", execution_id, exc_info=True,
        )

    # Launch async execution via the consolidated PlaybookEngine seam.
    # PRD-142 W3-S12: every backend launch site goes through the engine —
    # the strangler-fig that lets durability/observability land in one place.
    try:
        from services.playbook_engine import get_playbook_engine
        get_playbook_engine().launch(
            recipe_execution_id=execution_id,
            recipe_id=playbook.id,
            workspace_id=workspace_id,
            input_data=input_data,
        )
    except Exception as e:
        logger.error("[PlatformExecutor] Failed to launch playbook task: %s", e)
        # Mark execution as failed so it doesn't stay "pending" forever
        execution.status = "failed"
        execution.error_message = f"Failed to enqueue: {str(e)[:500]}"
        db.commit()
        return {"success": False, "error": f"Playbook triggered but failed to launch: {str(e)[:200]}"}

    logger.info(
        "[PlatformExecutor] Triggered playbook '%s' (id=%d) -- execution_id=%s",
        playbook.name, playbook.id, execution_id,
    )

    # F182 (night 6): the run's inputs, as stored, go back to the caller. Run 207
    # started with none and Auto told the owner it was running for Gull & Anchor.
    return {
        "success": True,
        "execution_id": execution_id,
        "playbook_id": playbook.id,
        "playbook_name": playbook.name,
        "status": "pending",
        "input_data": input_data,
        "message": (f"Playbook '{playbook.name}' triggered with inputs: {_inputs_text(input_data)}. "
                    f"Track with execution_id: {execution_id}"),
    }


async def get_playbook_execution(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Check status/results of a playbook execution."""
    from core.models.core import RecipeExecution

    execution_id = params.get("execution_id")
    playbook_id = params.get("playbook_id")

    if execution_id:
        execution = (
            db.query(RecipeExecution)
            .filter(
                RecipeExecution.execution_id == execution_id,
                RecipeExecution.workspace_id == workspace_id,
            )
            .first()
        )
        if not execution:
            return {"success": False, "error": f"Execution '{execution_id}' not found"}
        if _widget_turn():
            return {"success": True, "execution": _execution_visitor_view(execution)}

        # Summarize step_results (200 char preview per step)
        step_summaries = []
        for i, step in enumerate(execution.step_results or []):
            if isinstance(step, dict):
                output = str(step.get("output", step.get("result", "")))[:200]
                step_summaries.append({
                    "step": i,
                    "status": step.get("status", "unknown"),
                    "output_preview": output,
                })

        return {
            "success": True,
            "execution": {
                "execution_id": execution.execution_id,
                "playbook_id": execution.recipe_id,
                "status": execution.status,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "error_message": execution.error_message,
                "step_results": step_summaries,
                "current_step": execution.current_step,
            },
        }

    elif playbook_id:
        # List recent executions for this playbook
        executions = (
            db.query(RecipeExecution)
            .filter(
                RecipeExecution.recipe_id == playbook_id,
                RecipeExecution.workspace_id == workspace_id,
            )
            .order_by(RecipeExecution.started_at.desc())
            .limit(5)
            .all()
        )
        if _widget_turn():
            return {"success": True, "executions": [_execution_visitor_view(e) for e in executions],
                    "count": len(executions)}

        return {
            "success": True,
            "executions": [
                {
                    "execution_id": e.execution_id,
                    "status": e.status,
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                    "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                    "error_message": e.error_message,
                }
                for e in executions
            ],
            "count": len(executions),
        }

    return {"success": False, "error": "Provide execution_id or playbook_id"}


async def delete_playbook(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a playbook with full cleanup."""
    from core.models.core import WorkflowTemplate

    playbook_id = params.get("playbook_id")
    playbook_name = params.get("playbook_name")

    query = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.workspace_id == workspace_id
    )
    if playbook_id:
        playbook = query.filter(WorkflowTemplate.id == playbook_id).first()
        if not playbook:
            return {"success": False, "error": (
                f"No playbook #{playbook_id} in this workspace — nothing was deleted. "
                "List them (platform_list_playbooks) for the right id.")}
    elif playbook_name:
        # F185: a delete names one playbook. The name used to match any playbook
        # containing it and deleted the first; with namesakes that is a guess.
        named = _playbooks_called(db, workspace_id, playbook_name)
        if len(named) > 1:
            ids = ", ".join(str(match.id) for match in named)
            return {"success": False, "error": (
                f"{len(named)} playbooks are called '{named[0].name}' (ids {ids}) — nothing was "
                "deleted. Delete one by its playbook_id.")}
        if not named:
            return {"success": False, "error": (
                f"No playbook is called '{playbook_name}' in this workspace — nothing was deleted. "
                "List them (platform_list_playbooks) and delete by playbook_id.")}
        playbook = query.filter(WorkflowTemplate.id == named[0].id).first()
    else:
        return {"success": False, "error": "Provide playbook_id or playbook_name"}

    # Guard against system playbooks
    if getattr(playbook, "is_system", False):
        return {"success": False, "error": "System playbooks cannot be deleted"}

    playbook_info = {"id": playbook.id, "name": playbook.name}
    cleanup_notes = []

    # Trigger subscription cleanup (non-fatal)
    try:
        from api.workflow_recipes import _cleanup_trigger_subscriptions
        _cleanup_trigger_subscriptions(playbook.id, db)
        cleanup_notes.append("Trigger subscriptions deactivated")
    except Exception as e:
        logger.warning("[PlatformExecutor] Trigger cleanup failed for playbook %d: %s", playbook.id, e)
        cleanup_notes.append(f"Trigger cleanup failed: {e}")

    # Durable memory cleanup (non-fatal). The old HTTP cleanup deleted a
    # "playbook-{id}" namespace no writer ever used — this erases the real
    # buckets the playbook memory writer stores under: the recipe namespace
    # plus one recipe_agent namespace per agent in the steps (the same set
    # retrieve_relevant_memories enumerates).
    try:
        from modules.memory.unified_memory_service import get_unified_memory_service

        svc = get_unified_memory_service()
        ns = svc.namespace(str(playbook.workspace_id))
        playbook_key = playbook.template_id or str(playbook.id)
        erased = await svc._durable.erase_namespace(ns.recipe(playbook_key))
        agent_ids = {
            step.get("agent_id")
            for step in (playbook.steps or [])
            if isinstance(step, dict) and step.get("agent_id")
        }
        for aid in agent_ids:
            erased += await svc._durable.erase_namespace(ns.recipe_agent(playbook_key, aid))
        cleanup_notes.append(f"Playbook memories cleaned up ({erased})")
    except Exception as e:
        logger.warning("[PlatformExecutor] Durable-memory cleanup failed for playbook %d: %s", playbook.id, e)

    # Delete the playbook (cascades to executions via FK)
    db.delete(playbook)
    db.flush()
    cleanup_notes.append("Database record deleted")

    logger.info("[PlatformExecutor] Deleted playbook %s -- %s", playbook_info, ", ".join(cleanup_notes))

    return {
        "success": True,
        "deleted_playbook": playbook_info,
        "cleanup": cleanup_notes,
        "message": f"Playbook '{playbook_info['name']}' (ID {playbook_info['id']}) deleted.",
    }
