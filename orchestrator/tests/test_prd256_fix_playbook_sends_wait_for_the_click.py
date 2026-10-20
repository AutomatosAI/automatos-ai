"""PRD-256 P256-FIX-RVW-2: a playbook step's LinkedIn image post waits for the owner's click.

P256-FIX-T6 (F401): the step records the call as the card's own words, waiting ("Card raised: …",
never "Error executing …: Card raised: …"), so the step is not failed and the run is not either; its
output ends with what waits for the owner's click. A refused call keeps "Error executing …".

P256-FIX-RVW-39: a watch's rerun of a run Auto started (a new run, triggered_by
'watch_rerun', ``retry_of`` the first) is on Auto's brief too: its step's send asks.

The step ran LINKEDIN_CREATE_LINKED_IN_POST with images through the LinkedIn direct API
itself (api/recipe_executor ``_execute_step``) and never reached
``owner_only.asks_before_a_send``, so a step of a run Auto started (FX-011 AC3) posted
with no card. Now the step dispatches it through the spine like every other Composio
action, and ComposioToolExecutor's own workaround posts it, under the gate.

The spine had a second hole: ``ToolRouter.execute_and_format`` sends a Composio call with
an intent through ``execute_tool_with_validation``, which dropped the caller's context, so
the gate never saw the step's run (nor a chat's driving user). The router call's context is
now held for the call (``held_context``) and forwarded; a call the step makes with no
context (the model's own ``composio_execute``) is made for the step's run.

The chain below is real from the step to the gate: ``_execute_step`` (its LLM, agent and
tool search stubbed), the router, the validation path, the module ``execute_tool``, the
owner's-click gate and ``ComposioToolExecutor.execute`` (its entity and the LinkedIn API
stubbed); the executor around them carries the real gate, as in FX-011's tests.
"""
from __future__ import annotations

import ast
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from api.recipe_step_outcome import WAITING, WAITING_HEADER, WAITING_ON
from modules.tools.discovery.agent_sends import AGENT_SEND
from modules.tools.discovery.agent_sends import CARD_RAISED as SEND_CARD_RAISED
from modules.tools.discovery.owner_only import asks_before_a_send
from tests.test_prd251_composio_deny import _execution_sites, _function, _Playbook
from tests.test_prd256_fix_agent_sends_wait_for_the_click import _auto_run, _click, _grants, _told

LINKEDIN_POST = "LINKEDIN_CREATE_LINKED_IN_POST"
IMAGES = {"text": "Launch day", "images": ["/workspace/launch.png"]}
POSTED = {"success": True, "data": {"id": "urn:li:share:1"}, "error": None}
STEP = ("api/recipe_executor.py", "_execute_step")


class _Executor:
    """UnifiedToolExecutor's shape: the real owner's-click gate over the real
    ComposioToolExecutor.execute, whose entity and LinkedIn API calls are stubbed."""

    composio = None

    def __init__(self, db):
        self.db = db

    def _resolve_effective_call(self, tool_name, parameters):
        return parameters.get("action"), parameters.get("params"), True

    @asks_before_a_send
    async def execute_tool(self, tool_name, parameters, agent_id=0, tenant_id=None, workspace_id=None,
                           trace_id=None, caller_context=None):
        return await _Executor.composio.execute(
            action=parameters["action"], params=dict(parameters["params"]), agent_id=agent_id,
            workspace_id=workspace_id, skip_validation=True,
        )


@pytest.fixture
def desk(db_session, seed_workspace, monkeypatch):
    """A workspace with Auto and the step's agent; one playbook step that posts to LinkedIn."""
    import core.composio.deny_list as deny_list
    import core.composio.linkedin_image_workaround as lw
    import core.composio.post_gate as post_gate
    import core.composio.tool_executor as tool_executor
    import core.database.database as database
    import modules.tools.execution.tool_grants as tool_grants
    import modules.tools.execution.unified_executor as unified_executor
    import modules.tools.tool_router as tr
    from core.models.core import Agent

    resolve = tool_executor.resolve_file_uploads   # the real, decorated one: the step keeps it
    step = _Playbook(monkeypatch, LINKEDIN_POST, IMAGES)
    monkeypatch.setattr(tool_executor, "resolve_file_uploads", resolve)
    monkeypatch.setattr(tr, "get_tool_router", tr.ToolRouter)
    monkeypatch.setattr(tr, "composio_available", lambda: True)
    monkeypatch.setattr(tr, "validate_action_for_intent", lambda **kwargs: (True, ""))
    monkeypatch.setattr(database, "SessionLocal", MagicMock(name="SessionLocal"))
    monkeypatch.setattr(tr, "_get_executor_for_request", lambda session: _Executor(db_session))
    monkeypatch.setattr(unified_executor, "UnifiedToolExecutor", _Executor)
    for module in (deny_list, tool_executor):
        monkeypatch.setattr(module, "composio_action_denial_async", AsyncMock(return_value=None))
    for module in (post_gate, tool_executor):
        monkeypatch.setattr(module, "post_action_refusal", AsyncMock(return_value=None))
    monkeypatch.setattr(tool_grants, "_notify_approval_pending", lambda grant, ws: None)
    cache = MagicMock(name="db")
    cache.query.return_value.filter.return_value.first.return_value = None   # app name from the slug: LINKEDIN
    composio = tool_executor.ComposioToolExecutor(db=cache, client=MagicMock(name="client"))
    monkeypatch.setattr(composio, "get_entity_for_workspace", lambda ws: {"composio_entity_id": "entity-1"})
    monkeypatch.setattr(_Executor, "composio", composio)
    linkedin = AsyncMock(name="execute_linkedin_image_post", return_value=POSTED)
    monkeypatch.setattr(lw, "execute_linkedin_image_post", linkedin)

    ws = UUID(seed_workspace())
    common = dict(description="", status="active", configuration={}, workspace_id=ws, created_by="test",
                  owner_type="workspace", owner_id=str(ws))
    auto = Agent(name="Auto", slug=f"auto-{ws}", agent_type="system", is_system_agent=True, **common)
    poster = Agent(name="LAUNCH POSTER", agent_type="chatbot", **common)
    db_session.add_all([auto, poster])
    db_session.flush()
    return NS(db=db_session, ws=ws, poster=poster, step=step, linkedin=linkedin)


def _calls_composio_execute(desk):
    """The model calls the generic composio_execute itself (the step's hint path, or any call
    outside the SDK's matches): the step dispatches it with no context of its own."""
    import json

    (call,) = desk.step.responses[0].tool_calls
    call["function"] = {"name": "composio_execute",
                        "arguments": json.dumps({"action": LINKEDIN_POST, "params": IMAGES})}


def _run_step(desk, run):
    from api import recipe_executor

    result = asyncio.run(recipe_executor._execute_step(
        db=MagicMock(name="db"), agent=NS(id=desk.poster.id, name="LAUNCH POSTER"),
        clean_prompt="Post the launch to LinkedIn", workspace_id=desk.ws, step_order=2,
        max_iterations=3, recipe_execution_id=run,
    ))
    (call,) = result["execution"]["tool_calls"]
    return call


# ── The step on a run Auto started asks; nothing posts ───────────────────────────────

def test_a_step_of_an_auto_started_run_raises_the_card_and_posts_nothing(desk):
    call = _run_step(desk, _auto_run(desk))

    desk.linkedin.assert_not_called()
    assert call["result"].startswith("Card raised: ") and call["success"] is False
    assert call[WAITING] is True                                   # F401: waiting, not "Error executing …"
    (grant,) = _grants(desk)
    assert grant.details[AGENT_SEND]["lane"] == "playbook"
    told = {**IMAGES, "text": _told(desk, IMAGES["text"])}   # PRD-264 EUAI-07: the post starts with the AI line
    assert grant.details["params"] == {"action": LINKEDIN_POST, "params": told}


def test_the_owners_click_posts_once_with_the_images_as_given(desk):
    from api.approval_grants import _resume_tool_call

    _run_step(desk, _auto_run(desk))
    (grant,) = _grants(desk)
    _click(desk, grant)

    desk.linkedin.assert_called_once()
    assert desk.linkedin.call_args.kwargs["params"]["images"] == IMAGES["images"]
    assert grant.details["executed_result"]["success"] is True
    asyncio.run(_resume_tool_call(desk.db, grant))       # the same click replayed: nothing posts again
    desk.linkedin.assert_called_once()
    assert len(_grants(desk)) == 1


def test_the_models_own_composio_execute_in_an_auto_started_step_asks_too(desk):
    _calls_composio_execute(desk)
    run = _auto_run(desk)
    call = _run_step(desk, run)

    desk.linkedin.assert_not_called()
    assert call["result"].startswith("Card raised: ") and call[WAITING] is True
    (grant,) = _grants(desk)
    assert grant.details[AGENT_SEND]["lane"] == "playbook"
    assert grant.details[AGENT_SEND]["context"] == {"playbook_execution_id": run}


# ── A run a person started posts as before ──────────────────────────────────────────

def test_a_step_of_a_run_a_person_started_posts_as_before(desk):
    call = _run_step(desk, _auto_run(desk, triggered_by="user@example.com"))

    desk.linkedin.assert_called_once()
    assert desk.linkedin.call_args.kwargs["params"]["images"] == IMAGES["images"]   # never an s3key
    assert "urn:li:share:1" in call["result"] and call["success"] is True and WAITING not in call
    assert _grants(desk) == []


def test_the_models_own_composio_execute_in_a_step_a_person_started_posts(desk):
    _calls_composio_execute(desk)
    _run_step(desk, _auto_run(desk, triggered_by="user@example.com"))

    desk.linkedin.assert_called_once()
    assert _grants(desk) == []


# ── A watch's rerun of a run is on the brief of the run it reruns (P256-FIX-RVW-39) ────

def _rerun(desk, of):
    """The watch's corrective rerun (services/watch_rerun ``create_rerun_execution``): a NEW
    run, triggered_by 'watch_rerun', whose ``retry_of`` names the run it reruns."""
    from core.models.core import RecipeExecution
    from services.watch_rerun import TRIGGERED_BY_WATCH

    original = desk.db.query(RecipeExecution).filter(RecipeExecution.execution_id == of).one()
    run = f"rerun-{uuid4().hex[:12]}"
    desk.db.add(RecipeExecution(execution_id=run, recipe_id=original.recipe_id, workspace_id=desk.ws,
                                status="running", input_data={}, attempt_count=(original.attempt_count or 1) + 1,
                                triggered_by=TRIGGERED_BY_WATCH, retry_of=of))
    desk.db.flush()
    return run


def test_a_watchs_rerun_of_an_auto_started_run_raises_the_card_and_posts_nothing(desk):
    rerun = _rerun(desk, _auto_run(desk))
    call = _run_step(desk, rerun)

    desk.linkedin.assert_not_called()
    assert call["result"].startswith("Card raised: ") and call["success"] is False and call[WAITING] is True
    (grant,) = _grants(desk)
    assert grant.details[AGENT_SEND]["context"] == {"playbook_execution_id": rerun}


def test_a_rerun_of_a_rerun_of_an_auto_started_run_asks_too(desk):
    call = _run_step(desk, _rerun(desk, _rerun(desk, _auto_run(desk))))

    desk.linkedin.assert_not_called()
    assert call["result"].startswith("Card raised: ") and call[WAITING] is True
    assert len(_grants(desk)) == 1


def test_a_watchs_rerun_of_a_run_a_person_started_posts(desk):
    call = _run_step(desk, _rerun(desk, _auto_run(desk, triggered_by="user@example.com")))

    desk.linkedin.assert_called_once()
    assert call["success"] is True
    assert _grants(desk) == []


def test_the_rerun_chain_is_read_in_the_runs_own_workspace_only(desk, seed_workspace):
    """A run whose retry_of names another workspace's Auto-started run is not on this Auto's brief."""
    from modules.tools.discovery.agent_sends import _launched_by_auto

    elsewhere = UUID(seed_workspace())
    theirs = _auto_run(NS(db=desk.db, ws=elsewhere))
    ours = _rerun(desk, theirs)

    assert _launched_by_auto(desk.db, elsewhere, theirs) is True
    assert _launched_by_auto(desk.db, desk.ws, ours) is False


def test_the_rerun_chain_is_bounded_and_fails_closed(desk):
    """A chain deeper than RERUN_CHAIN_DEPTH, or a loop, stops reading and asks; a chain that
    ends at a run a person started, however long, does not."""
    from core.models.core import RecipeExecution
    from modules.tools.discovery import agent_sends

    first = _auto_run(desk, triggered_by="user@example.com")
    run = first
    for _ in range(agent_sends.RERUN_CHAIN_DEPTH - 1):
        run = _rerun(desk, run)
    assert agent_sends._launched_by_auto(desk.db, desk.ws, run) is False      # its root is read: a person's
    assert agent_sends._launched_by_auto(desk.db, desk.ws, _rerun(desk, run)) is True   # one past: asks
    looped = desk.db.query(RecipeExecution).filter(RecipeExecution.execution_id == first).one()
    looped.retry_of = _rerun(desk, first)
    desk.db.flush()
    assert agent_sends._launched_by_auto(desk.db, desk.ws, first) is True


def test_a_chats_composio_send_checked_against_the_intent_asks_the_owner(desk, monkeypatch):
    """The chat lane rides the same validation path (consumers/chatbot/service.py passes the
    turn's words as the intent): its driving user now reaches the gate, which asks."""
    import modules.tools.tool_router as tr

    composio = MagicMock(name="ComposioToolExecutor")
    composio.execute = AsyncMock(return_value=POSTED)
    monkeypatch.setattr(_Executor, "composio", composio)

    out = asyncio.run(tr.ToolRouter().execute_and_format(
        tool_name="composio_execute", tool_args={"action": LINKEDIN_POST, "params": IMAGES},
        agent_id=desk.poster.id, workspace_id=desk.ws, original_intent="post the launch",
        caller_context={"driving_user_id": "7", "conversation_id": "chat-1"},
    ))

    composio.execute.assert_not_called()
    assert out["success"] is False and out["llm_context"].startswith("Card raised: ")
    assert len(_grants(desk)) == 1


def test_a_step_meeting_the_card_a_chat_raised_still_waits_and_its_run_is_not_failed(desk):
    """P256-FIX-N17-3 (F439): the owner's chat raised the card for this exact post, then a step of
    Auto's run makes the same call. A step is no chat turn: it waits on that card (never replaces
    it, never 'already waiting'), so the step reads '(waiting)' and the run does not fail."""
    from api import recipe_executor as rex
    import modules.tools.tool_router as tr

    asyncio.run(tr.ToolRouter().execute_and_format(
        tool_name="composio_execute", tool_args={"action": LINKEDIN_POST, "params": IMAGES},
        agent_id=desk.poster.id, workspace_id=desk.ws, original_intent="post the launch",
        caller_context={"driving_user_id": "7", "conversation_id": str(uuid4()), "turn_id": "t-1"},
    ))
    (chats,) = _grants(desk)

    call = _run_step(desk, _auto_run(desk))

    desk.linkedin.assert_not_called()
    assert call["result"].startswith("Card raised: ") and call["success"] is False and call[WAITING] is True
    (grant,) = _grants(desk)
    assert grant.id == chats.id and grant.status == "pending" and grant.details[AGENT_SEND]["lane"] == "playbook"
    calls = rex._normalize_tool_calls([call])
    assert rex.step_failure(calls, {}) is None
    compact = rex._build_compact_step_result({"order": 2, "status": "completed", "tool_calls": calls})
    assert compact["tool_calls_summary"] == [f"{LINKEDIN_POST} (waiting)"]


# ── The seams ───────────────────────────────────────────────────────────────────────

def test_a_composio_call_checked_against_the_intent_keeps_the_callers_context(monkeypatch):
    """The chat's composio_execute and a step's action both ride this path."""
    import modules.tools.tool_router as tr

    seen = []

    async def execute_tool(tool_name, tool_args, agent_id=0, **kw):
        seen.append(kw.get("caller_context"))
        return {"success": True, "data": {}}

    monkeypatch.setattr(tr, "execute_tool", execute_tool)
    monkeypatch.setattr(tr, "composio_available", lambda: True)
    monkeypatch.setattr(tr, "validate_action_for_intent", lambda **kwargs: (True, ""))
    context = {"driving_user_id": "7", "playbook_execution_id": "exec-1"}

    asyncio.run(tr.ToolRouter().execute_and_format(
        tool_name="composio_execute", tool_args={"action": "GMAIL_SEND_EMAIL", "params": {}},
        agent_id=3, original_intent="send it", caller_context=context,
    ))

    assert seen == [context]


def test_the_held_context_is_the_inner_calls_and_none_outside_one():
    from modules.tools.execution.held_context import holds_the_callers_context, the_callers_context

    seen = []

    @holds_the_callers_context
    async def route(tool_name, caller_context=None, inner=None):
        seen.append(the_callers_context())
        if inner is not None:
            await route("inner", caller_context=inner)
            seen.append(the_callers_context())
        return {}

    asyncio.run(route("outer", caller_context={"session_task_id": 1}, inner={"board_task_id": 2}))

    assert seen == [{"session_task_id": 1}, {"board_task_id": 2}, {"session_task_id": 1}]
    assert the_callers_context() is None


def test_a_call_with_no_context_of_its_own_in_a_step_is_made_for_the_steps_run():
    from api.recipe_executor import step_context
    from modules.tools.execution.held_context import (
        acts_for_its_run, holds_the_callers_context, the_callers_context,
    )

    seen = []

    @holds_the_callers_context
    async def route(tool_name, caller_context=None):
        seen.append(the_callers_context())
        return {}

    @acts_for_its_run(lambda **step: step_context(step.get("recipe_execution_id"), step.get("step_order")))
    async def step(prompt, step_order=1, recipe_execution_id=None):
        await route("composio_execute")
        await route("composio_execute", caller_context={"playbook_execution_id": "exec-1", "playbook_step": 9})
        return {}

    asyncio.run(step("Post it", step_order=2, recipe_execution_id="exec-1"))
    asyncio.run(step("Post it"))                    # outside a run: for nobody, as before
    asyncio.run(route("composio_execute"))          # outside a step

    assert seen == [{"playbook_execution_id": "exec-1", "playbook_step": 2},
                    {"playbook_execution_id": "exec-1", "playbook_step": 9}, None,
                    {"playbook_execution_id": "exec-1", "playbook_step": 9}, None]


def test_the_upload_pass_leaves_an_image_posts_images_to_the_direct_api():
    import core.composio.tool_executor as tool_executor
    from core.composio.linkedin_image_workaround import leaves_its_images_to_the_direct_api

    resolved = []

    async def resolve(action, params, workspace_id):
        resolved.append(action)
        return {**params, "resolved": True}, []

    wrapped = leaves_its_images_to_the_direct_api(resolve)
    kept, temp = asyncio.run(wrapped(LINKEDIN_POST, IMAGES, "ws"))
    assert (kept, temp, resolved) == (IMAGES, [], [])
    asyncio.run(wrapped(LINKEDIN_POST, {"text": "No pictures today"}, "ws"))   # text-only: resolved as before
    asyncio.run(wrapped("TWITTER_UPLOAD_MEDIA", {"media": "/workspace/a.png"}, "ws"))
    assert resolved == [LINKEDIN_POST, "TWITTER_UPLOAD_MEDIA"]
    code = tool_executor.resolve_file_uploads.__code__
    assert code.co_qualname == "leaves_its_images_to_the_direct_api.<locals>.wrapped"


def _step_dispatch_lines():
    """Where a Playbook step hands a Composio action on: its upload pass and the spine."""
    lines = []
    for node in ast.walk(_function(*STEP)):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        spine = name == "execute_and_format" and any(
            kw.arg == "tool_name" and isinstance(kw.value, ast.Constant) and kw.value.value == "composio_execute"
            for kw in node.keywords
        )
        if name == "resolve_file_uploads" or spine:
            lines.append(node.lineno)
    return lines


def test_a_step_checks_the_deny_list_then_the_gate_before_its_uploads_and_the_spine():
    """The step no longer executes directly (no execution site in PRD-251's inventory), but
    its own checks still come first, in the same order as the agent site's (PRD-251 US-118)."""
    assert STEP not in _execution_sites()
    checks = sorted(
        (node.lineno, node.func.id) for node in ast.walk(_function(*STEP))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"composio_action_denial_async", "post_action_refusal"}
    )
    assert [name for _line, name in checks] == ["composio_action_denial_async", "post_action_refusal"], checks
    dispatched = _step_dispatch_lines()
    assert len(dispatched) == 2, dispatched
    assert checks[-1][0] < min(dispatched), (checks, dispatched)


# ── The step and the run read the card as waiting, never as failed (P256-FIX-T6, F401) ───

CARD = SEND_CARD_RAISED.format(subject="(no subject)", recipient="(no recipient named)")
ONE_STEP = {"step_id": "s1", "order": 1, "agent_id": 7, "error_handling": "stop", "max_retries": 0,
            "prompt_template": "Post the launch to LinkedIn."}


def test_a_refused_call_keeps_error_executing_and_is_not_waiting(desk, monkeypatch):
    import core.composio.deny_list as deny_list

    monkeypatch.setattr(deny_list, "composio_action_denial_async", AsyncMock(return_value="LinkedIn posts are denied"))
    call = _run_step(desk, _auto_run(desk))

    desk.linkedin.assert_not_called()
    assert call["result"] == f"Error executing {LINKEDIN_POST}: LinkedIn posts are denied"
    assert call["success"] is False and WAITING not in call
    assert _grants(desk) == []


def _step_records(*calls):
    """What ``_execute_step`` returns, through the T6 decorator, for a step that made these calls."""
    from api.recipe_step_outcome import records_the_wait

    async def step(**kwargs):
        return {"status": "success", "result": "Drafted the launch post; it waits for your click.",
                "execution": {"tokens_used": 900, "tool_calls": list(calls)}}

    return asyncio.run(records_the_wait(step)())


def test_the_step_records_the_card_text_itself_and_keeps_a_real_failure():
    raised = {"action": LINKEDIN_POST, "result": f"Error executing {LINKEDIN_POST}: {CARD}", "success": False,
              "composio_direct": True}
    failed = {"action": "GMAIL_SEND_EMAIL", "result": "Error executing GMAIL_SEND_EMAIL: 401 Unauthorized",
              "success": False, "composio_direct": True}
    told = {"action": "composio_execute", "result": CARD, "success": False}       # the generic branch's text
    posted = {"action": "LINKEDIN_GET_PROFILE", "result": "Card raised: nothing", "success": True}

    out = _step_records(raised, failed, told, posted)["execution"]["tool_calls"]

    assert out[0] == {**raised, "result": CARD, WAITING: True}
    assert out[1] == failed                                       # a real failure keeps "Error executing …"
    assert out[2] == {**told, WAITING: True}
    assert out[3] == posted                                       # a call that ran is never waiting


def test_a_step_whose_last_call_waits_is_not_failed_and_a_real_failure_still_is():
    from api import recipe_executor as rex

    calls = rex._normalize_tool_calls(_step_records(
        {"action": LINKEDIN_POST, "result": f"Error executing {LINKEDIN_POST}: {CARD}", "success": False},
    )["execution"]["tool_calls"])

    assert calls[0][WAITING] is True and calls[0]["result"] == CARD
    assert rex.step_failure(calls, {}) is None
    failed = [{"action": LINKEDIN_POST, "result": f"Error executing {LINKEDIN_POST}: 500", "success": False}]
    assert rex.step_failure(rex._normalize_tool_calls(failed), {}).startswith(
        f"its last tool call, {LINKEDIN_POST}, failed: Error executing")


def test_the_steps_record_says_waiting_and_names_the_card():
    from api import recipe_executor as rex

    compact = rex._build_compact_step_result({"order": 2, "status": "completed", "tool_calls": [
        {"action": "LINKEDIN_GET_PROFILE", "result": "{}", "success": True},
        {"action": LINKEDIN_POST, "result": CARD, "success": False, WAITING: True},
    ]})

    assert compact["tool_calls_summary"] == ["LINKEDIN_GET_PROFILE (success)", f"{LINKEDIN_POST} (waiting)"]
    assert compact[WAITING_ON] == [CARD]
    plain = rex._build_compact_step_result({"order": 1, "status": "completed", "tool_calls": []})
    assert WAITING_ON not in plain


def test_a_run_whose_step_waits_on_the_click_reads_as_waiting_not_failed(monkeypatch):
    from tests import helpers_playbook_run as hp

    outcome = _step_records({"action": LINKEDIN_POST, "result": f"Error executing {LINKEDIN_POST}: {CARD}",
                             "success": False, "composio_direct": True})
    execution, card = hp.run_playbook(monkeypatch, outcomes=[outcome], step_seconds=5, exec_config={},
                                      steps=[ONE_STEP])

    assert execution.status == "completed" and card.status == "done"
    (step,) = execution.step_results
    assert step["status"] == "completed" and step["retries"] == 0
    assert step["tool_calls_summary"] == [f"{LINKEDIN_POST} (waiting)"] and step[WAITING_ON] == [CARD]
    final = execution.output_data["final_output"]
    assert final.startswith("Drafted the launch post")
    assert f"{WAITING_HEADER}\n- Step 1: Card raised: send (no subject) to (no recipient named). " in final
    assert "end your turn" not in final                           # the model's instruction is not the owner's
    assert WAITING_HEADER in card.result


def test_a_run_whose_last_call_really_failed_still_fails(monkeypatch):
    from tests import helpers_playbook_run as hp

    outcome = _step_records({"action": LINKEDIN_POST, "result": f"Error executing {LINKEDIN_POST}: 500",
                             "success": False, "composio_direct": True})
    execution, _card = hp.run_playbook(monkeypatch, outcomes=[outcome], step_seconds=5, exec_config={},
                                       steps=[ONE_STEP])

    assert execution.status == "failed"
    assert execution.error_message.startswith(f"Step 1 failed: its last tool call, {LINKEDIN_POST}, failed")


def test_every_card_starts_with_the_words_the_step_reads():
    from api.recipe_step_outcome import CARD_TEXT, with_the_wait
    from modules.tools.execution.card_raised import FOR_THE_MODEL

    assert CARD_TEXT == "Card raised: "
    assert SEND_CARD_RAISED.startswith(CARD_TEXT) and FOR_THE_MODEL.startswith(CARD_TEXT)
    told = FOR_THE_MODEL.format(act="delete the document 'Q3 plan'")
    assert with_the_wait(None, [{"order": 3, WAITING_ON: [told]}]) == (
        f"{WAITING_HEADER}\n- Step 3: Card raised: delete the document 'Q3 plan'. Nothing happens until you click.")
    assert with_the_wait("Done.", [{"order": 1}]) == "Done."
