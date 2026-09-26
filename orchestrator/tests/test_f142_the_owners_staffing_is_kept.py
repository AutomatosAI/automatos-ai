"""F142 (a)/(b)/(d2): when the owner names who does what, the plan keeps it.

25 Sep: "WRITER drafts, COUNTINGHOUSE checks every number, OPS lists the free
November slots and books nothing." The plan brought in other agents, made
WRITER pull it together, and stretched the work. Nothing carried the owner's
staffing: the goal was free text, and capability routing picked whoever
scored best.

Now:
- (a) a mission may be created or replanned with ``staffing``, one
  {agent, does} entry per named agent. It is resolved against the
  workspace's active agents. A name several agents share, or none has, is
  refused with the ids, never guessed. A caller cannot seed it through
  config.
- (b) the planner gives each named agent its work (``staffed_by``). A plan
  that drops or invents staffing is sent back, and no synthesis step is
  added. The coordinator pins each staffed task to its agent, as an
  approval edit pins.
- (d2) the owner's words for the work are copied verbatim into the task.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

ROSTER = [NS(id=58, name="WRITER", slug="writer", status="active"),
          NS(id=306, name="WRITER", slug="writer-2", status="active"),
          NS(id=307, name="COUNTINGHOUSE", slug="countinghouse", status="active"),
          NS(id=308, name="OPS", slug="ops", status="active")]
STAFFING = [{"agent_id": 306, "agent_name": "WRITER", "does": "drafts the club newsletter"},
            {"agent_id": 307, "agent_name": "COUNTINGHOUSE", "does": "checks every number"}]


# ── (a) resolving the owner's staffing ─────────────────────────────────────

def test_the_owners_staffing_is_resolved_by_id_slug_or_a_name_only_one_agent_has():
    from services.coordinator_service import resolve_staffing

    assert resolve_staffing([{"agent": "306", "does": "drafts the club newsletter"},
                             {"agent": "COUNTINGHOUSE", "does": "checks every number"}], ROSTER) == STAFFING
    assert resolve_staffing([{"agent": "writer-2", "does": "x"}], ROSTER)[0]["agent_id"] == 306


@pytest.mark.parametrize("staffing,refusal", [
    ([{"agent": "WRITER", "does": "drafts"}], "2 active agents are called 'WRITER' (ids 58, 306)"),
    ([{"agent": "BARISTA", "does": "makes coffee"}], "No active agent in this workspace is called 'BARISTA'"),
    ([{"agent": "OPS"}], "Each staffing entry names one agent and its work"),
    ([{"agent": "OPS", "does": "lists slots"}, {"agent": "308", "does": "books nothing"}], "named twice"),
])
def test_staffing_that_names_no_agent_or_several_is_refused(staffing, refusal):
    from services.coordinator_service import StaffingError, resolve_staffing

    with pytest.raises(StaffingError, match=refusal.replace("(", r"\(").replace(")", r"\)")):
        resolve_staffing(staffing, ROSTER)


def test_an_agent_named_like_a_role_word_is_still_that_agent():
    from services.coordinator_service import resolve_staffing

    only_one = [NS(id=9, name="Writer", slug="writer", status="active")]
    assert resolve_staffing([{"agent": "writer", "does": "drafts"}], only_one)[0]["agent_id"] == 9


def test_the_owners_words_are_kept_but_capped():
    from services.coordinator_service import STAFFING_DUTY_CHARS, resolve_staffing

    does = resolve_staffing([{"agent": "OPS", "does": "y" * (STAFFING_DUTY_CHARS + 50)}], ROSTER)[0]["does"]
    assert does == "y" * STAFFING_DUTY_CHARS


def _agents(db, ws, *names):
    from sqlalchemy import text

    return {name: db.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
                                  "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
                             {"n": name, "w": str(ws)}).scalar() for name in names}


def test_a_mission_keeps_the_staffing_it_was_given_and_never_a_callers_config(db_session, seed_workspace):
    from services.coordinator_service import CoordinatorService

    ws = UUID(seed_workspace())
    ids = _agents(db_session, ws, "COUNTINGHOUSE", "OPS")
    run = asyncio.run(CoordinatorService().create_mission(
        db=db_session, workspace_id=ws, goal="g", created_by="user_test",
        config={"async_planning": True, "staffing": [{"agent_id": 999, "agent_name": "X", "does": "anything"}]},
        staffing=[{"agent": "COUNTINGHOUSE", "does": "checks every number"}]))
    assert run.config["staffing"] == [{"agent_id": ids["COUNTINGHOUSE"], "agent_name": "COUNTINGHOUSE",
                                       "does": "checks every number"}]
    seeded = asyncio.run(CoordinatorService().create_mission(
        db=db_session, workspace_id=ws, goal="g", created_by="user_test",
        config={"async_planning": True, "staffing": [{"agent_id": ids["OPS"], "agent_name": "OPS", "does": "x"}]}))
    assert "staffing" not in seeded.config


def _ids(db, ws, name):
    from sqlalchemy import text

    return sorted(row[0] for row in db.execute(text("SELECT id FROM agents WHERE workspace_id = CAST(:w AS uuid) "
                                                    "AND name = :n"), {"w": str(ws), "n": name}))


def test_the_tool_says_which_agents_share_the_name(db_session, seed_workspace, monkeypatch):
    from modules.tools.discovery import handlers_missions as missions

    monkeypatch.setattr(missions, "_recent_chat_context", lambda *a, **k: [])
    ws = UUID(seed_workspace())
    _agents(db_session, ws, "WRITER")
    _agents(db_session, ws, "WRITER")
    writers = ", ".join(str(i) for i in _ids(db_session, ws, "WRITER"))
    reply = asyncio.run(missions.create_mission(db_session, ws, {
        "goal": "Draft the newsletter", "config": {"async_planning": True},
        "staffing": [{"agent": "WRITER", "does": "drafts the club newsletter"}]}))
    assert reply == {"success": False, "error": (
        f"2 active agents are called 'WRITER' (ids {writers}). Name the one you mean by its id.")}


def test_a_replan_that_names_no_agent_changes_nothing(db_session, seed_workspace):
    from core.models.orchestration import OrchestrationRun
    from services.coordinator_service import CoordinatorService, StaffingError

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="g", state="failed", created_by="user_test", config={})
    db_session.add(run)
    db_session.flush()
    with pytest.raises(StaffingError):
        asyncio.run(CoordinatorService().replan_mission(db_session, run.id, "user_test",
                                                        staffing=[{"agent": "BARISTA", "does": "x"}]))
    assert run.state == "failed"


# ── (b) the planner keeps it ────────────────────────────────────────────────

def _plan(*staffed_by, group="work"):
    return {"tasks": [{"temp_id": f"t{i}", "title": f"Task {i}", "description": "Do it.", "agent_role": "writer",
                       "staffed_by": who, "parallel_group": group, "definition_of_done": "Done."}
                      for i, who in enumerate(staffed_by, start=1)]}


def test_the_planner_is_told_who_does_what_and_a_plan_that_drops_it_is_sent_back(monkeypatch):
    from modules.coordination import planner

    prompts, answers = [], iter([_plan(306), _plan(306, 307)])

    class _LLM:
        async def generate_response(self, messages):
            prompts.append(messages[1]["content"])
            return NS(content=json.dumps(next(answers)))

    monkeypatch.setattr(planner, "create_llm_manager", lambda **kwargs: _LLM())
    monkeypatch.setattr(planner, "_validate_plan", lambda *a, **k: [])
    monkeypatch.setattr(planner, "_render_agent_roster", lambda agents: "(roster)")
    monkeypatch.setattr(planner, "match_template", lambda goal: NS(id="newsletter-template"))
    result = asyncio.run(planner.MissionPlanner.decompose(goal="Draft and check the newsletter", workspace_id=UUID(int=1),
                                                          agents=[], config={"staffing": STAFFING}))
    assert "## Who does what: the owner's choice" in prompts[0]
    assert '- COUNTINGHOUSE (id 307): "checks every number"' in prompts[0]
    assert "The owner named COUNTINGHOUSE (id 307)" in prompts[1]  # the first plan dropped it
    assert [task.staffed_by for task in result.tasks] == [306, 307]  # and no synthesis step was added
    assert result.template_used is None


def test_a_plan_cannot_staff_an_agent_the_owner_did_not_name():
    from modules.coordination.planner import PlannedTask, _staffing_errors

    def task(who):
        return PlannedTask(temp_id="t", title="Book the slots", description="", agent_role="scheduler",
                           sequence_number=1, task_type="llm_generation", verification_criteria=[],
                           required_tools=[], dependencies=[], staffed_by=who)

    assert _staffing_errors([task(306), task(307), task(308)], STAFFING) == [
        "Task 'Book the slots' has staffed_by 308, which is not an agent the owner named"]
    assert _staffing_errors([task(306)], STAFFING, every_named=False) == []


# ── (b)+(d2) the coordinator pins it, in the owner's words ─────────────────

def test_a_staffed_task_is_pinned_and_carries_the_owners_words(db_session, seed_workspace, monkeypatch):
    import services.coordinator_service as coordinator
    from core.models.orchestration import OrchestrationRun
    from modules.coordination.planner import DecompositionResult, PlannedTask

    monkeypatch.setattr(coordinator, "emit_event", lambda *a, **k: None)
    run = OrchestrationRun(workspace_id=UUID(seed_workspace()), goal="g", state="planning", created_by="user_test",
                           config={"staffing": STAFFING})
    db_session.add(run)
    db_session.flush()

    def planned(temp_id, who):
        return PlannedTask(temp_id=temp_id, title=f"Task {temp_id}", description="Check the draft's figures.",
                           agent_role="analyst", sequence_number=1, task_type="llm_generation",
                           verification_criteria=[], required_tools=[], dependencies=[], staffed_by=who)

    tasks = coordinator.CoordinatorService()._persist_decomposition(
        db_session, run, DecompositionResult(tasks=[planned("t1", 307), planned("t2", None)], dependencies=[],
                                             token_estimate=1000))
    checked, other = tasks["t1"], tasks["t2"]
    assert (checked.agent_role, checked.input_context["pinned_agent_id"]) == ("COUNTINGHOUSE", 307)
    assert checked.description == 'Check the draft\'s figures.\n\nThe owner\'s words for this work: "checks every number"'
    assert (other.agent_role, other.description) == ("analyst", "Check the draft's figures.")
    assert "pinned_agent_id" not in (other.input_context or {})
    assert [task.get("pinned_agent_id") for task in run.plan["tasks"]] == [307, None]


def test_the_rest_routes_bound_each_staffing_entry_and_pass_plain_entries_on(monkeypatch):
    """Each entry's agent and work are length-limited at the edge, and the
    coordinator gets plain {agent, does} entries (resolve_staffing reads them
    with .get); [] (clear the staffing, on a replan) stays []."""
    from fastapi import HTTPException
    from pydantic import ValidationError

    import api.missions as missions
    from services.coordinator_service import StaffingError

    with pytest.raises(ValidationError):
        missions.MissionCreateRequest(goal="g", staffing=[{"agent": "OPS", "does": "x" * 2001}])
    with pytest.raises(ValidationError):
        missions.MissionCreateRequest(goal="g", staffing=[{"agent": "OPS"}])

    passed = []

    class _Coordinator:
        async def create_mission(self, **kwargs):
            passed.append(kwargs["staffing"])
            raise StaffingError("No active agent in this workspace is called 'BARISTA'.")

    monkeypatch.setattr(missions, "get_coordinator_service", lambda: _Coordinator())
    ctx = NS(workspace_id=UUID(int=1), user=NS(id="user_test", clerk_user_id=None))
    body = missions.MissionCreateRequest(goal="g", staffing=[{"agent": "BARISTA", "does": "makes coffee"}])
    with pytest.raises(HTTPException) as refused:
        asyncio.run(missions.create_mission(body=body, ctx=ctx, db=NS(rollback=lambda: None)))
    assert refused.value.status_code == 422
    assert passed == [[{"agent": "BARISTA", "does": "makes coffee"}]]
    assert missions._staffing([]) == [] and missions._staffing(None) is None


def test_a_staffing_entry_may_name_its_agent_by_a_numeric_id():
    """The field says "name, slug or id": a bare JSON number is an id."""
    import api.missions as missions

    body = missions.MissionCreateRequest(goal="g", staffing=[{"agent": 306, "does": "drafts the club newsletter"}])
    assert missions._staffing(body.staffing) == [{"agent": "306", "does": "drafts the club newsletter"}]
