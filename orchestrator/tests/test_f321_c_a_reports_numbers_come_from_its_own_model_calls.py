"""F321-C (night 9b, build 15) — a report's numbers are its run's own model calls,
said the same way on every report kind, and never contradict each other.

4 Oct 15:49:29Z: Auto started "Monday green stock" from chat (run
exec-45d8ac862a79, card #0102). Report f6d590ab said "LLM calls: 0 … Tokens
0 / 0 / 35294 … Cost $0.0000". In llm_usage, the run's 19 calls (agent 342,
anthropic/claude-sonnet-4: 307,554 in, 10,318 out, 317,872 total, $0.3928) were
booked as execution_id 'chat:dd0b6649-ffd8-4c82-a9e5-c84745c194d1', request_type
'chat': the run's task copied the chat's usage scope, and the scope wins over the
manager's own 'recipe'. The 35,294 was each step's LAST call (13,406 + 21,888).
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace as NS

WS = "febae41b-374b-4580-a5ef-f698bdd382e4"
RUN = "exec-45d8ac862a79"
CHAT = "chat:dd0b6649-ffd8-4c82-a9e5-c84745c194d1"
SONNET = "anthropic/claude-sonnet-4"
# The run's calls as llm_usage holds them, rolled up by model.
RUN_BY_MODEL = [NS(model_id=SONNET, calls=19, in_tok=307_554, out_tok=10_318, tot_tok=317_872,
                   cost=0.39279, sub_calls=0, errors=0)]
STEPS = [
    {"order": 1, "agent_name": "Shopify Inventory Watchdog", "status": "completed", "duration_ms": 12047,
     "tokens_used": 13406, "started_at": "2026-10-04T15:49:36.446524+00:00",
     "completed_at": "2026-10-04T15:49:48.494515+00:00", "output_preview": "Perfect! I can see the database …"},
    {"order": 2, "agent_name": "Shopify Inventory Watchdog", "status": "completed", "duration_ms": 98153,
     "tokens_used": 21888, "started_at": "2026-10-04T15:49:48.533323+00:00",
     "completed_at": "2026-10-04T15:51:26.686942+00:00", "output_preview": "The stock report has been generated …"},
]
# Three of the run's calls (row created_at, total_tokens): one in step 1, two in step 2.
RUN_CALLS = [NS(created_at=datetime(2026, 10, 4, 15, 49, 48, 493481), total_tokens=13406),
             NS(created_at=datetime(2026, 10, 4, 15, 50, 1, 706525), total_tokens=14218),
             NS(created_at=datetime(2026, 10, 4, 15, 51, 26, 674756), total_tokens=21888)]


class _Db:
    """llm_usage as the rollup and the per-step read see it."""

    def __init__(self, by_model, calls=()):
        self.by_model, self.calls = by_model, list(calls)

    def execute(self, statement, params):
        from services import report_metrics

        rows = self.calls if statement is report_metrics._RUN_CALLS_SQL else self.by_model
        return NS(fetchall=lambda: rows)


def _manager(tracking):
    from core.llm.manager import LLMManager

    manager = LLMManager.__new__(LLMManager)
    manager._tracking_ctx = tracking
    manager.config = NS(model=SONNET, provider=NS(value="openrouter"))
    manager.service_name = "orchestrator"
    manager._log_cost_audit = lambda **kwargs: None
    return manager


def test_a_run_auto_starts_from_chat_books_its_calls_to_the_run(monkeypatch):
    from api import recipe_executor as rex
    from core.llm import usage_tracker
    from core.llm.usage_context import usage_scope
    import services.playbook_run_refusal as refusal

    booked = []
    monkeypatch.setattr(usage_tracker.UsageTracker, "track", staticmethod(lambda **kw: booked.append(kw)))
    # The step agent's manager, as the step sets it: its agent, 'recipe', the run's id.
    agent = _manager({"agent_id": 342, "workspace_id": WS, "request_type": "recipe", "execution_id": RUN})

    async def _inner(recipe_execution_id, recipe_id, workspace_id, input_data, db_url=None):
        agent._track_usage(NS(usage={"input_tokens": 13_075, "output_tokens": 331}), time.monotonic())

    async def _not_refused(*args, **kwargs):
        return False

    monkeypatch.setattr(rex, "_execute_recipe_inner", _inner)
    monkeypatch.setattr(refusal, "refused_before_it_runs", _not_refused)

    async def chat_turn():
        with usage_scope(request_type="chat", execution_id=CHAT, agent_id=346):
            run = asyncio.create_task(rex.execute_recipe_direct(RUN, 115, WS, {}))
        await run

    asyncio.run(chat_turn())

    assert [(b["execution_id"], b["request_type"], b["agent_id"]) for b in booked] == [(RUN, "recipe", 342)]


def test_a_steps_helper_calls_are_booked_to_the_steps_agent_not_auto(monkeypatch):
    from core.llm import usage_tracker
    from core.llm.usage_context import usage_scope
    from services.playbook_usage import books_the_step_to_its_agent

    booked = []
    monkeypatch.setattr(usage_tracker.UsageTracker, "track", staticmethod(lambda **kw: booked.append(kw)))
    helper = _manager({})                                    # a tool-routing helper: no agent of its own

    async def _step(**kwargs):
        helper._track_usage(NS(usage={"input_tokens": 2025, "output_tokens": 585}), time.monotonic())
        return {"status": "success"}

    async def run():
        with usage_scope(request_type="recipe", execution_id=RUN, workspace_id=WS, inherit=False):
            await books_the_step_to_its_agent(_step)(agent=NS(id=342), clean_prompt="Generate a report")

    with usage_scope(request_type="chat", execution_id=CHAT, agent_id=346):
        asyncio.run(run())

    assert [(b["execution_id"], b["agent_id"]) for b in booked] == [(RUN, 342)]


def _report(monkeypatch, db):
    from services import playbook_report
    from services.report_service import ReportService

    filed = []

    async def _create(self, **kwargs):
        filed.append(kwargs)
        return {"success": True}

    monkeypatch.setattr(ReportService, "create_report", _create)
    execution = NS(started_at=datetime(2026, 10, 4, 15, 49, 28, 995045),
                   completed_at=datetime(2026, 10, 4, 15, 51, 26, 716798))
    asyncio.run(playbook_report.auto_create_playbook_report(
        db=db, workspace_id=WS, recipe=NS(id=115, name="Monday green stock"), recipe_execution_id=RUN,
        execution=execution, step_results=STEPS, total_duration_ms=110283, total_tokens=35294,
        final_output="The stock report has been generated and stored.", success=True))
    return filed[0]


def test_the_playbook_report_counts_every_call_of_the_run(monkeypatch):
    report = _report(monkeypatch, _Db(RUN_BY_MODEL, RUN_CALLS))

    assert "- LLM calls: 19" in report["content"]
    assert "- Tokens (in/out/total): 307554 / 10318 / 317872" in report["content"]
    assert "- Cost: $0.3928" in report["content"]
    assert "- **Step 1: Shopify Inventory Watchdog** — completed · 13406 tokens" in report["content"]
    assert "- **Step 2: Shopify Inventory Watchdog** — completed · 36106 tokens" in report["content"]
    assert report["summary"].startswith("2 steps · $0.3928 · 117721 ms")
    assert report["metrics"]["llm_calls"] == 19 and report["metrics"]["tokens_used"] == 317_872


def test_a_run_whose_calls_were_not_recorded_says_so_and_prints_no_zeros(monkeypatch):
    report = _report(monkeypatch, _Db([]))                    # f6d590ab: nothing booked under the run

    assert "- LLM calls: none recorded for this run" in report["content"]
    assert "- Tokens (in/out/total): not known: no model call was recorded for this run" in report["content"]
    assert "- Cost: not known: no model call was recorded for this run" in report["content"]
    assert "35294" not in report["content"] and "$0.0000" not in report["content"]
    assert "cost not recorded" in report["summary"] and "tokens not recorded" in report["content"]


def test_every_report_kind_says_a_session_costs_plan_usage_and_unknowns_in_words():
    from services.heartbeat_report import heartbeat_report_content
    from services.report_metrics import compute_execution_metrics
    from services.task_report import task_report_content

    on_plan = compute_execution_metrics(_Db([NS(model_id="claude-sonnet-5-5", calls=1, in_tok=5_344_489,
                                                out_tok=158_244, tot_tok=5_502_733, cost=0.0, sub_calls=1,
                                                errors=0)]), WS, agent_id=301)
    task = NS(title="Weekly roast plan", status="done", error_message=None)
    unbooked = compute_execution_metrics(_Db([]), WS, agent_id=342, extra={"reported_tokens": 8238})

    session_report = task_report_content(task, "ROASTER", "Done.", {"runtime": "cli"}, on_plan)
    heartbeat = heartbeat_report_content("Shopify Inventory Watchdog", {"status": "success"}, unbooked)

    assert "- Cost: plan usage (subscription), no dollar figure" in session_report
    assert "- LLM calls: 1" in session_report and "5344489 / 158244 / 5502733" in session_report
    assert "- LLM calls: none recorded for this run" in heartbeat
    assert "(the run reported 8238 in total)" in heartbeat and "$0.0000" not in heartbeat
