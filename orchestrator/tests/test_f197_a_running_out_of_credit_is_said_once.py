"""F197 (night 6) — running out of model credit is said once, in plain words, and
a scheduled report that failed on it runs again when credit is back.

From 05:42Z the provider refused calls for credit. The raw 402, carrying the
account's internal user id, went onto:
- tickets 1160 and 1162-1165 ("Task execution failed after 2 attempts: Error
  code: 402 - {'error': ...");
- Reports;
- the bell, 17 notices in 14 minutes;
- the failed Monday Stock Report ("Step 1 failed: Error code: 402 - {...").
Nothing said it once in plain words, and nothing ran again when credit returned.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
# Night 6's refusal, as OpenRouter wrote it (the account id is made up here).
RAW_402 = ("Error code: 402 - {'error': {'message': 'This request requires more credits, or fewer max_tokens. "
           "You requested up to 65535 tokens, but can only afford 8512. To increase, visit "
           "https://openrouter.ai/settings/credits and add more credits', 'code': 402, "
           "'metadata': {'provider_name': None}}, 'user_id': 'user_2mYFakeAccount0000'}")
IN_FLIGHT_402 = ("Error code: 402 - {'error': {'message': 'This request would exceed your available credits "
                 "given your current in-flight requests. Retry after in-flight requests complete.', 'code': 402}}")


@pytest.fixture(autouse=True)
def _fresh_outages():
    try:
        from core.llm import credit

        credit.reset()
    except ImportError:
        pass
    yield


# ── the ticket ─────────────────────────────────────────────────────────────

class _Session:
    def __init__(self, found=None):
        self.found, self.commits = found, 0

    def get(self, *a, **k):
        return self.found

    def query(self, model):
        return NS(filter=lambda *a, **k: NS(first=lambda: self.found if model.__name__ == "RecipeExecution" else None))

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


@pytest.mark.parametrize("raw", [RAW_402, IN_FLIGHT_402], ids=["more-credits", "in-flight"])
def test_a_ticket_that_ran_out_of_credit_says_so_in_plain_words(monkeypatch, raw):
    from api import board_tasks

    async def _nothing(*a, **k):
        return None

    monkeypatch.setattr(board_tasks, "_dispatch_task_failed", _nothing)
    monkeypatch.setattr(board_tasks, "_auto_create_task_report", _nothing)
    task = NS(id=1164, status="in_progress", error_message=None, completed_at=None)
    asyncio.run(board_tasks.finalize_board_task_run(
        _Session(task), task_id=1164, workspace_id=WS, agent_id=325,
        exec_result={"status": "error", "error": f"Task execution failed after 2 attempts: {raw}"}))

    assert task.status == "failed"
    assert task.error_message == ("The AI provider's account ran out of credit, so this stopped before it finished. "
                                  "Top up the provider account, then run it again.")


def test_another_provider_refusal_gives_its_status_not_its_payload():
    from core.llm.credit import plain_failure

    said = plain_failure("Error code: 500 - {'error': {'message': 'upstream', 'user_id': 'user_2mYFakeAccount0000'}}")
    assert said == "The AI provider refused the request (HTTP 500), so this stopped. The details are in the server log."
    assert plain_failure("the file was not found for user_2mYFakeAccount0000") == "the file was not found for user_…"


# ── the playbook ───────────────────────────────────────────────────────────

@pytest.fixture
def fail_run(monkeypatch):
    from api import recipe_executor

    async def _nothing(*a, **k):
        return None

    monkeypatch.setattr(recipe_executor, "_ingest_playbook_terminal_watch", lambda *a, **k: None)
    monkeypatch.setattr(recipe_executor, "_dispatch_playbook_event", _nothing)
    monkeypatch.setattr(recipe_executor, "_update_agent_performance_metrics", lambda *a, **k: None)
    import services.playbook_engine_heartbeat as heartbeat

    monkeypatch.setattr(heartbeat, "_emit_playbooks_primitive", lambda *a, **k: None)

    def run(triggered_by):
        execution = NS(execution_id="exec-77379d9b84c9", status="running", error_message=None, completed_at=None,
                       step_results=None, triggered_by=triggered_by, execution_metadata={"total_steps": 1},
                       workspace_id=WS, recipe_id=104, started_at=None)
        asyncio.run(recipe_executor._fail_execution(_Session(execution), execution.execution_id,
                                                    f"Step 1 failed: {IN_FLIGHT_402}"))
        return execution
    return run


def test_a_scheduled_report_that_ran_out_of_credit_is_marked_to_run_again(fail_run):
    execution = fail_run("cron_scheduler")

    assert execution.status == "failed"
    assert execution.error_message == ("The AI provider's account ran out of credit, so this stopped before it "
                                       "finished. It runs again by itself once credit is back.")
    assert execution.execution_metadata == {"total_steps": 1, "out_of_credit": True}


def test_a_run_someone_started_is_not_run_again_by_itself(fail_run):
    """Night 6's Monday Stock Report run was started by Auto (platform_action)."""
    execution = fail_run("platform_action")

    assert "Top up the provider account, then run it again." in execution.error_message
    assert "out_of_credit" not in execution.execution_metadata


# ── the bell ───────────────────────────────────────────────────────────────

@pytest.fixture
def bell(monkeypatch):
    from core.services.notification_dispatcher import NotificationDispatcher

    rung = []
    monkeypatch.setattr(NotificationDispatcher, "_load_auto_reporting", lambda self: {})
    monkeypatch.setattr(NotificationDispatcher, "_get_preferences", lambda self, *a, **k: [])
    monkeypatch.setattr(NotificationDispatcher, "_insert_in_app",
                        lambda self, user_id, event_type, title, message, *a, **k: rung.append((event_type, title,
                                                                                                message)))

    def ring(event_type, title, message, ws=WS):
        return asyncio.run(NotificationDispatcher(_Session(), ws).dispatch(event_type=event_type, title=title,
                                                                           message=message, status="error"))
    return NS(ring=ring, rung=rung)


def test_the_bell_says_it_once_per_outage(bell):
    plain = ("The AI provider's account ran out of credit, so this stopped before it finished. "
             "Top up the provider account, then run it again.")
    bell.ring("task_failed", "Task failed: Who owes me - wholesale invoices", plain)
    bell.ring("report_submitted", "Report: Task: Who owes me - wholesale invoices", plain)
    bell.ring("task_failed", "Task failed: Friendly payment reminders", f"Task execution failed: {RAW_402}")
    bell.ring("playbook_failed", "Playbook failed", "Step 1 failed: " + IN_FLIGHT_402)

    assert [title for _, title, _ in bell.rung] == ["Out of AI credit"]
    assert "user_" not in bell.rung[0][2] and "402" not in bell.rung[0][2]


def test_other_failures_and_other_workspaces_still_ring(bell):
    bell.ring("task_failed", "Task failed: A", "Out of credit is not this: the file was not found.")
    bell.ring("task_failed", "Task failed: B", RAW_402)
    bell.ring("task_failed", "Task failed: C", RAW_402, ws="00000000-0000-0000-0000-0000000000c1")

    assert [title for _, title, _ in bell.rung] == ["Task failed: A", "Out of AI credit", "Out of AI credit"]


# ── credit back ────────────────────────────────────────────────────────────

def test_the_first_call_that_works_after_an_outage_runs_the_marked_reports_once(monkeypatch):
    from core.llm import credit

    went = []

    async def rerun(ws):
        went.append(ws)

    monkeypatch.setattr(credit, "_rerun_marked", rerun)

    async def calls():
        credit.note_model_success(WS, reserved=8000)   # the first since start: marks from before a restart
        credit.first_notice_of_outage(WS)              # credit runs out
        credit.note_model_success(WS, reserved=8000)   # and comes back
        credit.note_model_success(WS, reserved=8000)   # an ordinary call
        await asyncio.sleep(0)

    asyncio.run(calls())
    assert went == [WS, WS]


def test_a_call_with_no_loop_running_leaves_the_outage_for_the_next_one(monkeypatch):
    from core.llm import credit

    credit.first_notice_of_outage(WS)
    credit.note_model_success(WS, reserved=8000)       # a sync call: nothing can be launched, nothing consumed
    assert credit.first_notice_of_outage(WS) is False


# ── F196 × F197: small calls fit under leftover credit ─────────────────────

LEFTOVER_402 = ("Error code: 402 - {'error': {'message': 'This request requires more credits, or fewer max_tokens. "
                "You requested up to 8000 tokens, but can only afford 4013', 'code': 402}}")
MSG = [{"role": "user", "content": "go"}]


class _Provider:
    def __init__(self, outcome):
        self.outcome = outcome

    async def generate_response(self, messages, tools=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return NS(content="ok", tool_calls=None, finish_reason="stop", usage=None)


def _call(max_tokens, outcome=None):
    """One model call for the workspace, reserving ``max_tokens``: refused when
    ``outcome`` is an exception."""
    from core.llm.clients.base import LLMConfig, LLMProvider
    from core.llm.manager import LLMManager

    config = LLMConfig(provider=LLMProvider.OPENROUTER, model="google/gemini-2.5-flash", max_tokens=max_tokens,
                       api_key="k")
    mgr = LLMManager(config=config, workspace_id=WS, agent_id=325)
    mgr.provider, mgr._track_usage = _Provider(outcome), (lambda *a, **k: None)

    async def run():
        try:
            await mgr.generate_response(MSG)
        except Exception:
            pass
        await asyncio.sleep(0)
    asyncio.run(run())


@pytest.fixture
def reruns(monkeypatch):
    from core.llm import credit

    went = []

    async def rerun(ws):
        went.append(ws)

    monkeypatch.setattr(credit, "_rerun_marked", rerun)
    return went


def test_a_digest_that_fits_the_leftover_credit_does_not_end_the_outage(bell, reruns):
    """TESTER: failure(8,000), success(1,024), failure rang the bell twice and
    spent the mark. Now it rings once, and the mark waits for a success of 8,000."""
    _call(8000, Exception(LEFTOVER_402))                                    # a ticket's run is refused
    bell.ring("task_failed", "Task failed: Who owes me", f"Task execution failed: {LEFTOVER_402}")
    _call(1024)                                                             # the digest fits under 4,013
    bell.ring("task_failed", "Task failed: Friendly payment reminders", f"Task execution failed: {LEFTOVER_402}")

    assert [title for _, title, _ in bell.rung] == ["Out of AI credit"] and reruns == []
    _call(8000)                                                             # credit is back
    assert reruns == [WS]


def test_after_a_restart_only_a_success_as_big_as_an_agent_run_looks_for_marks(reruns):
    _call(1024)
    assert reruns == []
    _call(8000)
    assert reruns == [WS]


def test_a_rerun_that_fails_on_credit_again_is_marked_again():
    from core.llm.credit import mark_for_rerun

    rerun = NS(triggered_by="credit_back", execution_metadata={"rerun_of": "cron-3f0c2b8e1d4a"})
    assert mark_for_rerun(rerun) is True and rerun.execution_metadata["out_of_credit"] is True


def test_an_agent_that_reserves_far_more_does_not_keep_the_outage_open(reruns):
    _call(65535, Exception(LEFTOVER_402))                                   # its own setting: 65,535
    _call(8000)                                                             # an ordinary agent run works again
    assert reruns == [WS]


# ── code review ────────────────────────────────────────────────────────────

def test_a_report_about_a_customers_insufficient_funds_rings_as_itself(bell):
    """Only the providers' own refusals and our sentence count as a credit failure."""
    bell.ring("report_submitted", "Report: Weekly accounts", "Two orders bounced for insufficient funds this week.")
    bell.ring("task_failed", "Task failed: Who owes me", RAW_402)

    assert [title for _, title, _ in bell.rung] == ["Report: Weekly accounts", "Out of AI credit"]


def test_a_marked_run_another_process_is_claiming_is_left_to_it(test_engine):
    """Two processes ending the same outage could each stage a rerun of the same
    run. The claim is a row lock now; the second process skips a claimed row."""
    import threading
    import uuid

    from sqlalchemy import text
    from sqlalchemy.orm import sessionmaker

    from core.llm import credit
    from core.models import WorkflowTemplate
    from core.models.core import RecipeExecution

    Session = sessionmaker(bind=test_engine)
    ws = str(uuid.uuid4())
    setup = Session()
    setup.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f197-lock')"), {"id": ws})
    recipe = WorkflowTemplate(template_id=f"f197-{uuid.uuid4().hex[:10]}", name="Monday Stock Report",
                              description="F197", workspace_id=ws, template_definition={"steps": []},
                              created_by="user_test", steps=[])
    setup.add(recipe)
    setup.flush()
    run_id = f"cron-{uuid.uuid4().hex[:12]}"
    setup.add(RecipeExecution(execution_id=run_id, recipe_id=recipe.id, workspace_id=ws, status="failed",
                              input_data={}, attempt_count=1, triggered_by="cron_scheduler",
                              execution_metadata={"out_of_credit": True}))
    setup.commit()
    claimer = Session()
    try:
        claimer.query(RecipeExecution).filter(RecipeExecution.execution_id == run_id).with_for_update().one()
        let_go = threading.Timer(2.0, claimer.rollback)      # the other process finishes its claim
        let_go.start()
        staged = credit._stage_reruns(ws)
        let_go.join()
        assert [r.retry_of for r in staged] == []
    finally:
        claimer.close()
        setup.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
        setup.commit()
        setup.close()
