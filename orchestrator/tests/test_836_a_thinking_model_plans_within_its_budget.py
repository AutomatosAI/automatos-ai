"""#836: output budgets truncated thinking models, so every mission plan failed.

With the system LLM on anthropic/claude-sonnet-5.5 through OpenRouter, whose
thinking tokens count toward max_tokens, every plan was cut at the planner's
2,903-token budget (measured on a non-thinking model's answers), and the planner
said "LLM response did not contain valid JSON" three times: HTTP 422. A new
llm_output_budget row then looked ignored: a manager built on the event loop
read a cold per-purpose cache, used the table, and refreshed afterwards.

Now:
- a model the catalogue says thinks reserves its purpose's budget plus the
  thinking allowance (LLM_THINKING_ALLOWANCE_RATIO times it), capped by the
  model's ceiling; a settings row is the operator's whole reservation;
- the planner says the answer was cut, at how many tokens;
- each worker loads the rows and the thinking models before it serves, and
  keeps them fresh on a thread, so the first manager reads them.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from config import config
from core.llm import budget_snapshot
from core.llm.budget_snapshot import ModelFacts, Snapshot
from core.llm.clients.base import LLMConfig, LLMProvider

SONNET = "anthropic/claude-sonnet-5.5"
SONNET_CEILING = 64000
FLASH_LITE = "google/gemini-2.5-flash-lite"
PLANNER_TABLE = 2903
CUT_PLAN = '{"tasks": [{"temp_id": "task_1", "title": "Draft the café vis'


def _config(model=SONNET):
    return LLMConfig(provider=LLMProvider.OPENROUTER, model=model, max_tokens=8000, api_key="k")


def _snapshot(budgets=None, models=None):
    return Snapshot(budgets=budgets or {}, models=models or {}, loaded_at=time.monotonic())


THINKS = {SONNET: ModelFacts(thinks=True, ceiling=SONNET_CEILING)}


@pytest.fixture
def built(monkeypatch):
    """A planner manager built from settings on ``model``, against ``snapshot``."""
    from core.llm import manager as llm_manager

    def build(model=SONNET, snapshot=None, service_name="planner"):
        monkeypatch.setattr(budget_snapshot, "_snapshot", snapshot or _snapshot(models=THINKS))
        monkeypatch.setattr(llm_manager.LLMManager, "_load_config_from_settings",
                            lambda self, *a, **k: _config(model))
        mgr = llm_manager.LLMManager(service_name=service_name)
        mgr._track_usage = lambda *a, **k: None
        return mgr
    return build


# ── 1. a thinking model's budget carries its thinking ──────────────────────

def test_a_thinking_planner_reserves_the_answer_plus_its_thinking_allowance(built):
    mgr = built()
    expected = PLANNER_TABLE + int(PLANNER_TABLE * config.LLM_THINKING_ALLOWANCE_RATIO)
    assert mgr._service_budget == expected > PLANNER_TABLE                 # was 2,903
    assert mgr._call_budget() == expected


def test_a_model_that_does_not_think_keeps_the_measured_budget(built):
    assert built(model=FLASH_LITE)._service_budget == PLANNER_TABLE


def test_the_thinking_allowance_never_passes_the_models_ceiling(built):
    small = {SONNET: ModelFacts(thinks=True, ceiling=8192)}
    assert built(snapshot=_snapshot(models=small))._service_budget == 8192


def test_a_settings_row_is_the_operators_whole_reservation(built):
    row = _snapshot(budgets={"planner": 16000}, models=THINKS)              # the night's workaround
    assert built(snapshot=row)._service_budget == 16000


def test_a_long_deliverable_on_a_thinking_model_is_capped_by_its_ceiling(built):
    assert built()._long_budget == SONNET_CEILING


# ── 2. the planner says the plan was cut ───────────────────────────────────

class _CutProvider:
    """Answers every plan cut at the budget, as sonnet-5.5 did; keeps the prompts."""

    def __init__(self):
        self.prompts = []

    async def generate_response(self, messages, tools=None):
        self.prompts.append(messages[-1]["content"])
        return NS(content=CUT_PLAN, tool_calls=None, finish_reason="length", usage=None, streamed=False)


@pytest.fixture
def cut_planner(monkeypatch, built):
    from modules.coordination import planner

    mgr = built(model=FLASH_LITE)                                           # the table's 2,903
    mgr.provider = _CutProvider()
    monkeypatch.setattr(planner, "create_llm_manager", lambda **kwargs: mgr)
    monkeypatch.setattr(planner, "match_template", lambda goal: None)
    monkeypatch.setattr(planner, "record_error", lambda **kwargs: None)
    return mgr


def test_a_plan_cut_at_its_budget_is_reported_as_cut_not_as_no_json(cut_planner):
    from modules.coordination.planner import MissionPlanner, PlanValidationError

    with pytest.raises(PlanValidationError) as failed:
        asyncio.run(MissionPlanner.decompose(goal="Email each café the day and time I am visiting",
                                             workspace_id=UUID(int=1), agents=[]))
    assert "LLM output cut at 2,903 tokens" in str(failed.value)
    assert "did not contain valid JSON" not in str(failed.value)
    assert "output cut at 2,903 tokens" in cut_planner.provider.prompts[1]   # the retry is told why


def test_a_replan_cut_at_its_budget_is_reported_as_cut(cut_planner):
    from modules.coordination.planner import MissionPlanner, PlanValidationError

    with pytest.raises(PlanValidationError) as failed:
        asyncio.run(MissionPlanner.replan(goal="Email each café", workspace_id=UUID(int=1), agents=[],
                                          completed_outputs=[], failed_task_title="Draft the emails",
                                          failed_task_reason="cut"))
    assert failed.value.errors == [
        "LLM output cut at 2,903 tokens (the planner's output budget) before the plan was complete, "
        "so it held no valid JSON. Return a shorter plan as a single JSON object."]


def test_an_answer_that_is_not_cut_still_says_no_json():
    from modules.coordination.planner import NO_JSON_ERROR, _unreadable_plan_error

    assert _unreadable_plan_error(NS(content="Here is the plan.", finish_reason="stop")) == NO_JSON_ERROR


# ── 3. an override is in force on the first read ───────────────────────────

@pytest.fixture
def database(monkeypatch):
    """The two reads the snapshot makes, answering what ``rows`` holds now."""
    rows = {"budgets": {"planner": 16000}, "models": {}}
    monkeypatch.setattr(budget_snapshot, "_load_budgets", lambda: dict(rows["budgets"]))
    monkeypatch.setattr(budget_snapshot, "_load_thinking_models", lambda: dict(rows["models"]))
    monkeypatch.setattr(budget_snapshot, "_snapshot", None)
    return rows


def test_after_the_boot_load_the_first_manager_on_the_loop_uses_the_row(database, monkeypatch):
    from core.boot.startup_tasks import warm_output_budgets_on_startup
    from core.llm import manager as llm_manager

    monkeypatch.setattr(config, "LLM_OUTPUT_BUDGET_REFRESH_SECONDS", 0)
    monkeypatch.setattr(llm_manager.LLMManager, "_load_config_from_settings",
                        lambda self, *a, **k: _config(FLASH_LITE))

    async def boot_then_plan():
        await warm_output_budgets_on_startup()
        return llm_manager.LLMManager(service_name="planner")._service_budget

    assert asyncio.run(boot_then_plan()) == 16000                           # was 2,903 until a refresh


def test_every_worker_loads_the_budgets_before_it_serves():
    import inspect

    import main

    source = inspect.getsource(main._seed_semantic_embeddings)
    assert "await warm_output_budgets_on_startup()" in source


def test_a_new_row_is_in_force_within_the_refresh_interval_without_a_restart(database, monkeypatch):
    monkeypatch.setattr(config, "LLM_OUTPUT_BUDGET_REFRESH_SECONDS", 0.01)

    async def run():
        await budget_snapshot.warm()
        refresher = asyncio.create_task(budget_snapshot.keep_fresh())
        database["budgets"] = {"planner": 20000}                            # the operator adds a row
        for _ in range(200):
            if budget_snapshot.stored_budget("planner") == 20000:
                break
            await asyncio.sleep(0.01)
        refresher.cancel()
        return budget_snapshot.stored_budget("planner")

    assert asyncio.run(run()) == 20000


def test_a_failed_read_keeps_the_last_snapshot(database, monkeypatch):
    budget_snapshot.refresh()

    def down():
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(budget_snapshot, "_load_budgets", down)
    assert budget_snapshot.refresh().budgets == {"planner": 16000}
