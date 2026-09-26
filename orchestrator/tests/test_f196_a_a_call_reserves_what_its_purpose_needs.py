"""F196 (night 6) — a model call reserves the output its purpose needs, not the
model's maximum.

From 05:42Z OpenRouter refused calls with 402. One read: "This request requires
more credits, or fewer max_tokens. You requested up to 65535 tokens, but can only
afford 8512". Between 06:11 and 06:40Z, 264 refusals asked for 8,000 tokens and
16 asked for 65,535. A service call reserved its settings category's 8,000, a
one-line classifier included. AgentFactory gave Auto, the system tier and any
agent without its own Max Output Tokens the model's ceiling (65,535 for
gemini-2.5-flash), and ignored the 8,000 in Settings.

Now a service manager reserves its purpose's measured budget. An agent reserves
its own Max Output Tokens, else 8,000, capped by the model's ceiling. A long
deliverable reserves 16,000. A call cut at its budget says so.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from types import SimpleNamespace as NS

import pytest

from core.llm.clients.base import LLMConfig, LLMProvider

AFFORDABLE_AT_06_11 = 8512
GEMINI_FLASH_CEILING = 65535


class _OpenRouterShaped:
    """A provider that builds its request exactly as OpenRouter's does and keeps
    the max_tokens it would have sent."""

    def __init__(self, config, reply=None):
        self.config, self.sent = config, []
        self.reply = reply or NS(content="ok", tool_calls=None, finish_reason="stop", usage=None, streamed=False)

    async def generate_response(self, messages, tools=None):
        from core.llm.clients.openai_compatible_client import OpenAICompatibleProvider

        stub = NS(config=self.config)
        self.sent.append(OpenAICompatibleProvider._base_kwargs(stub, messages)["max_tokens"])
        return self.reply


def _config(max_tokens=8000, ceiling=None):
    return LLMConfig(provider=LLMProvider.OPENROUTER, model="google/gemini-2.5-flash", max_tokens=max_tokens,
                     api_key="k", **({"output_ceiling": ceiling} if ceiling else {}))


@pytest.fixture
def settings_built(monkeypatch):
    """A manager built from settings: the category's max_tokens is 8,000 on night 6,
    and no llm_output_budget rows exist."""
    from core.llm import manager as llm_manager

    monkeypatch.setattr(llm_manager.LLMManager, "_load_config_from_settings", lambda self, *a, **k: _config())
    try:
        from core.llm import output_budget

        monkeypatch.setattr(output_budget, "_stored", lambda purpose: None)
    except ImportError:
        pass

    def build(service_name, **kwargs):
        mgr = llm_manager.LLMManager(service_name=service_name, **kwargs)
        mgr.provider = _OpenRouterShaped(mgr.config)
        mgr._track_usage = lambda *a, **k: None
        return mgr
    return build


def _ask(mgr, scope=None):
    from core.llm.usage_context import usage_scope

    async def run():
        with usage_scope(request_type=scope) if scope else _nothing():
            return await mgr.generate_response([{"role": "user", "content": "classify: hello"}])
    asyncio.run(run())
    return mgr.provider.sent[-1]


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ── the night's two reservations ────────────────────────────────────────────

def test_a_classifier_reserves_its_measured_budget_not_the_categorys_8000(settings_built):
    asked = _ask(settings_built("complexity_assessor"))
    assert asked == 1617 < AFFORDABLE_AT_06_11                          # p99 1,078 × 1.5


@pytest.fixture
def factory(monkeypatch):
    from core.llm import manager as llm_manager
    from modules.agents.factory.agent_factory import AgentFactory

    settings = {("orchestrator_llm", "provider"): "openrouter", ("orchestrator_llm", "model"): "google/gemini-2.5-flash",
                ("orchestrator_llm", "max_tokens"): "8000", ("system_llm", "provider"): "openrouter",
                ("system_llm", "model"): "google/gemini-2.5-flash", ("system_llm", "max_tokens"): "8000"}
    monkeypatch.setattr(llm_manager, "get_system_setting", lambda c, k, d=None: settings.get((c, k), d))

    class _Registry:
        def query(self, *a):
            return self

        def filter_by(self, **k):
            return self

        def first(self):
            return NS(context_window=1048576, max_output_tokens=GEMINI_FLASH_CEILING)

    made = AgentFactory.__new__(AgentFactory)
    made.db_session, made.logger = _Registry(), logging.getLogger("test_f196")
    return made


@pytest.mark.parametrize("tier", ["_get_default_llm_config_from_settings", "_get_system_llm_config_from_settings"])
def test_a_tier_reserves_its_settings_max_output_tokens_not_the_models_ceiling(factory, tier):
    assert getattr(factory, tier)()["max_tokens"] == 8000                # was 65,535


def test_an_agent_that_sets_nothing_reserves_an_agent_runs_budget(factory):
    from modules.agents.factory import agent_factory

    assert factory._output_budget(None, GEMINI_FLASH_CEILING) == 8000
    assert factory._output_budget(4000, GEMINI_FLASH_CEILING) == 4000     # its own setting wins
    assert factory._output_budget(100000, GEMINI_FLASH_CEILING) == GEMINI_FLASH_CEILING
    activate = inspect.getsource(agent_factory.AgentFactory.activate_agent)
    assert '"max_tokens": self._output_budget(agent_model_config.get("max_tokens"), ceiling)' in activate


def test_switching_an_agents_model_keeps_its_own_setting_and_never_stores_the_ceiling():
    from api import agent_endpoints

    source = inspect.getsource(agent_endpoints)
    assert '"max_tokens": request.get("max_tokens") or current_config.get("max_tokens")' in source
    assert "new_model.max_output_tokens)" not in source


# ── what keeps its budget ───────────────────────────────────────────────────

def test_chat_and_agent_runs_keep_their_configured_budget(settings_built):
    from core.llm.manager import LLMManager

    agent = LLMManager(config=_config(8000, GEMINI_FLASH_CEILING), agent_id=322)
    agent.provider, agent._track_usage = _OpenRouterShaped(agent.config), (lambda *a, **k: None)
    assert _ask(agent, scope="chat") == 8000
    assert _ask(agent, scope="heartbeat") == 8000     # an agent's heartbeat run is not the heartbeat service


def test_a_generic_manager_keeps_its_budget_inside_an_agent_run(settings_built):
    assert _ask(settings_built("orchestrator"), scope="board_task") == 8000
    assert _ask(settings_built("orchestrator")) == 1024                 # alone: p99 288


def test_graph_extraction_keeps_its_own_cap_of_at_least_2000(settings_built):
    """F051: cutting extraction's cap to 2,000 once lost whole JSON answers; its
    cap is its own and the table never lowers it."""
    from config import config
    from modules.knowledge.graph_extraction import _extraction_llm

    llm = _extraction_llm(settings_built("graph_extraction"))
    assert _ask(llm) == config.GRAPH_EXTRACTION_MAX_OUTPUT_TOKENS >= 2000


# ── long deliverables ───────────────────────────────────────────────────────

@pytest.mark.parametrize("ceiling, asked", [(GEMINI_FLASH_CEILING, 16000), (8192, 8192)], ids=["flash", "small-model"])
def test_a_mission_final_write_reserves_a_long_deliverables_budget(settings_built, ceiling, asked):
    from core.llm.manager import LLMManager
    from core.llm.output_budget import LONG_DELIVERABLE, output_purpose

    agent = LLMManager(config=_config(8000, ceiling), agent_id=325)
    agent.provider, agent._track_usage = _OpenRouterShaped(agent.config), (lambda *a, **k: None)
    with output_purpose(LONG_DELIVERABLE):
        assert _ask(agent, scope="mission") == asked


def test_a_synthesis_task_runs_as_a_long_deliverable():
    from services import coordinator_service

    source = inspect.getsource(coordinator_service)
    assert 'writes_the_deliverable = getattr(task, "task_type", None) == TaskType.SYNTHESIS.value' in source
    assert "output_purpose(LONG_DELIVERABLE) if writes_the_deliverable" in source


def test_a_cut_report_is_written_again_at_a_long_deliverables_budget():
    from core.llm import output_budget
    from modules.tools.execution.tool_loop import ToolLoopExecutor

    seen = []

    async def llm(messages, tools):
        seen.append((output_budget._purpose.get(), messages[-1]["role"]))
        return NS(content=None, tool_calls=[{"id": "c2", "function": {"name": "platform_execute",
                                                                      "arguments": "{}"}}], finish_reason="stop")

    cut = NS(finish_reason="length", tool_calls=[{"id": "c1", "function": {
        "name": "platform_execute", "arguments": '{"action": "platform_submit_report", "params": {"content": "## Wee'}}])
    loop = ToolLoopExecutor(llm_callback=llm, tool_callback=None, max_iterations=3)
    messages = [{"role": "user", "content": "Write the staff-meeting summary."}]
    retry = asyncio.run(loop._maybe_recover_truncated_args(cut, messages, [{}]))

    assert retry.finish_reason == "stop"
    assert seen == [(output_budget.LONG_DELIVERABLE, "user")]            # no "write it shorter" first


# ── a cut is never silent ──────────────────────────────────────────────────

def test_a_cut_answer_is_logged_and_flagged_and_its_final_writer_says_so(settings_built, caplog):
    from core.llm.manager import LLMManager
    from core.llm.output_budget import cut_note_for

    cut = NS(content="Here is the newsletter: …", tool_calls=None, finish_reason="length", usage=None, streamed=False)
    agent = LLMManager(config=_config(8000, GEMINI_FLASH_CEILING), agent_id=322)
    agent.provider, agent._track_usage = _OpenRouterShaped(agent.config, cut), (lambda *a, **k: None)
    with caplog.at_level(logging.WARNING, logger="core.llm.output_budget"):
        _ask(agent, scope="chat")

    assert "[F196] chat output cut at its 8,000-token budget (model google/gemini-2.5-flash)" in caplog.text
    assert cut.cut == 8000 and cut.content == "Here is the newsletter: …"      # the text is the caller's
    assert cut_note_for(cut) == "\n\n[Cut here: this answer reached its 8,000-token limit.]"


def test_the_chat_and_an_agent_run_add_the_note_to_the_finished_answer():
    from consumers.chatbot.service import StreamingChatService
    from modules.agents.factory import agent_factory

    cut = NS(content="Here is the newsletter: …", tool_calls=None, finish_reason="length", cut=8000)
    assert StreamingChatService._answer_additions(None, cut) == [
        "\n\n[Cut here: this answer reached its 8,000-token limit.]"]
    run = inspect.getsource(agent_factory.AgentFactory)
    assert run.index("Completed %d continuation(s)") < run.index("_cut = cut_note_for(response)")


def test_an_agent_answer_continued_after_a_cut_has_no_note_in_its_middle(settings_built):
    """Code review of the first build: the note went onto the first part, and the
    agent run's own continuation ("continue exactly where you left off") was
    appended after it."""
    from core.llm.manager import LLMManager
    from core.llm.output_budget import cut_note_for

    first = NS(content="Part one of the report", tool_calls=None, finish_reason="length", usage=None, streamed=False)
    agent = LLMManager(config=_config(8000, GEMINI_FLASH_CEILING), agent_id=325)
    agent.provider, agent._track_usage = _OpenRouterShaped(agent.config, first), (lambda *a, **k: None)
    _ask(agent, scope="board_task")
    first.content += ", and part two."                                          # the factory's continuation
    first.finish_reason = "stop"

    assert first.content == "Part one of the report, and part two." and cut_note_for(first) is None


def test_a_json_helpers_cut_inside_a_chat_turn_is_left_parseable(settings_built):
    """Code review: the note followed the lane (chat) where the budget followed
    the manager's own purpose. A cut entity list got a text note after its
    closing bracket, and the extractor's json.loads failed."""
    mgr = settings_built("entity_extraction")
    mgr.provider.reply = NS(content='[{"name": "Harbourline"}]', tool_calls=None, finish_reason="length", usage=None)
    _ask(mgr, scope="chat")
    assert mgr.provider.reply.content == '[{"name": "Harbourline"}]' and mgr.provider.reply.cut == 2457


def test_a_budget_setting_is_never_read_on_the_event_loop(monkeypatch):
    """Code review: a stale cache read system_settings on the loop, once a
    minute (F105's shape). It is read on a thread; the table answers meanwhile."""
    import threading

    from core.llm import manager as llm_manager
    from core.llm import output_budget

    on_loop_thread = []

    def read(category, key):
        on_loop_thread.append(threading.current_thread() is threading.main_thread())
        return None

    monkeypatch.setattr(llm_manager, "read_system_setting", read)
    monkeypatch.setattr(output_budget, "_stored_cache", {})
    monkeypatch.setattr(output_budget, "_refreshing", set(), raising=False)
    monkeypatch.setattr(llm_manager.LLMManager, "_load_config_from_settings", lambda self, *a, **k: _config())

    async def build():
        mgr = llm_manager.LLMManager(service_name="digest")
        for _ in range(200):
            if len(on_loop_thread) >= 2:
                break
            await asyncio.sleep(0.01)
        return mgr

    mgr = asyncio.run(build())
    assert on_loop_thread and not any(on_loop_thread)
    assert mgr._service_budget == 1024
